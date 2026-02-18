#!/usr/bin/env python3
"""MediaPipe HandLandmarker wrapper for ROS hand tracking."""

import copy
import json
import cv2 as cv
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandTracker:
    """Wraps MediaPipe HandLandmarker (Tasks API) for per-frame hand tracking."""

    # Connections between landmarks for drawing skeleton
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17),
    ]

    FINGERTIP_INDICES = {4, 8, 12, 16, 20}

    def __init__(self, model_path, num_hands=1,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, bgr_image):
        """Run hand landmark detection on a BGR image.

        Args:
            bgr_image: OpenCV BGR image (numpy array).

        Returns:
            List of dicts, each containing:
                - 'handedness': 'Left' or 'Right'
                - 'landmarks': list of 21 dicts with 'x', 'y', 'z' (normalized)
                - 'pixel_landmarks': list of 21 [px, py] in image coordinates
        """
        rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        self._timestamp_ms += 33  # ~30fps increment
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands = []
        if not result.hand_landmarks:
            return hands

        h, w = bgr_image.shape[:2]
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name

            norm_landmarks = []
            pixel_landmarks = []
            for lm in landmarks:
                norm_landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                px = min(int(lm.x * w), w - 1)
                py = min(int(lm.y * h), h - 1)
                pixel_landmarks.append([px, py])

            hands.append({
                'handedness': handedness,
                'landmarks': norm_landmarks,
                'pixel_landmarks': pixel_landmarks,
            })

        return hands

    def draw(self, image, hands):
        """Draw hand landmarks and skeleton on image.

        Args:
            image: BGR image to draw on (modified in-place).
            hands: Output from detect().

        Returns:
            The annotated image.
        """
        for hand in hands:
            pts = hand['pixel_landmarks']

            # Draw skeleton connections
            for a, b in self.HAND_CONNECTIONS:
                cv.line(image, tuple(pts[a]), tuple(pts[b]), (0, 0, 0), 6)
                cv.line(image, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)

            # Draw keypoints
            for idx, pt in enumerate(pts):
                r = 8 if idx in self.FINGERTIP_INDICES else 5
                cv.circle(image, (pt[0], pt[1]), r, (255, 255, 255), -1)
                cv.circle(image, (pt[0], pt[1]), r, (0, 0, 0), 1)

            # Draw bounding rect + handedness label
            brect = self._calc_bounding_rect(pts)
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)
            cv.rectangle(image, (brect[0], brect[1]),
                         (brect[2], brect[1] - 22), (0, 0, 0), -1)
            cv.putText(image, hand['handedness'],
                       (brect[0] + 5, brect[1] - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)

        return image

    def close(self):
        self.landmarker.close()

    @staticmethod
    def hands_to_json(hands):
        """Serialize hand data to JSON string for ROS publishing."""
        return json.dumps(hands)

    @staticmethod
    def _calc_bounding_rect(pixel_landmarks):
        arr = np.array(pixel_landmarks)
        x, y, w, h = cv.boundingRect(arr)
        return [x, y, x + w, y + h]
