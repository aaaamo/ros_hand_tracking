#!/usr/bin/env python3
"""Draw gesture sequence state machine (ROS node version)."""

import json
import rospy
from geometry_msgs.msg import Point, Pose, PoseArray


# ===========================================================
# Drawing gesture sequence configuration  (edit as needed)
# ===========================================================
DRAW_START_GESTURE = "Victory"  # begin draw session (trained gesture)
DRAW_END_GESTURE = "Thumb_Up"  # finish session and publish (trained gesture)
DRAW_LANDMARK_IDX = 8  # landmark to trace (8 = index fingertip)
PINCH_THRESHOLD = 0.08  # normalised distance threshold for pinch
#   active draw = thumb tip (4) and index tip (8) within PINCH_THRESHOLD
# ===========================================================


def is_pinch(landmarks):
    """Return True when thumb tip (4) and index tip (8) are pinched together."""
    t, i = landmarks[4], landmarks[8]
    dx, dy = t["x"] - i["x"], t["y"] - i["y"]
    return (dx * dx + dy * dy) ** 0.5 < PINCH_THRESHOLD


class DrawSequence:
    """Two-state draw sequence machine.

    idle  → (DRAW_START_GESTURE) → running
    running + pinch              → collect point
    running + DRAW_END_GESTURE   → publish via publish_fn, back to idle
    """

    def __init__(self, publish_fn):
        """
        Args:
            publish_fn: callable(json_str) — called with the completed path JSON.
        """
        self.running = False
        self._path = []
        self._publish = publish_fn

    @property
    def path(self):
        return self._path
    
    def get_posearray(self) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "base_link"
        for p in self._path:
            if p is None or len(p) != 3:
                continue
            pose = Pose()
            pose.position = Point(x=p[0], y=p[1], z=p[2])
            pose_array.poses.append(pose)
        return pose_array

    def update(self, gesture, point_3d, landmarks):
        """Call once per frame with the first hand's data.

        Args:
            gesture:   gesture string (may be None)
            point_3d:  [x, y, z] world coords of DRAW_LANDMARK_IDX (may be None)
            landmarks: list of 21 normalized landmark dicts {'x', 'y', 'z'}
        """
        if not self.running:
            if gesture == DRAW_START_GESTURE:
                self.running = True
                self._path = []
        else:
            # End gesture is checked first so Closed_Fist always wins over pinch
            if gesture == DRAW_END_GESTURE:
                # self._publish(json.dumps(self._path))
                pose_array = self.get_posearray()
                if pose_array.poses:
                    self._publish(pose_array)
                self.running = False
                self._path = []
            elif is_pinch(landmarks):
                if point_3d is not None:
                    self._path.append(point_3d)
