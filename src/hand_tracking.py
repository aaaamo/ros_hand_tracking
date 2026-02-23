#!/usr/bin/env python3
"""ROS node for hand tracking and gesture recognition using MediaPipe."""

import sys
import os

# Ensure src/ directory is on Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf.transformations
import cv2 as cv

from hand_tracker import HandTracker
from cvfpscalc import CvFpsCalc


# ===========================================================
# Drawing gesture sequence configuration  (edit as needed)
# ===========================================================
DRAW_START_GESTURE  = "Pointing_Up"   # begin draw session
DRAW_ACTIVE_GESTURE = "Victory"       # collect path while this gesture is held
DRAW_END_GESTURE    = "Open_Palm"     # finish session and publish collected path
DRAW_LANDMARK_IDX   = 8              # which landmark to trace (8 = index fingertip)
# ===========================================================


class HandTrackingNode:

    def __init__(self):
        rospy.init_node("hand_tracking", anonymous=True)

        # Parameters
        image_topic = rospy.get_param(
            "~subscribe_image_topic", "/camera/color/image_raw"
        )
        landmarks_topic = rospy.get_param(
            "~publish_landmarks_topic", "/hand_tracking/landmarks"
        )
        gesture_topic = rospy.get_param(
            "~publish_gesture_topic", "/hand_tracking/gesture"
        )
        model_path = rospy.get_param("~model_path", "")
        num_hands = rospy.get_param("~num_hands", 1)
        min_detection = rospy.get_param("~min_detection_confidence", 0.7)
        min_tracking = rospy.get_param("~min_tracking_confidence", 0.5)
        self.show_image = rospy.get_param("~show_image", True)
        enable_gesture = rospy.get_param("~enable_gesture", False)

        # Resolve default model path relative to package
        if not model_path:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if enable_gesture:
                model_path = os.path.join(pkg_dir, "model", "gesture_recognizer.task")
            else:
                model_path = os.path.join(pkg_dir, "model", "hand_landmarker.task")

        rospy.loginfo("Loading model from: %s (gesture=%s)", model_path, enable_gesture)

        self.tracker = HandTracker(
            model_path=model_path,
            num_hands=num_hands,
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
            enable_gesture=enable_gesture,
        )

        self.bridge = CvBridge()
        self.fps_calc = CvFpsCalc(buffer_len=10)

        # Publishers
        self.landmarks_pub = rospy.Publisher(landmarks_topic, String, queue_size=10)
        if enable_gesture:
            self.gesture_pub = rospy.Publisher(gesture_topic, String, queue_size=10)
        else:
            self.gesture_pub = None

        # Subscriber
        self.image_sub = rospy.Subscriber(image_topic, Image, self.callback)

        # 3D projection parameters
        transform_topic = rospy.get_param(
            "~transform_topic", "/hand_tracking/camera_transform"
        )
        landmarks_3d_topic = rospy.get_param(
            "~publish_landmarks_3d_topic", "/hand_tracking/landmarks_3d"
        )
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )
        depth_topic = rospy.get_param(
            "~depth_topic", "/camera/depth/image_rect_raw"
        )
        self.use_depth = rospy.get_param("~use_depth", False)
        self.fixed_depth = rospy.get_param("~fixed_depth", 0.5)

        # Camera intrinsics (overwritten when CameraInfo arrives)
        self.fx = rospy.get_param("~camera_fx", 600.0)
        self.fy = rospy.get_param("~camera_fy", 600.0)
        self.cx = rospy.get_param("~camera_cx", 320.0)
        self.cy = rospy.get_param("~camera_cy", 240.0)

        # State
        self.latest_transform = None  # geometry_msgs/TransformStamped
        self.latest_depth = None      # numpy array (mm, uint16)

        # 3D publisher
        self.landmarks_3d_pub = rospy.Publisher(
            landmarks_3d_topic, String, queue_size=10
        )

        # Draw path publisher
        draw_path_topic = rospy.get_param(
            "~publish_draw_path_topic", "/hand_tracking/draw_path"
        )
        self.draw_path_pub = rospy.Publisher(draw_path_topic, String, queue_size=10)

        # Draw sequence state
        self._draw_running = False
        self._draw_path = []  # list of [x, y, z] world coords

        # 3D-related subscribers
        self.transform_sub = rospy.Subscriber(
            transform_topic, TransformStamped, self._transform_cb
        )
        self.camera_info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self._camera_info_cb
        )
        if self.use_depth:
            self.depth_sub = rospy.Subscriber(
                depth_topic, Image, self._depth_cb
            )

        rospy.loginfo("HandTrackingNode ready. Subscribed to: %s", image_topic)
        rospy.loginfo("3D projection: transform=%s  use_depth=%s  fixed_depth=%.2f m",
                      transform_topic, self.use_depth, self.fixed_depth)

    # ------------------------------------------------------------------
    # 3D-related callbacks
    # ------------------------------------------------------------------

    def _transform_cb(self, msg):
        self.latest_transform = msg

    def _camera_info_cb(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.camera_info_sub.unregister()  # only need once

    def _depth_cb(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            rospy.logerr("Depth CvBridge error: %s", e)

    def _pixel_to_3d(self, px, py):
        """Back-project pixel (px, py) to 3D world coords using latest transform.

        Returns [x, y, z] in world frame, or None if transform not available.
        """
        if self.latest_transform is None:
            return None

        # Depth
        if self.use_depth and self.latest_depth is not None:
            ix, iy = int(px), int(py)
            if (0 <= iy < self.latest_depth.shape[0] and
                    0 <= ix < self.latest_depth.shape[1]):
                d = float(self.latest_depth[iy, ix]) / 1000.0  # mm → m
                depth = d if d > 0.0 else self.fixed_depth
            else:
                depth = self.fixed_depth
        else:
            depth = self.fixed_depth

        # Back-project to camera frame
        x_cam = (px - self.cx) / self.fx * depth
        y_cam = (py - self.cy) / self.fy * depth
        z_cam = depth

        # Apply camera→world transform
        t = self.latest_transform.transform
        q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        mat = tf.transformations.quaternion_matrix(q)  # 4x4 rotation
        mat[0, 3] = t.translation.x
        mat[1, 3] = t.translation.y
        mat[2, 3] = t.translation.z

        p_world = mat @ np.array([x_cam, y_cam, z_cam, 1.0])
        return [round(float(v), 4) for v in p_world[:3]]

    def _update_draw_state(self, gesture, point_3d):
        """Update draw sequence state machine.

        Args:
            gesture:  gesture string of the tracked hand (may be None)
            point_3d: [x, y, z] world coords of DRAW_LANDMARK_IDX (may be None)
        """
        if not self._draw_running:
            if gesture == DRAW_START_GESTURE:
                self._draw_running = True
                self._draw_path = []
                rospy.loginfo("Draw sequence: START")
        else:
            if gesture == DRAW_ACTIVE_GESTURE:
                if point_3d is not None:
                    self._draw_path.append(point_3d)
            elif gesture == DRAW_END_GESTURE:
                rospy.loginfo("Draw sequence: END — publishing %d points", len(self._draw_path))
                self.draw_path_pub.publish(json.dumps(self._draw_path))
                self._draw_running = False
                self._draw_path = []

    def callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        cv_image = cv.flip(cv_image, 1)  # Mirror display
        hands = self.tracker.detect(cv_image)

        # Publish landmarks as JSON
        if hands:
            json_str = HandTracker.hands_to_json(hands)
            self.landmarks_pub.publish(json_str)

            # Publish gesture separately (first hand's gesture)
            if self.gesture_pub and "gesture" in hands[0]:
                self.gesture_pub.publish(hands[0]["gesture"])

            # Publish 3D landmarks + drive draw state machine
            if self.latest_transform is not None:
                hands_3d = []
                for hand in hands:
                    lm3d = [
                        self._pixel_to_3d(px, py)
                        for px, py in hand["pixel_landmarks"]
                    ]
                    entry = {"handedness": hand["handedness"], "landmarks_3d": lm3d}
                    if "gesture" in hand:
                        entry["gesture"] = hand["gesture"]
                    hands_3d.append(entry)
                self.landmarks_3d_pub.publish(json.dumps(hands_3d))

                # Draw sequence: use first hand only
                first = hands_3d[0]
                gesture = first.get("gesture")
                point_3d = first["landmarks_3d"][DRAW_LANDMARK_IDX]
                self._update_draw_state(gesture, point_3d)

        # Debug visualization
        if self.show_image:
            debug_image = cv_image.copy()
            self.tracker.draw(debug_image, hands)
            fps = self.fps_calc.get()
            cv.putText(
                debug_image,
                "FPS: {:.1f}".format(fps),
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                debug_image,
                "FPS: {:.1f}".format(fps),
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
            cv.imshow("Hand Tracking", debug_image)
            cv.waitKey(1)

    def shutdown(self):
        self.tracker.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = HandTrackingNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
