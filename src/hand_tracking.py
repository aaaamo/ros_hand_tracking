#!/usr/bin/env python3
"""ROS node for hand tracking and gesture recognition using MediaPipe."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import rospy
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from hand_tracker import HandTracker
from cvfpscalc import CvFpsCalc
from projection_3d import Projection3D
from draw_sequence import DrawSequence, DRAW_LANDMARK_IDX


class HandTrackingNode:

    def __init__(self):
        rospy.init_node("hand_tracking", anonymous=True)

        # --- Core tracker params ---
        image_topic    = rospy.get_param("~subscribe_image_topic", "/camera/color/image_raw")
        landmarks_topic = rospy.get_param("~publish_landmarks_topic", "/hand_tracking/landmarks")
        gesture_topic  = rospy.get_param("~publish_gesture_topic",   "/hand_tracking/gesture")
        model_path     = rospy.get_param("~model_path", "")
        num_hands      = rospy.get_param("~num_hands", 1)
        min_detection  = rospy.get_param("~min_detection_confidence", 0.7)
        min_tracking   = rospy.get_param("~min_tracking_confidence",  0.5)
        self.show_image = rospy.get_param("~show_image", True)
        enable_gesture = rospy.get_param("~enable_gesture", False)

        if not model_path:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_file = "gesture_recognizer.task" if enable_gesture else "hand_landmarker.task"
            model_path = os.path.join(pkg_dir, "model", model_file)

        rospy.loginfo("Loading model: %s  gesture=%s", model_path, enable_gesture)

        self.tracker   = HandTracker(model_path, num_hands, min_detection, min_tracking, enable_gesture)
        self.bridge    = CvBridge()
        self.fps_calc  = CvFpsCalc(buffer_len=10)

        # --- Publishers ---
        self.landmarks_pub = rospy.Publisher(landmarks_topic, String, queue_size=10)
        self.gesture_pub   = rospy.Publisher(gesture_topic,   String, queue_size=10) if enable_gesture else None

        # --- 3D projection ---
        transform_topic     = rospy.get_param("~transform_topic",           "/hand_tracking/camera_transform")
        landmarks_3d_topic  = rospy.get_param("~publish_landmarks_3d_topic","/hand_tracking/landmarks_3d")
        camera_info_topic   = rospy.get_param("~camera_info_topic",         "/camera/color/camera_info")
        depth_topic         = rospy.get_param("~depth_topic",               "/camera/depth/image_rect_raw")

        self.proj = Projection3D(
            bridge=self.bridge,
            use_depth=rospy.get_param("~use_depth", False),
            fixed_depth=rospy.get_param("~fixed_depth", 0.5),
            fx=rospy.get_param("~camera_fx", 600.0),
            fy=rospy.get_param("~camera_fy", 600.0),
            cx=rospy.get_param("~camera_cx", 320.0),
            cy=rospy.get_param("~camera_cy", 240.0),
        )
        self.landmarks_3d_pub = rospy.Publisher(landmarks_3d_topic, String, queue_size=10)

        # --- Draw sequence ---
        draw_path_topic = rospy.get_param("~publish_draw_path_topic", "/hand_tracking/draw_path")
        self.draw_path_pub = rospy.Publisher(draw_path_topic, String, queue_size=10)
        self.draw_seq = DrawSequence(publish_fn=self.draw_path_pub.publish)

        # --- Subscribers ---
        rospy.Subscriber(image_topic,       Image,            self.callback)
        rospy.Subscriber(transform_topic,   TransformStamped, self.proj.on_transform)
        rospy.Subscriber(camera_info_topic, CameraInfo,       self._camera_info_cb)
        if self.proj.use_depth:
            rospy.Subscriber(depth_topic, Image, self.proj.on_depth)

        rospy.loginfo("HandTrackingNode ready. image=%s  transform=%s  use_depth=%s",
                      image_topic, transform_topic, self.proj.use_depth)

    def _camera_info_cb(self, msg):
        self.proj.on_camera_info(msg)
        # unregister after first message
        self._cam_info_sub = getattr(self, "_cam_info_sub", None)

    def callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        cv_image = cv.flip(cv_image, 1)
        hands = self.tracker.detect(cv_image)

        if hands:
            self.landmarks_pub.publish(HandTracker.hands_to_json(hands))
            if self.gesture_pub and "gesture" in hands[0]:
                self.gesture_pub.publish(hands[0]["gesture"])

            if self.proj.ready:
                hands_3d = []
                for hand in hands:
                    lm3d = [self.proj.pixel_to_3d(px, py) for px, py in hand["pixel_landmarks"]]
                    entry = {"handedness": hand["handedness"], "landmarks_3d": lm3d}
                    if "gesture" in hand:
                        entry["gesture"] = hand["gesture"]
                    hands_3d.append(entry)
                self.landmarks_3d_pub.publish(json.dumps(hands_3d))

                first = hands_3d[0]
                self.draw_seq.update(
                    gesture=first.get("gesture"),
                    point_3d=first["landmarks_3d"][DRAW_LANDMARK_IDX],
                    landmarks=hands[0]["landmarks"],
                )

        if self.show_image:
            debug_image = cv_image.copy()
            self.tracker.draw(debug_image, hands)
            fps = self.fps_calc.get()
            cv.putText(debug_image, "FPS: {:.1f}".format(fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(debug_image, "FPS: {:.1f}".format(fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
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
