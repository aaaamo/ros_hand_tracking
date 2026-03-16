#!/usr/bin/env python3
"""ROS node for hand tracking and gesture recognition using MediaPipe."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Point, PoseArray
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
        landmarks_3d_topic  = rospy.get_param("~publish_landmarks_3d_topic","/hand_tracking/landmarks_3d")
        camera_info_topic   = rospy.get_param("~camera_info_topic",         "/roi/aligned_depth_to_color/camera_info")
        depth_topic         = rospy.get_param("~depth_topic",               "/roi/depth/image_rect_raw")

        self.proj = Projection3D(
            bridge=self.bridge,
            use_depth=rospy.get_param("~use_depth", False),
            fixed_depth=rospy.get_param("~fixed_depth", 0.5),
            fx=rospy.get_param("~camera_fx", 600.0),
            fy=rospy.get_param("~camera_fy", 600.0),
            cx=rospy.get_param("~camera_cx", 320.0),
            cy=rospy.get_param("~camera_cy", 240.0),
        )
        self.landmarks_3d_pub = rospy.Publisher(landmarks_3d_topic, PoseArray, queue_size=10)

        # --- Draw sequence ---
        draw_path_topic = rospy.get_param("~publish_draw_path_topic", "/hand_tracking/draw_path")
        draw_path_2d_topic = rospy.get_param("~publish_draw_path_2d_topic", "/hand_tracking/draw_path_2d")
        self.draw_path_pub = rospy.Publisher(draw_path_topic, PoseArray, queue_size=10)
        self.draw_path_2d_pub = rospy.Publisher(draw_path_2d_topic, PoseArray, queue_size=10)
        self.draw_seq = DrawSequence(
            publish_fn=self.draw_path_pub.publish,
            publish_2d_fn=self.draw_path_2d_pub.publish,
        )

        # --- Subscribers ---
        rospy.Subscriber(image_topic,       Image,      self.callback)
        rospy.Subscriber(camera_info_topic, CameraInfo, self._camera_info_cb)
        if self.proj.use_depth:
            rospy.Subscriber(depth_topic, Image, self.proj.on_depth)

        rospy.loginfo("HandTrackingNode ready. image=%s  use_depth=%s",
                      image_topic, self.proj.use_depth)

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
                for hand in hands:
                    lm3d = [self.proj.pixel_to_3d(px, py) for px, py in hand["pixel_landmarks"]]
                    hand["landmarks_3d"] = lm3d

                    # Publish 21 landmarks as PoseArray
                    pose_array = PoseArray()
                    pose_array.header.stamp = rospy.Time.now()
                    pose_array.header.frame_id = "base_link" if self.proj.has_transform else self.proj.camera_frame
                    for pt in lm3d:
                        if pt is None:
                            continue
                        pose = Pose()
                        pose.position = Point(x=pt[0], y=pt[1], z=pt[2])
                        pose_array.poses.append(pose)
                    self.landmarks_3d_pub.publish(pose_array)

                first = hands[0]
                draw_pt = first.get("landmarks_3d", [None] * 21)[DRAW_LANDMARK_IDX]
                pixel_lms = first.get("pixel_landmarks")
                draw_px = tuple(pixel_lms[DRAW_LANDMARK_IDX]) if pixel_lms else None
                self.draw_seq.update(
                    gesture=first.get("gesture"),
                    point_3d=draw_pt,
                    landmarks=first["landmarks"],
                    pixel_point=draw_px,
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
