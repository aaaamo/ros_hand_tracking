#!/usr/bin/env python3
"""ROS node for hand tracking and gesture recognition using MediaPipe."""

import sys
import os

# Ensure src/ directory is on Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv

from hand_tracker import HandTracker
from cvfpscalc import CvFpsCalc


class HandTrackingNode:

    def __init__(self):
        rospy.init_node('hand_tracking', anonymous=True)

        # Parameters
        image_topic = rospy.get_param('~subscribe_image_topic', '/image_raw')
        landmarks_topic = rospy.get_param('~publish_landmarks_topic',
                                          '/hand_tracking/landmarks')
        gesture_topic = rospy.get_param('~publish_gesture_topic',
                                        '/hand_tracking/gesture')
        model_path = rospy.get_param('~model_path', '')
        num_hands = rospy.get_param('~num_hands', 1)
        min_detection = rospy.get_param('~min_detection_confidence', 0.7)
        min_tracking = rospy.get_param('~min_tracking_confidence', 0.5)
        self.show_image = rospy.get_param('~show_image', True)
        enable_gesture = rospy.get_param('~enable_gesture', False)

        # Resolve default model path relative to package
        if not model_path:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if enable_gesture:
                model_path = os.path.join(pkg_dir, 'model', 'gesture_recognizer.task')
            else:
                model_path = os.path.join(pkg_dir, 'model', 'hand_landmarker.task')

        rospy.loginfo('Loading model from: %s (gesture=%s)', model_path, enable_gesture)

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

        rospy.loginfo('HandTrackingNode ready. Subscribed to: %s', image_topic)

    def callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr('CvBridge error: %s', e)
            return

        cv_image = cv.flip(cv_image, 1)  # Mirror display
        hands = self.tracker.detect(cv_image)

        # Publish landmarks as JSON
        if hands:
            json_str = HandTracker.hands_to_json(hands)
            self.landmarks_pub.publish(json_str)

            # Publish gesture separately (first hand's gesture)
            if self.gesture_pub and 'gesture' in hands[0]:
                self.gesture_pub.publish(hands[0]['gesture'])

        # Debug visualization
        if self.show_image:
            debug_image = cv_image.copy()
            self.tracker.draw(debug_image, hands)
            fps = self.fps_calc.get()
            cv.putText(debug_image, 'FPS: {:.1f}'.format(fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(debug_image, 'FPS: {:.1f}'.format(fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Hand Tracking', debug_image)
            cv.waitKey(1)

    def shutdown(self):
        self.tracker.close()
        cv.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = HandTrackingNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
