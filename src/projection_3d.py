#!/usr/bin/env python3
"""3D back-projection helper: pixel coords → world coords via tf2."""

import rospy
import tf2_ros
from tf2_geometry_msgs import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridgeError


class Projection3D:
    """Converts pixel (u, v) to 3-D world coordinates in base_link frame.

    Uses tf2 to look up the camera→base_link transform automatically.
    Supports fixed-depth plane or RealSense depth image.
    """

    def __init__(self, bridge, target_frame="base_link",
                 use_depth=False, fixed_depth=0.5,
                 fx=600.0, fy=600.0, cx=320.0, cy=240.0):
        self.bridge = bridge
        self.camera_frame = None  # auto-detected from CameraInfo header
        self.target_frame = target_frame
        self.use_depth = use_depth
        self.fixed_depth = fixed_depth
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.latest_depth = None

        # tf2 persistent buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    # --- ROS callbacks ---

    def on_camera_info(self, msg):
        """Update intrinsics and camera frame from CameraInfo."""
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id
            rospy.loginfo_once("Camera frame auto-detected: '%s'", self.camera_frame)

    def on_depth(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            rospy.logerr("Depth CvBridge error: %s", e)

    # --- Projection ---

    @property
    def ready(self):
        """True once camera_frame is known (from CameraInfo)."""
        return self.camera_frame is not None

    @property
    def has_transform(self):
        """True when tf2 can resolve camera_frame → target_frame."""
        if not self.ready:
            return False
        try:
            self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.0))
            return True
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return False

    def pixel_to_3d(self, px, py):
        """Back-project pixel (px, py) to [x, y, z] in target_frame.

        Returns None if tf2 transform is not available.
        """
        # Determine depth
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

        # Transform camera_frame → target_frame via tf2
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0,
                "tf2 lookup '%s' -> '%s' failed, returning camera-frame coords: %s",
                self.camera_frame, self.target_frame, e)
            return [round(x_cam, 4), round(y_cam, 4), round(z_cam, 4)]

        pt = PointStamped()
        pt.header.frame_id = self.camera_frame
        pt.header.stamp = rospy.Time(0)
        pt.point.x = x_cam
        pt.point.y = y_cam
        pt.point.z = z_cam

        pt_world = tf2_geometry_msgs.do_transform_point(pt, transform)
        return [round(pt_world.point.x, 4),
                round(pt_world.point.y, 4),
                round(pt_world.point.z, 4)]
