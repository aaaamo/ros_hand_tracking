#!/usr/bin/env python3
"""3D back-projection helper: pixel coords → world coords via a ROS transform."""

import numpy as np
import rospy
import tf.transformations
from cv_bridge import CvBridgeError


class Projection3D:
    """Converts pixel (u, v) to 3-D world coordinates.

    Requires a camera→world TransformStamped (set via on_transform).
    Supports fixed-depth plane or RealSense depth image.
    """

    def __init__(self, bridge, use_depth=False, fixed_depth=0.5,
                 fx=600.0, fy=600.0, cx=320.0, cy=240.0):
        self.bridge = bridge
        self.use_depth = use_depth
        self.fixed_depth = fixed_depth
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.latest_transform = None   # geometry_msgs/TransformStamped
        self.latest_depth = None       # numpy array (mm, uint16)

    # --- ROS callbacks ---

    def on_transform(self, msg):
        self.latest_transform = msg

    def on_camera_info(self, msg):
        """Update intrinsics from CameraInfo. Returns True so caller can unregister."""
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        return True

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
        return self.latest_transform is not None

    def pixel_to_3d(self, px, py):
        """Back-project pixel (px, py) to [x, y, z] in world frame.

        Returns None if no transform is available yet.
        """
        if not self.ready:
            return None

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

        # Apply camera→world transform
        t = self.latest_transform.transform
        q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        mat = tf.transformations.quaternion_matrix(q)
        mat[0, 3] = t.translation.x
        mat[1, 3] = t.translation.y
        mat[2, 3] = t.translation.z

        p_world = mat @ np.array([x_cam, y_cam, z_cam, 1.0])
        return [round(float(v), 4) for v in p_world[:3]]
