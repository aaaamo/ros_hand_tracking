#!/usr/bin/env python3
"""Standalone test for hand tracking / gesture recognition without ROS."""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2 as cv
from hand_tracker import HandTracker
from cvfpscalc import CvFpsCalc
from draw_sequence import (
    DrawSequence as _BaseDrawSequence,
    DRAW_START_GESTURE,
    DRAW_END_GESTURE,
    DRAW_LANDMARK_IDX,
    PINCH_THRESHOLD,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive, safe for background saving
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("WARNING: matplotlib not found; 3D plot will be skipped.")


# ===========================================================
# Arbitrary camera→world transform used when no ROS is available.
# Identity = camera frame equals world frame (origin at camera, no rotation).
# Replace with a real 4x4 homogeneous matrix to simulate a different pose.
# ===========================================================
ARBITRARY_TRANSFORM = np.eye(4, dtype=np.float64)
# ===========================================================


def pixel_to_3d(px, py, fx, fy, cx, cy, depth):
    """Back-project a pixel to 3D world coords via ARBITRARY_TRANSFORM."""
    x_cam = (px - cx) / fx * depth
    y_cam = (py - cy) / fy * depth
    z_cam = depth
    p = ARBITRARY_TRANSFORM @ np.array([x_cam, y_cam, z_cam, 1.0])
    return [round(float(v), 4) for v in p[:3]]


class DrawSequence(_BaseDrawSequence):
    """Standalone draw sequence: extends base with pixel tracking, file output, and 3D plot."""

    def __init__(self, output_file):
        super().__init__(publish_fn=self._on_end)
        self._pixel_path = []
        self.display_pixel_path = []
        self.output_file = output_file

    @property
    def state(self):
        return "running" if self.running else "idle"

    @property
    def pixel_path(self):
        return self._pixel_path

    def update(self, gesture, point_3d, pixel_xy, landmarks):
        was_running = self.running
        n_before = len(self._path)
        super().update(gesture, point_3d, landmarks)
        if not was_running and self.running:
            self._pixel_path = []
            self.display_pixel_path = []
            print("[DRAW] START")
        elif self.running and len(self._path) > n_before:
            self._pixel_path.append(pixel_xy)

    def _on_end(self, json_str):
        n = len(self._path)
        print("[DRAW] END — %d points collected" % n)
        if n > 0:
            with open(self.output_file, "a") as f:
                f.write(json_str + "\n")
            print("[DRAW] appended to %s" % self.output_file)
            self._save_plot()
        self.display_pixel_path = list(self._pixel_path)
        self._pixel_path = []

    def _save_plot(self):
        if not _HAS_MPL:
            return
        xs = [p[0] for p in self._path]
        ys = [p[1] for p in self._path]
        zs = [p[2] for p in self._path]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, "b-", linewidth=1)
        ax.scatter(xs, ys, zs, c="blue", s=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Draw Path (%d pts)" % len(self._path))

        # Equal axis scaling
        half = (
            max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2.0 or 1e-3
        )
        mid_x = (max(xs) + min(xs)) / 2.0
        mid_y = (max(ys) + min(ys)) / 2.0
        mid_z = (max(zs) + min(zs)) / 2.0
        ax.set_xlim(mid_x - half, mid_x + half)
        ax.set_ylim(mid_y - half, mid_y + half)
        ax.set_zlim(mid_z - half, mid_z + half)
        base = os.path.splitext(self.output_file)[0]
        plot_file = "%s_%d.png" % (base, int(time.time()))
        plt.savefig(plot_file, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("[DRAW] 3D plot saved to %s" % plot_file)


class WebcamSource:
    """Standard webcam via OpenCV."""

    def __init__(self, device=0, width=960, height=540):
        self.cap = cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()


class VideoFileSource:
    """Pre-recorded video file via OpenCV."""

    def __init__(self, path, loop=False):
        self.cap = cv.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError("Cannot open video: %s" % path)
        self.loop = loop
        self.path = path
        fps = self.cap.get(cv.CAP_PROP_FPS) or 30
        self.frame_delay = max(1, int(1000 / fps))
        total = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("Video: %s  (%.1f fps, %d frames)" % (path, fps, total))

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                return frame if ret else None
            return None
        return frame

    def close(self):
        self.cap.release()


class RealSenseSource:
    """Intel RealSense color stream via pyrealsense2."""

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        print("RealSense started: %dx%d @ %dfps" % (width, height, fps))

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        self.pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="", help="Path to a video file for testing"
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop video file continuously"
    )
    parser.add_argument(
        "--realsense", action="store_true", help="Use Intel RealSense camera"
    )
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--gesture", action="store_true", help="Enable gesture recognition"
    )
    parser.add_argument("--num_hands", type=int, default=1)
    parser.add_argument("--min_detection", type=float, default=0.7)
    parser.add_argument("--min_tracking", type=float, default=0.5)
    parser.add_argument(
        "--model", type=str, default="", help="Path to .task model file"
    )
    # 3D projection (standalone uses ARBITRARY_TRANSFORM defined above)
    parser.add_argument(
        "--fixed_depth",
        type=float,
        default=0.5,
        help="Assumed depth in metres for 3D back-projection",
    )
    parser.add_argument("--fx", type=float, default=600.0)
    parser.add_argument("--fy", type=float, default=600.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    # Draw sequence output
    parser.add_argument(
        "--output",
        type=str,
        default="draw_path.txt",
        help="File to append completed draw paths (JSON lines)",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model:
        model_path = args.model
    else:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.gesture:
            model_path = os.path.join(pkg_dir, "model", "gesture_recognizer.task")
        else:
            model_path = os.path.join(pkg_dir, "model", "hand_landmarker.task")

    # Camera source
    if args.video:
        source = VideoFileSource(path=args.video, loop=args.loop)
    elif args.realsense:
        source = RealSenseSource(width=args.width, height=args.height)
    else:
        source = WebcamSource(device=args.device, width=args.width, height=args.height)

    print("Model: %s" % model_path)
    print("Gesture: %s" % args.gesture)
    if args.video:
        print("Source: Video file %s%s" % (args.video, " (loop)" if args.loop else ""))
    else:
        print(
            "Source: %s"
            % ("RealSense" if args.realsense else "Webcam %d" % args.device)
        )
    print("Press ESC to quit.")

    if not args.gesture:
        print("WARNING: --gesture not set; draw sequence requires gesture recognition.")

    tracker = HandTracker(
        model_path=model_path,
        num_hands=args.num_hands,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
        enable_gesture=args.gesture,
    )

    fps_calc = CvFpsCalc(buffer_len=10)
    draw_seq = DrawSequence(output_file=args.output)

    print(
        "Draw gestures: start=%s  draw=pinch(%.2f)  end=%s  landmark=%d"
        % (DRAW_START_GESTURE, PINCH_THRESHOLD, DRAW_END_GESTURE, DRAW_LANDMARK_IDX)
    )

    while True:
        frame = source.read()
        if frame is None:
            break

        frame = cv.flip(frame, 1)
        hands = tracker.detect(frame)

        # Draw
        debug_image = frame.copy()
        tracker.draw(debug_image, hands)

        # FPS
        fps = fps_calc.get()
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

        # Print detected info
        y_offset = 60
        for hand in hands:
            info = hand["handedness"]
            if "gesture" in hand:
                info += " | %s (%.2f)" % (hand["gesture"], hand["gesture_score"])
            cv.putText(
                debug_image,
                info,
                (10, y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            y_offset += 30

        # Draw sequence update (first hand, gesture mode only)
        if args.gesture and hands:
            hand = hands[0]
            gesture = hand.get("gesture")
            px, py = hand["pixel_landmarks"][DRAW_LANDMARK_IDX]
            pt3d = pixel_to_3d(
                px, py, args.fx, args.fy, args.cx, args.cy, args.fixed_depth
            )
            draw_seq.update(gesture, pt3d, (px, py), hand["landmarks"])

        # Overlay: completed path (yellow) — persists after sequence ends
        pts_done = draw_seq.display_pixel_path
        for i in range(1, len(pts_done)):
            cv.line(
                debug_image, pts_done[i - 1], pts_done[i], (0, 215, 255), 2, cv.LINE_AA
            )
        for pt in pts_done:
            cv.circle(debug_image, pt, 3, (0, 215, 255), -1)

        # Overlay: currently collecting path (green)
        pts_live = draw_seq.pixel_path
        for i in range(1, len(pts_live)):
            cv.line(
                debug_image, pts_live[i - 1], pts_live[i], (0, 255, 80), 2, cv.LINE_AA
            )
        for pt in pts_live:
            cv.circle(debug_image, pt, 3, (0, 255, 80), -1)

        # Draw sequence state label
        state_color = {"idle": (80, 80, 80), "running": (0, 255, 80)}
        color = state_color.get(draw_seq.state, (255, 255, 255))
        label = "DRAW: %s (%d pts)" % (draw_seq.state.upper(), len(draw_seq.path))
        h_img = debug_image.shape[0]
        cv.putText(
            debug_image,
            label,
            (10, h_img - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            debug_image,
            label,
            (10, h_img - 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv.LINE_AA,
        )

        cv.imshow("Hand Tracking Test", debug_image)
        delay = source.frame_delay if hasattr(source, "frame_delay") else 1
        if cv.waitKey(delay) == 27:  # ESC
            break

    tracker.close()
    source.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
