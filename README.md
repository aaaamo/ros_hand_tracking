# ROS Hand Tracking

A ROS package for real-time hand landmark tracking using MediaPipe HandLandmarker (Tasks API).

Detects 21 hand keypoints and publishes them as JSON over a ROS topic. No gesture classification — pure tracking only.

## Dependencies

```
mediapipe >= 0.10.32
opencv-python >= 4.13.0
numpy >= 2.4.0
```

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Clone this repo into a `src` folder of a catkin workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone <your-repo-url> ros_hand_tracking
   ```

3. Build:
   ```bash
   cd ~/catkin_ws
   catkin build
   ```

The `hand_landmarker.task` model file is included in the `model/` directory.

## Usage

1. Source the workspace:
   ```bash
   source ~/catkin_ws/devel/setup.bash
   ```

2. Launch an image publisher (camera node), then:
   ```bash
   roslaunch ros_hand_tracking hand_tracking.launch
   ```

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subscribe_image_topic` | `/image_raw` | Input image topic |
| `publish_landmarks_topic` | `/hand_tracking/landmarks` | Output landmarks topic (JSON) |
| `model_path` | `$(find ros_hand_tracking)/model/hand_landmarker.task` | Model file path |
| `num_hands` | `1` | Max number of hands to detect |
| `min_detection_confidence` | `0.7` | Detection confidence threshold |
| `min_tracking_confidence` | `0.5` | Tracking confidence threshold |
| `show_image` | `True` | Show debug visualization window |

## Published Topics

- `/hand_tracking/landmarks` (`std_msgs/String`) — JSON array of detected hands, each containing:
  - `handedness`: `"Left"` or `"Right"`
  - `landmarks`: 21 normalized `{x, y, z}` coordinates
  - `pixel_landmarks`: 21 pixel `[x, y]` coordinates

## Subscribed Topics

- `/image_raw` (`sensor_msgs/Image`) — Camera image input

## File Structure

```
ros_hand_tracking/
├── src/
│   ├── hand_tracking.py    # ROS node
│   ├── hand_tracker.py     # MediaPipe HandLandmarker wrapper
│   └── cvfpscalc.py        # FPS utility
├── model/
│   └── hand_landmarker.task  # MediaPipe model
├── launch/
│   └── hand_tracking.launch
├── requirements.txt
├── package.xml
└── CMakeLists.txt
```

## Credits

Based on [ros_hand_gesture_recognition](https://github.com/TrinhNC/ros_hand_gesture_recognition) by TrinhNC. Migrated from MediaPipe legacy `mp.solutions.hands` API to the new `mp.tasks.vision.HandLandmarker` Tasks API.
