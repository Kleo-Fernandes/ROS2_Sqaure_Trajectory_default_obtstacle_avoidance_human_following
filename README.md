# Unified waypoint follower with depth avoidance and human follow

**File:** `unified_waypoint_follow_depth_avoid_and_human_follow.py`

Simulation Demonstration: https://drive.google.com/file/d/1Rff3KlTEVmTZJt59RrR4Pg0o91UqzAbb/view?usp=drive_link
## Overview

ROS 2 Humble node for PX4 + MAVROS that combines three behaviours in one controller:

* Auto takeoff to target altitude, then a square waypoint mission with yaw turns.
* Depth-image based ob


https://github.com/user-attachments/assets/adea5ab7-cd07-4e68-aef8-c493dd8dcf8d


https://github.com/user-attachments/assets/5096f1b5-9f31-44da-a078-0453c6815c35

stacle avoidance using `/depth_camera`.
* Human detection via YOLO on `/camera`. If a person wearing red is detected the node switches to FOLLOW_HUMAN mode and tracks that person while preserving the same depth-based avoidance override.

Control loop runs at 20 Hz. Topics and QoS match the original node expectations so it can be dropped into existing MAVROS setups with minimal changes.

## Features

* Automatic OFFBOARD enable + arming sequence.
* Square waypoint generation relative to takeoff pose.
* Shared obstacle-avoid override used by both mission navigation and human follow.
* Red-shirt person detection using YOLOv8 and a simple HSV red ratio filter to trigger follow mode.
* Debug image publisher on `/debug_image` with detection overlays.

## Prerequisites

* ROS 2 Humble

* PX4 flight stack and MAVROS configured for offboard control

* Python 3.8+ with the following packages installed:

  * `rclpy`, `geometry_msgs`, `sensor_msgs`, `mavros_msgs` (ROS 2 packages)
  * `cv_bridge`
  * `numpy`
  * `opencv-python`
  * `torch` (PyTorch) compatible with your CUDA or CPU setup
  * `ultralytics` (for YOLOv8 wrapper)

* YOLO model file: `yolov8n.pt` accessible to the node

## Topics

**Publishes**

* `/mavros/setpoint_velocity/cmd_vel_unstamped` (`geometry_msgs/Twist`)  - body/ENU velocity commands
* `/mavros/setpoint_position/local` (`geometry_msgs/PoseStamped`) - fallback/dummy setpoints for OFFBOARD
* `/debug_image` (`sensor_msgs/Image`) - visualization output with drawn bounding boxes

**Subscribes**

* `/mavros/state` (`mavros_msgs/State`) - MAVROS state
* `/mavros/local_position/pose` (`geometry_msgs/PoseStamped`) - local pose
* `/depth_camera` (`sensor_msgs/Image`) - depth image for obstacle avoidance
* `/camera` (`sensor_msgs/Image`) - RGB camera for human detection

## Configuration and Parameters (in-code constants)

Primary parameters live in the node as class constants. Notable ones you may tune:

* `TARGET_ALT` (default 1.5) - target takeoff altitude in meters
* `POS_ACC_RAD` (default 1.0) - waypoint acceptance radius in meters
* `MAX_SPD_FWD`, `MAX_SPD_LAT`, `MAX_SPD_Z` - max speeds for forward, lateral and vertical motion
* Obstacle avoidance: `OBS_THRESH` (clearance threshold in meters), `AVOID_SPD`, `CLEAR_GAIN`
* YOLO / human follow: `IMG_W`, `IMG_H`, `RED_RATIO_THRESH`, `FOLLOW_FWD_BASE`, `FOLLOW_K_YAW`, `FOLLOW_K_STRAFE`, `FOLLOW_MIN_BOX`
* `DEPTH_DOWNSCALE` - resolution used for depth processing (default 160x120)
* Gains and smoothing: `KP_YAW`, `KP_TRACK`, `ALPHA_FWD`, `ALPHA_LAT`, `ALPHA_YAW`

If you need runtime parameter tuning, move these constants into ROS 2 parameters or use a YAML file and `declare_parameter` in the node.

## Installation

1. Install ROS 2 Humble and MAVROS according to official instructions.
2. Install Python dependencies in your ROS workspace or a Python virtual environment:

```bash
pip install numpy opencv-python torch ultralytics
# install cv_bridge via your ROS distro packages or build from source
```

3. Place `yolov8n.pt` in a directory accessible to the node (home, package data, or provide code changes to a custom path).
4. Put the node script in a ROS 2 Python package with an executable entry point or run it directly for quick testing.

## Running the node

Example quick test using direct execution (ensure ROS environment is sourced):

```bash
ros2 run <your_package> unified_waypoint_follow_depth_avoid_and_human_follow.py
```

Or if running the script directly:

```bash
python3 unified_waypoint_follow_depth_avoid_and_human_follow.py
```

Notes:

* Ensure `/mavros` topics are active and PX4 is communicating. For SITL testing use PX4 SITL + MAVROS.
* `yolov8n.pt` must be present. If using CUDA, ensure `torch.cuda.is_available()` returns true and drivers are installed.

## Testing and recommended workflow

1. Start PX4 SITL and MAVROS. Confirm `/mavros/state` and `/mavros/local_position/pose` publish correctly.
2. Start camera and depth sensor publishers or bag file playback for `/camera` and `/depth_camera`.
3. Launch the node. Check logs for "Pose locked" and YOLO device info.
4. Verify debug image on `/debug_image` to confirm bounding boxes and red-shirt detection.
5. Test obstacle avoidance by placing obstacles in front of the depth sensor in SITL or by simulating range data.
6. Test human follow using a person wearing a red shirt or a recorded video that contains detectable red regions.

## Troubleshooting

* Node logs warn "CUDA not available": either fall back to CPU or install compatible CUDA and PyTorch.
* YOLO returns no detections: verify `yolov8n.pt` path and camera resolution matches `IMG_W`/`IMG_H` assumptions.
* Depth conversion failed: check depth image encoding and types. The node expects either 16-bit millimetre depth or float32 meters.
* OFFBOARD not engaging: ensure MAVROS and PX4 parameterization allow offboard, and sufficient setpoint streaming occurs.

---

If you want this README exported as `README.md` in your workspace, tell me where to save it and I will produce the file content.
