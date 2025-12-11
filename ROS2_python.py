#!/usr/bin/env python3
# unified_waypoint_follow_depth_avoid_and_human_follow.py
# ROS 2 Humble, PX4 + MAVROS
# - Auto takeoff -> square waypoint mission with yaw turns
# - Obstacle avoidance from /depth_camera
# - Human detection from /camera (YOLO). If a person wearing red is seen -> FOLLOW_HUMAN mode
# - While following, still apply the SAME obstacle-avoid override used in mission mode
#
# Topics kept the same as your originals:
#   - Publishes velocity to: /mavros/setpoint_velocity/cmd_vel_unstamped
#   - Publishes position setpoint to: /mavros/setpoint_position/local
#   - Subscribes MAVROS state: /mavros/state
#   - Subscribes local pose: /mavros/local_position/pose
#   - Subscribes depth image: /depth_camera
#   - Subscribes RGB camera: /camera
#
# Control loop frequency: 20 Hz (0.05 s timer), same as before.
# QoS profiles: kept as in your obstacle-avoid node.

from _future_ import annotations
import math
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from sensor_msgs.msg import Image
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointFollower(Node):
    # ---- Mission & controller params (same as your node) ----
    TARGET_ALT = 1.5
    ALT_TOL = 1.2
    TKO_TO = 90.0

    POS_ACC_RAD = 1.0
    YAW_ACC_RAD = 0.15

    MAX_SPD_FWD = 1.0
    MAX_SPD_LAT = 0.5
    MAX_SPD_Z = 0.5

    KP_YAW = 0.8
    KP_TRACK = 0.3

    OBS_THRESH = 1.0
    AVOID_SPD = MAX_SPD_FWD
    CLEAR_GAIN = 0.8

    ALPHA_FWD = 0.25
    ALPHA_LAT = 0.3
    ALPHA_YAW = 0.25

    DEPTH_DOWNSCALE = (160, 120)

    # ---- Human-follow params ----
    IMG_W = 640   # assumed camera width (used for normalized errors)
    IMG_H = 360   # assumed camera height (used for normalized errors)
    RED_RATIO_THRESH = 0.15  # fraction of pixels in bbox that must be red
    FOLLOW_FWD_BASE = 1.0    # nominal forward speed (m/s) when following
    FOLLOW_K_YAW = 0.6       # yaw rate gain from pixel error
    FOLLOW_K_STRAFE = 0.5    # lateral speed gain from pixel error
    FOLLOW_MIN_BOX = 60      # desired bbox height (px) ~ "target distance"
    FOLLOW_K_FWD = 0.01      # forward speed adjust based on distance error

    LOST_HUMAN_TIMEOUT = 1.0  # seconds to persist follow if temporarily lost

    def _init_(self) -> None:
        super()._init_("waypoint_follower_depth_avoid")
        

        self.bridge = CvBridge()
        self.debug_pub = self.create_publisher(Image, "/debug_image", 10)
        self.cb_group = ReentrantCallbackGroup()

        # ---- QoS: kept identical to your original ----
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sp_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---- Publishers / Clients (same topics) ----
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", sp_qos)
        self.pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", sp_qos)

        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")

        # ---- Subscriptions (same topics + added /camera) ----
        self.create_subscription(State, "/mavros/state", self.state_cb, state_qos)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, pose_qos)
        self.create_subscription(Image, "/depth_camera", self.depth_cb, img_qos, callback_group=self.cb_group)
        self.create_subscription(Image, "/camera", self.camera_cb, img_qos, callback_group=self.cb_group)

        # ---- MAVROS state ----
        self.state = State()
        self.pose = PoseStamped()
        self.yaw = 0.0

        self.have_pose = False
        self.offboard = False
        self.last_req = self.get_clock().now()

        # ---- Depth handling ----
        self.depth_msg: Optional[Image] = None
        self.depth_img: Optional[np.ndarray] = None

        # ---- Filters for velocity smoothing ----
        self.prev_fwd: Optional[float] = None
        self.prev_lat: Optional[float] = None
        self.prev_yaw: Optional[float] = None

        # ---- Mission plan ----
        self.waypoints: List[Tuple[float, float, float, Optional[float]]] = []
        self.wp_idx = 0

        # ---- State machine ----
        # States: TAKEOFF, HOVER, MISSION, TURN, FOLLOW_HUMAN, FINISHED, LANDED
        self.state_machine = "TAKEOFF"
        self.hover_start = None
        self.turn_start = None
        self.target_yaw: Optional[float] = None

        # ---- YOLO setup ----
        if torch.cuda.is_available():
            self.device = 0
            self.get_logger().info(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.get_logger().warn("CUDA not available, using CPU.")
        self.yolo = YOLO("yolov8n.pt")
        self.last_human_time = None
        self.target_human = None  # (cx, cy, box_w, box_h) in pixels

        # ---- Misc ----
        self._last_setpoint_log = self.get_clock().now()
        self._last_avoid_log = self.get_clock().now()

        # 20 Hz control timer (same as original)
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Unified waypoint follower + human follow with depth avoidance initialised")

    # ------------------- Callbacks -------------------
    def state_cb(self, msg: State) -> None:
        self.state = msg

    def pose_cb(self, msg: PoseStamped) -> None:
        self.pose = msg
        self.yaw = quaternion_to_yaw(msg.pose.orientation)
        if not self.have_pose and msg.header.stamp.sec:
            self.have_pose = True
            self.get_logger().info("Pose locked")

    def depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg

    def camera_cb(self, msg: Image) -> None:
        """YOLO inference; set FOLLOW_HUMAN if red-shirt person detected."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # For consistent control gains, work at known resolution:
        frame = cv2.resize(frame, (self.IMG_W, self.IMG_H))

        results = self.yolo(frame, device=self.device, verbose=False)
        best = None
        best_area = 0

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if int(cls) != 0:   # only "person"
                    continue
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(self.IMG_W-1, x1))
                y1 = max(0, min(self.IMG_H-1, y1))
                x2 = max(0, min(self.IMG_W-1, x2))
                y2 = max(0, min(self.IMG_H-1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # crude "red shirt" check: HSV mask for red
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
                mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
                mask = cv2.bitwise_or(mask1, mask2)
                red_ratio = float(mask.sum()) / float(roi.shape[0] * roi.shape[1] * 255.0)

                if red_ratio >= self.RED_RATIO_THRESH:
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        best = (cx, cy, x2 - x1, y2 - y1)

        if best is not None:
            cx, cy, bw, bh = best
            # draw bbox and center
            cv2.rectangle(frame, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            self.target_human = best
            self.last_human_time = self.get_clock().now()
            # switch to FOLLOW_HUMAN unless we're landing/finished
            if self.state_machine not in ("FINISHED", "LANDED", "TAKEOFF"):
                if self.state_machine != "FOLLOW_HUMAN":
                    self.get_logger().info("Red-shirt human detected → FOLLOW_HUMAN")
                self.state_machine = "FOLLOW_HUMAN"

        # Publish debug image instead of imshow
        debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.debug_pub.publish(debug_msg)


    # ------------------- MAVROS helpers -------------------
    def _call(self, cli, req, label: str) -> None:
        if cli.wait_for_service(1.0):
            cli.call_async(req)
            self.get_logger().info(f"{label} request")
        else:
            self.get_logger().warn(f"{label} service unavailable")

    def arm(self) -> None:
        self._call(self.arm_cli, CommandBool.Request(value=True), "Arm")

    def set_mode(self, mode: str) -> None:
        self._call(self.mode_cli, SetMode.Request(custom_mode=mode), f"Mode {mode}")

    # ------------------- Utility -------------------
    def dummy_sp(self) -> PoseStamped:
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"
        if self.have_pose:
            sp.pose = self.pose.pose
        else:
            sp.pose.position = Point(x=0.0, y=0.0, z=0.1)
            sp.pose.orientation = euler_to_quaternion(0.0, 0.0, 0.0)
        return sp

    def at_altitude(self) -> bool:
        return abs(self.pose.pose.position.z - self.TARGET_ALT) < self.ALT_TOL

    def wp_reached(self, wp) -> bool:
        dx = wp[0] - self.pose.pose.position.x
        dy = wp[1] - self.pose.pose.position.y
        dz = (wp[2] if wp[2] is not None else self.TARGET_ALT) - self.pose.pose.position.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist < self.POS_ACC_RAD

    @staticmethod
    def wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _convert_and_downscale_depth(self) -> None:
        if self.depth_msg is None:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(self.depth_msg, "passthrough")
            if img.dtype == np.uint16:
                img = img.astype(np.float32) * 0.001
            else:
                img = img.astype(np.float32)
            img = np.nan_to_num(img, nan=10.0, posinf=10.0, neginf=10.0)
            img[img < 0.1] = 10.0
            target_w, target_h = self.DEPTH_DOWNSCALE
            h, w = img.shape[:2]
            if (w, h) != (target_w, target_h):
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            self.depth_img = img
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    # ------------- Obstacle-avoid override (shared) -------------
    def avoid_override(self, fwd_cmd: float, lat_cmd: float) -> Tuple[float, float]:
        """Given desired forward/lat commands, override with depth-based avoidance if needed."""
        if self.depth_img is None or self.pose.pose.position.z <= 0.5:
            return fwd_cmd, lat_cmd

        dimg = self.depth_img
        rows = slice(int(2 * dimg.shape[0] / 3), dimg.shape[0])
        cols = slice(int(dimg.shape[1] / 3), int(2 * dimg.shape[1] / 3))
        roi = dimg[rows, cols]
        valid = (roi > 0.1) & (roi < 10.0) & np.isfinite(roi)
        if not np.any(valid):
            return fwd_cmd, lat_cmd

        min_depth = float(roi[valid].min())
        if min_depth >= self.OBS_THRESH:
            return fwd_cmd, lat_cmd

        mid_col = roi.shape[1] // 2
        left_roi, right_roi = roi[:, :mid_col], roi[:, mid_col:]
        left_valid, right_valid = valid[:, :mid_col], valid[:, mid_col:]
        left_mean = float(np.nanmean(np.where(left_valid, left_roi, np.nan))) if np.any(left_valid) else float("inf")
        right_mean = float(np.nanmean(np.where(right_valid, right_roi, np.nan))) if np.any(right_valid) else float("inf")

        # choose lateral direction toward more free space
        if left_mean == right_mean:
            lat_cmd_new = 0.0
            direction = "dead-ahead"
        else:
            if left_mean > right_mean:
                lat_cmd_new = self.AVOID_SPD
                direction = "sidestep left"
            else:
                lat_cmd_new = -self.AVOID_SPD
                direction = "sidestep right"

        # slow down proportionally to clearance
        fwd_cmd_new = max(0.0, self.CLEAR_GAIN * (min_depth / self.OBS_THRESH) * max(0.0, fwd_cmd))

        now = self.get_clock().now()
        if (now - self._last_avoid_log).nanoseconds / 1e9 > 1.0:
            self.get_logger().info(
                f"Obstacle {min_depth:.1f} m ahead → left={left_mean:.1f} m right={right_mean:.1f} m → {direction}"
            )
            self._last_avoid_log = now

        return fwd_cmd_new, lat_cmd_new

    # ------------------- Control loop -------------------
    def control_loop(self) -> None:
        now = self.get_clock().now()

        if self.depth_msg is not None:
            self._convert_and_downscale_depth()
            self.depth_msg = None

        if (now - self._last_setpoint_log).nanoseconds / 1e9 > 2.0:
            self.get_logger().info(f"SM={self.state_machine} z={self.pose.pose.position.z:.2f} depth={'yes' if self.depth_img is not None else 'no'}")
            self._last_setpoint_log = now

        if not self.state.connected or not self.have_pose:
            self.pos_pub.publish(self.dummy_sp())
            return

        # Establish OFFBOARD + arm
        if not self.offboard:
            self.pos_pub.publish(self.dummy_sp())
            if (now - self.last_req).nanoseconds / 1e9 > 1.0:
                if self.state.mode != "OFFBOARD":
                    self.set_mode("OFFBOARD")
                elif not self.state.armed:
                    self.arm()
                self.last_req = now
            if self.state.mode == "OFFBOARD" and self.state.armed:
                self.offboard = True
                self.tko_start = now
            return

        # FOLLOW_HUMAN keep-alive: if we haven't seen a human recently, return to mission
        if self.state_machine == "FOLLOW_HUMAN":
            if self.last_human_time is None or (now - self.last_human_time).nanoseconds / 1e9 > self.LOST_HUMAN_TIMEOUT:
                self.get_logger().info("Human lost → Hover and wait")
                self.state_machine = "HOVER_HUMAN_LOST"
                self.hover_lost_start = now
                self.target_human = None

        # If hovering after losing human
        if self.state_machine == "HOVER_HUMAN_LOST":
            # if human comes back → go FOLLOW_HUMAN
            if self.last_human_time is not None and (now - self.last_human_time).nanoseconds / 1e9 <= self.LOST_HUMAN_TIMEOUT:
                self.get_logger().info("Human reacquired → FOLLOW_HUMAN")
                self.state_machine = "FOLLOW_HUMAN"
            else:
                # how long to hover before giving up
                hover_dur = 3.0  # seconds (tune as needed)
                if (now - self.hover_lost_start).nanoseconds / 1e9 > hover_dur:
                    self.get_logger().info("Human not found → back to MISSION")
                    self.state_machine = "MISSION"
                else:
                    # hover in place
                    self.vel_pub.publish(Twist())
            return

        # ----------------- State machine -----------------
        if self.state_machine == "TAKEOFF":
            sp = PoseStamped()
            sp.header.stamp = now.to_msg()
            sp.header.frame_id = "map"
            sp.pose.position.x = self.pose.pose.position.x
            sp.pose.position.y = self.pose.pose.position.y
            sp.pose.position.z = self.TARGET_ALT
            sp.pose.orientation = self.pose.pose.orientation
            self.pos_pub.publish(sp)

            dz = self.TARGET_ALT - self.pose.pose.position.z
            tw = Twist()
            tw.linear.z = max(-self.MAX_SPD_Z, min(self.MAX_SPD_Z, 0.6 * dz))
            self.vel_pub.publish(tw)

            if self.at_altitude():
                origin_x = float(self.pose.pose.position.x)
                origin_y = float(self.pose.pose.position.y)
                origin_yaw = float(self.yaw)
                self.waypoints = self.generate_square_waypoints(5.0, origin_x, origin_y, origin_yaw)
                self.get_logger().info(f"Generated waypoints: {self.waypoints}")
                self.state_machine = "HOVER"
                self.hover_start = now
                self.get_logger().info("Takeoff complete → Hovering")
            return

        if self.state_machine == "HOVER":
            if (now - self.hover_start).nanoseconds * 1e-9 > 2.0:
                if self.wp_idx < len(self.waypoints):
                    self.state_machine = "MISSION"
                else:
                    self.state_machine = "FINISHED"
            else:
                self.vel_pub.publish(Twist())
            return

        if self.state_machine == "MISSION":
            if self.wp_idx >= len(self.waypoints):
                self.state_machine = "FINISHED"
                return
            wp = self.waypoints[self.wp_idx]
            self.navigate_to_waypoint(wp)
            if self.wp_reached(wp):
                self.get_logger().info(f"Reached WP {self.wp_idx}")
                prev_heading = self.waypoints[self.wp_idx][3]
                self.target_yaw = self.wrap_pi(prev_heading + math.pi / 2)
                self.wp_idx += 1
                self.state_machine = "TURN"
                self.turn_start = now
            return

        if self.state_machine == "TURN":
            if self.target_yaw is None:
                self.state_machine = "FINISHED"
                return
            yaw_err = self.wrap_pi(self.target_yaw - self.yaw)
            tw = Twist()
            tw.angular.z = self.KP_YAW * yaw_err
            self.vel_pub.publish(tw)
            if abs(yaw_err) < self.YAW_ACC_RAD:
                self.state_machine = "HOVER"
                self.hover_start = now
                self.get_logger().info("Turn complete → Hovering")
            return

        if self.state_machine == "FOLLOW_HUMAN":
            self.follow_human()
            return

        if self.state_machine == "FINISHED":
            self.get_logger().info("Mission complete → Landing")
            self.set_mode("AUTO.LAND")
            self.state_machine = "LANDED"
            return

        if self.state_machine == "LANDED":
            self.vel_pub.publish(Twist())
            return

    # ------------------- Trajectory -------------------
    def generate_square_waypoints(self, side_len: float, origin_x: float, origin_y: float, heading: float) -> List[Tuple[float, float, float, float]]:
        wp: List[Tuple[float, float, float, float]] = []
        x, y = origin_x, origin_y
        h = heading
        for _ in range(4):
            x += side_len * math.cos(h)
            y += side_len * math.sin(h)
            wp.append((x, y, self.TARGET_ALT, h))
            h += math.pi / 2
        return wp

    # ------------------- Mission navigation -------------------
    def navigate_to_waypoint(self, wp: Tuple[float, float, float, float]) -> None:
        wp_x, wp_y, wp_z, wp_yaw = wp
        dx = wp_x - self.pose.pose.position.x
        dy = wp_y - self.pose.pose.position.y
        dz = wp_z - self.pose.pose.position.z
        dist_xy = math.hypot(dx, dy)

        tgt_yaw = math.atan2(dy, dx)
        base_fwd = min(self.MAX_SPD_FWD, self.KP_TRACK * dist_xy)
        fwd_cmd, lat_cmd = base_fwd, 0.0

        # depth-based avoidance override (shared)
        fwd_cmd, lat_cmd = self.avoid_override(fwd_cmd, lat_cmd)

        # simple smoothing like before
        self.prev_fwd = fwd_cmd if self.prev_fwd is None else (self.ALPHA_FWD * fwd_cmd + (1 - self.ALPHA_FWD) * self.prev_fwd)
        self.prev_lat = lat_cmd if self.prev_lat is None else (self.ALPHA_LAT * lat_cmd + (1 - self.ALPHA_LAT) * self.prev_lat)
        self.prev_yaw = 0.0 if self.prev_yaw is None else self.prev_yaw

        psi = tgt_yaw
        enu_vx = self.prev_fwd * math.cos(psi) - self.prev_lat * math.sin(psi)
        enu_vy = self.prev_fwd * math.sin(psi) + self.prev_lat * math.cos(psi)
        enu_vz = max(-self.MAX_SPD_Z, min(self.MAX_SPD_Z, self.KP_TRACK * dz))
        yaw_rate = 0.0

        twist = Twist()
        twist.linear.x = float(enu_vx)
        twist.linear.y = float(enu_vy)
        twist.linear.z = float(enu_vz)
        twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(twist)

    # ------------------- Follow human (with avoidance) -------------------
    def follow_human(self) -> None:
        if self.target_human is None:
            # nothing to do, will time out back to mission
            self.vel_pub.publish(Twist())
            return

        cx, cy, bw, bh = self.target_human

        # pixel errors normalized to [-1, 1]
        err_x = (cx - self.IMG_W / 2) / (self.IMG_W / 2)   # +right, -left
        err_y = (cy - self.IMG_H / 2) / (self.IMG_H / 2)   # +down, -up

        # distance proxy from bbox height
        dist_err = (self.FOLLOW_MIN_BOX - bh)  # positive if person is "small" (far)
        fwd_cmd = self.FOLLOW_FWD_BASE + self.FOLLOW_K_FWD * dist_err
        fwd_cmd = max(0.0, min(self.MAX_SPD_FWD, fwd_cmd))

        # yaw to center horizontally, strafe to center vertically
        yaw_rate_cmd = -self.FOLLOW_K_YAW * err_x
        lat_cmd = self.FOLLOW_K_STRAFE * (-err_y)  # up in image -> move left/right depends on camera mounting; adjust if needed
        lat_cmd = max(-self.MAX_SPD_LAT, min(self.MAX_SPD_LAT, lat_cmd))

        # apply obstacle avoidance override to forward/lat
        fwd_cmd, lat_cmd = self.avoid_override(fwd_cmd, lat_cmd)

        # smooth like mission
        self.prev_fwd = fwd_cmd if self.prev_fwd is None else (self.ALPHA_FWD * fwd_cmd + (1 - self.ALPHA_FWD) * self.prev_fwd)
        self.prev_lat = lat_cmd if self.prev_lat is None else (self.ALPHA_LAT * lat_cmd + (1 - self.ALPHA_LAT) * self.prev_lat)
        self.prev_yaw = yaw_rate_cmd if self.prev_yaw is None else (self.ALPHA_YAW * yaw_rate_cmd + (1 - self.ALPHA_YAW) * self.prev_yaw)

        # body-frame -> ENU using current target yaw = current heading
        psi = self.yaw
        enu_vx = self.prev_fwd * math.cos(psi) - self.prev_lat * math.sin(psi)
        enu_vy = self.prev_fwd * math.sin(psi) + self.prev_lat * math.cos(psi)

        twist = Twist()
        twist.linear.x = float(enu_vx)
        twist.linear.y = float(enu_vy)
        twist.linear.z = 0.0
        twist.angular.z = float(self.prev_yaw)
        self.vel_pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointFollower()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
