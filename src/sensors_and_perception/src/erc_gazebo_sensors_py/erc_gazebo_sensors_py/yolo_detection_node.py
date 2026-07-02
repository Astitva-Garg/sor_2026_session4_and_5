#!/usr/bin/env python3
"""
yolo_detection_node.py
----------------------
Stage 1: Accept a target object via ROS parameter, filter detections to it.
Stage 2: Subscribe to depth image, estimate distance to the target using
         the bounding box centre pixel.
Stage 3: If a target is set but not visible, slowly rotate in place to search.
         Stop rotating the moment the target appears.

Run:
    ros2 run erc_gazebo_sensors_py yolo_detection
Press Q or Escape to quit.
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # yolov8s is a good balance of speed vs accuracy
        self.model = YOLO("yolov8s.pt")
        self.get_logger().info("YOLO model loaded")

        # --- Stage 1: Target object parameter ---
        # Set from another terminal while running:
        #   ros2 param set /yolo_detector target_object person
        self.declare_parameter('target_object', '')

        self.bridge = CvBridge()

        # --- Stage 2: Synchronised RGB + depth subscribers ---
        # ApproximateTimeSynchronizer pairs the two images by timestamp
        self.rgb_sub = Subscriber(self, Image, 'camera/image')
        self.depth_sub = Subscriber(self, Image, 'camera/depth_image')
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        self.latest_rgb   = None
        self.latest_depth = None
        self.frame_lock   = threading.Lock()

        self.running    = True
        self.prev_time  = time.time()

        # --- Stage 3 + 5: search/approach state ---
        self.target_visible    = False
        self.target_error_x    = 0
        self.target_distance   = None
        self._last_target_cx   = 0   # pixel x of target centre in latest frame

        # --- Stage 6: mission completion ---
        self.mission_complete  = False  # latched True once target is reached
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Motion controller runs at 10 Hz
        self.create_timer(0.1, self._search_behaviour)

        # Log when target changes
        self.create_timer(0.5, self._log_target)
        self._last_logged_target = None

        self.spin_thread = threading.Thread(
            target=self.spin_thread_func, daemon=True)
        self.spin_thread.start()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sync_callback(self, rgb_msg, depth_msg):
        """Receive RGB and depth frames with matching timestamps."""
        rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   'bgr8')
        # depth image is 32-bit float, each pixel = distance in metres
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        with self.frame_lock:
            self.latest_rgb   = rgb
            self.latest_depth = depth

    def stop(self):
        self.running = False
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

    def _log_target(self):
        """Print a status line whenever the target changes."""
        target = self.get_parameter('target_object').get_parameter_value().string_value.strip().lower()
        if target != self._last_logged_target:
            if target:
                self.get_logger().info(f"Searching for: {target}")
            else:
                self.get_logger().info("No target set — showing all detections. "
                                       "Set one with:  ros2 param set /yolo_detector target_object <class>")
            # New target = new mission, reset completion flag
            self.mission_complete = False
            self.target_visible   = False
            self.target_distance  = None
            self._last_logged_target = target

    def _search_behaviour(self):
        """Stages 3 + 5 + 6: unified motion controller.

        States:
          - Mission complete      → publish zero and do nothing further
          - No target set         → do nothing
          - Target set, invisible → rotate to search   (Stage 3)
          - Target visible, misaligned → turn to align (Stage 5)
          - Target visible, centred, far → drive forward (Stage 5)
          - Within STOP_DIST      → mission complete    (Stage 6)
        """
        msg = Twist()  # zero by default

        # Stage 6 — mission already done, keep publishing zero
        if self.mission_complete:
            self.cmd_vel_pub.publish(msg)
            return

        target = self.get_parameter('target_object').get_parameter_value().string_value.strip().lower()

        if not target:
            self.cmd_vel_pub.publish(msg)
            return

        if not self.target_visible:
            # Stage 3 — rotate in place to scan
            msg.angular.z = 0.3
            self.cmd_vel_pub.publish(msg)
            return

        # Stage 5 — target is visible
        error_x = self.target_error_x
        dist_m  = self.target_distance

        STOP_DIST    = 0.6   # metres — safe stopping distance
        ALIGN_PIXELS = 20    # pixels — alignment tolerance

        # Stage 6 — close enough → mission complete
        if dist_m is not None and dist_m <= STOP_DIST:
            # Publish zero velocity explicitly (belt and braces)
            self.cmd_vel_pub.publish(msg)

            if not self.mission_complete:
                self.mission_complete = True
                self.get_logger().info("=" * 40)
                self.get_logger().info("Mission Completed")
                self.get_logger().info("Target Reached Successfully")
                self.get_logger().info(f"Final distance: {dist_m:.2f} m")
                self.get_logger().info("=" * 40)
            return

        # Still approaching — align then drive
        if abs(error_x) > ALIGN_PIXELS:
            # Proportional turn toward target
            msg.angular.z = -0.004 * error_x
            msg.angular.z = max(-0.6, min(0.6, msg.angular.z))
        else:
            msg.linear.x = 0.2

        self.cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------

    def display_image(self):
        cv2.namedWindow("YOLO Detection",
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("YOLO Detection", 1600, 900)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                rgb   = None if self.latest_rgb   is None else self.latest_rgb.copy()
                depth = None if self.latest_depth is None else self.latest_depth.copy()
                self.latest_rgb   = None
                self.latest_depth = None

            if rgb is not None:
                result = self.run_yolo(rgb, depth)
                cv2.imshow("YOLO Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # YOLO inference + drawing
    # ------------------------------------------------------------------

    def run_yolo(self, frame, depth):
        target = self.get_parameter('target_object').get_parameter_value().string_value.strip().lower()

        CONF_THRESHOLD = 0.35
        results = self.model(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

        detections  = []
        target_dist = None
        found_now   = False   # did we see the target in THIS frame?

        h, w = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                # Stage 1 — filter to target
                if target and class_name.lower() != target:
                    continue

                found_now = True

                # Stage 2 — depth sampling
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Stage 5 — store centre x for alignment
                self._last_target_cx = cx
                dist_m = None

                if depth is not None:
                    sx = max(0, min(cx, w - 1))
                    sy = max(0, min(cy, h - 1))
                    patch = depth[max(0, sy-2):sy+3, max(0, sx-2):sx+3]
                    valid = patch[np.isfinite(patch) & (patch > 0)]
                    if valid.size > 0:
                        dist_m = float(np.median(valid))

                if dist_m is not None:
                    label = f"{class_name} {confidence:.2f} | {dist_m:.2f}m"
                    detections.append(f"{class_name} ({confidence:.2f}) — {dist_m:.2f} m")
                    if target_dist is None or dist_m < target_dist:
                        target_dist = dist_m
                else:
                    label = f"{class_name} {confidence:.2f}"
                    detections.append(f"{class_name} ({confidence:.2f})")

                color = self.class_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                text_y = max(y1 - 10, th + 10)
                cv2.rectangle(frame,
                              (x1, text_y - th - baseline),
                              (x1 + tw + 10, text_y + baseline),
                              color, -1)
                cv2.putText(frame, label, (x1 + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        # --- Stage 3: update visibility flag ---
        # Print "Target Found!" once when transitioning from not-found → found
        if found_now and not self.target_visible:
            self.get_logger().info(f"Target Found!")
        self.target_visible = found_now

        # --- Stage 5: store tracking info for the motion controller ---
        if found_now and target_dist is not None:
            self.target_distance = target_dist
            # error_x: how far the target centre is from the image centre
            # positive = target is right of centre, negative = left
            self.target_error_x  = self._last_target_cx - (w // 2)
            self.get_logger().info(
                f"Distance: {target_dist:.2f} m", throttle_duration_sec=0.5)
        elif not found_now:
            self.target_distance = None
            self.target_error_x  = 0

        # ---- FPS ----
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        # ---- Dashboard ----
        dw = 380
        dashboard = np.zeros((frame.shape[0], dw, 3), dtype=np.uint8)

        # Target / status
        if target:
            status_text  = f"Target: {target}"
            status_color = (0, 255, 0)
        else:
            status_text  = "Target: (all)"
            status_color = (0, 200, 255)

        cv2.putText(dashboard, status_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(dashboard, f"FPS    : {fps:.1f}", (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(dashboard, f"Found  : {len(detections)}", (15, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Stage 2 + 3 + 5 + 6 — status and distance
        if self.mission_complete:
            # Big green banner
            cv2.putText(dashboard, "MISSION", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(dashboard, "COMPLETED", (15, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(dashboard, "Target Reached", (15, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(dashboard, "Successfully", (15, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        elif target and self.target_visible and target_dist is not None:
            cv2.putText(dashboard, "Target Locked", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(dashboard, f"Distance: {target_dist:.2f} m", (15, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        elif target and self.target_visible:
            cv2.putText(dashboard, "Target Found!", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif target:
            cv2.putText(dashboard, "Searching...", (15, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Detection list
        y = 225
        for det in detections[:15]:
            cv2.putText(dashboard, det, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            y += 28

        # Hint at bottom
        cv2.putText(dashboard, "ros2 param set /yolo_detector", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)
        cv2.putText(dashboard, "  target_object <class>", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

        return np.hstack((frame, dashboard))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def class_color(self, class_id):
        np.random.seed(class_id)
        return tuple(int(c) for c in np.random.randint(100, 255, 3))


# ---------------------------------------------------------------------------

def main(args=None):
    print("OpenCV Version:", cv2.__version__)
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        node.display_image()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
