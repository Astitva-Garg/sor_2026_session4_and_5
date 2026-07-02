#!/usr/bin/env python3
"""
yolo_detection_node.py
----------------------
Subscribes to /camera/image and runs YOLOv8s inference on every frame.
Draws bounding boxes, labels, and a side-panel dashboard showing FPS
and the list of detected objects.

Usage (after building and sourcing):
    ros2 run erc_sor_ros_session1_py yolo_detection

Press Q or Escape to quit.
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # yolov8s is a good balance of speed vs accuracy for simulation
        self.model = YOLO('yolov8s.pt')
        self.get_logger().info('YOLO model loaded.')

        self.subscription = self.create_subscription(
            Image, 'camera/image', self.image_callback, 1)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.prev_time = time.time()

        self.spin_thread = threading.Thread(
            target=self._spin, daemon=True)
        self.spin_thread.start()

    # ------------------------------------------------------------------
    # ROS spin / callbacks
    # ------------------------------------------------------------------

    def _spin(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.frame_lock:
            self.latest_frame = frame

    def stop(self):
        self.running = False
        self.spin_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------

    def display_image(self):
        cv2.namedWindow('YOLO Detection',
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('YOLO Detection', 1280, 720)

        while rclpy.ok() and self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None \
                    else self.latest_frame.copy()

            if frame is not None:
                result = self._run_yolo(frame)
                cv2.imshow('YOLO Detection', result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # Q or Escape
                self.running = False
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # YOLO inference + drawing
    # ------------------------------------------------------------------

    def _run_yolo(self, frame):
        results = self.model(frame, conf=0.35, imgsz=640, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                detections.append(f'{class_name} ({confidence:.2f})')

                color = self._class_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f'{class_name} {confidence:.2f}'
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = max(y1 - 10, th + 10)
                cv2.rectangle(
                    frame,
                    (x1, text_y - th - baseline),
                    (x1 + tw + 10, text_y + baseline),
                    color, -1)
                cv2.putText(
                    frame, label, (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # centroid dot
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, color, -1)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        # Side dashboard
        dash_w = 320
        dashboard = np.zeros((frame.shape[0], dash_w, 3), dtype=np.uint8)
        cv2.putText(dashboard, 'Detections', (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(dashboard, f'FPS     : {fps:.1f}', (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(dashboard, f'Objects : {len(detections)}', (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y = 165
        for det in detections[:20]:
            cv2.putText(dashboard, det, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 28

        return np.hstack((frame, dashboard))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_color(class_id):
        rng = np.random.default_rng(class_id)
        return tuple(int(c) for c in rng.integers(100, 255, 3))


# ---------------------------------------------------------------------------

def main(args=None):
    print('OpenCV version:', cv2.__version__)
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
