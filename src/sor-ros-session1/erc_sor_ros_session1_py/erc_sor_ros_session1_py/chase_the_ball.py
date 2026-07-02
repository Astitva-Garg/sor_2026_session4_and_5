#!/usr/bin/env python3
"""
chase_the_ball.py
-----------------
Subscribes to /camera/image, detects the largest red blob via colour
thresholding, and publishes /cmd_vel to steer the robot toward the ball.

Four OpenCV windows are composited into a single view:
  [original] [red mask] [contour] [crosshair]
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image, 'camera/image', self.image_callback, 1)

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Spin in a background thread so image_callback never stalls
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
        cv2.namedWindow('chase_the_ball', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('chase_the_ball', 800, 300)

        while rclpy.ok():
            with self.frame_lock:
                frame = None if self.latest_frame is None \
                    else self.latest_frame.copy()
                self.latest_frame = None

            if frame is not None:
                mask, contour, crosshair = self.process_image(frame)
                result = self.add_small_pictures(
                    frame, [mask, contour, crosshair])
                cv2.imshow('chase_the_ball', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_image(self, img):
        """Detect the largest red blob and publish a cmd_vel."""
        msg = Twist()
        rows, cols = img.shape[:2]

        R, G, B = self._to_rgb(img)
        red_mask = self._threshold_binary(R, (220, 255))

        stacked = np.dstack((red_mask, red_mask, red_mask))
        contour_vis = stacked.copy()
        crosshair_vis = stacked.copy()

        contours, _ = cv2.findContours(
            red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 10)
            cv2.circle(contour_vis, (cx, cy), 5, (0, 255, 0), -1)

            cv2.line(crosshair_vis, (cx, 0),          (cx, rows),        (0, 0, 255), 10)
            cv2.line(crosshair_vis, (0, cy),           (cols, cy),        (0, 0, 255), 10)
            cv2.line(crosshair_vis, (cols // 2, 0),    (cols // 2, rows), (255, 0, 0), 10)

            # Steering logic
            if abs(cols / 2 - cx) > 20:
                msg.linear.x = 0.0
                msg.angular.z = 0.2 if cols / 2 > cx else -0.2
            else:
                msg.linear.x = 0.2
                msg.angular.z = 0.0
        else:
            # Ball not in view — stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher.publish(msg)
        return red_mask, contour_vis, crosshair_vis

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb(img):
        return img[:, :, 2], img[:, :, 1], img[:, :, 0]

    @staticmethod
    def _threshold_binary(img, thresh=(200, 255)):
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return (binary * 255).astype(np.uint8)

    @staticmethod
    def add_small_pictures(img, small_images, size=(160, 120)):
        """Overlay thumbnail views across the top of *img* in-place."""
        x_offset = 40
        y_offset = 10
        for small in small_images:
            small = cv2.resize(small, size)
            if small.ndim == 2:
                small = np.dstack((small, small, small))
            img[y_offset:y_offset + size[1],
                x_offset:x_offset + size[0]] = small
            x_offset += size[0] + 40
        return img


# ---------------------------------------------------------------------------

def main(args=None):
    print('OpenCV version:', cv2.__version__)
    rclpy.init(args=args)
    node = ImageSubscriber()
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
