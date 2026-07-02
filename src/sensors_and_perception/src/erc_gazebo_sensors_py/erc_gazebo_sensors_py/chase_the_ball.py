#!/usr/bin/env python3
"""
chase_the_ball.py
-----------------
Subscribes to /camera/image, detects the largest red blob via colour
thresholding, and publishes /cmd_vel to steer the robot toward the ball.

Run:
    ros2 run erc_gazebo_sensors_py chase_the_ball
Then spawn the red_ball_10in model in Gazebo via the Resource Spawner.
Press Q to quit.
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

        # Spin in background so image_callback keeps receiving frames
        # even when display_image() is busy
        self.spin_thread = threading.Thread(target=self.spin_thread_func)
        self.spin_thread.start()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):
        with self.frame_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def stop(self):
        self.running = False
        self.spin_thread.join()

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------

    def display_image(self):
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800, 600)

        while rclpy.ok():
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                self.latest_frame = None

            if frame is not None:
                mask, contour, crosshair = self.process_image(frame)
                result = self.add_small_pictures(frame, [mask, contour, crosshair])
                cv2.imshow("frame", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
        self.running = False

    # ------------------------------------------------------------------
    # Image processing + ball chasing
    # ------------------------------------------------------------------

    def process_image(self, img):
        msg = Twist()
        rows, cols = img.shape[:2]

        R, G, B = self.convert2rgb(img)
        redMask = self.threshold_binary(R, (220, 255))

        stackedMask = np.dstack((redMask, redMask, redMask))
        contourMask = stackedMask.copy()
        crosshairMask = stackedMask.copy()

        (contours, hierarchy) = cv2.findContours(
            redMask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            cv2.drawContours(contourMask, contours, -1, (0, 255, 0), 10)
            cv2.circle(contourMask, (cx, cy), 5, (0, 255, 0), -1)

            cv2.line(crosshairMask, (cx, 0),           (cx, rows),        (0, 0, 255), 10)
            cv2.line(crosshairMask, (0, cy),            (cols, cy),        (0, 0, 255), 10)
            cv2.line(crosshairMask, (int(cols/2), 0),   (int(cols/2), rows), (255, 0, 0), 10)

            # Chase the ball
            if abs(cols / 2 - cx) > 20:
                msg.linear.x = 0.0
                msg.angular.z = 0.2 if cols / 2 > cx else -0.2
            else:
                msg.linear.x = 0.2
                msg.angular.z = 0.0
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher.publish(msg)
        return redMask, contourMask, crosshairMask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def convert2rgb(self, img):
        return img[:, :, 2], img[:, :, 1], img[:, :, 0]

    def threshold_binary(self, img, thresh=(200, 255)):
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return binary * 255

    def add_small_pictures(self, img, small_images, size=(160, 120)):
        x_base_offset = 40
        y_base_offset = 10
        x_offset = x_base_offset
        y_offset = y_base_offset
        for small in small_images:
            small = cv2.resize(small, size)
            if len(small.shape) == 2:
                small = np.dstack((small, small, small))
            img[y_offset:y_offset + size[1], x_offset:x_offset + size[0]] = small
            x_offset += size[0] + x_base_offset
        return img


# ---------------------------------------------------------------------------

def main(args=None):
    print("OpenCV version: %s" % cv2.__version__)
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
