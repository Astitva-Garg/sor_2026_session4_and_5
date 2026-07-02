import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped
import math


def yaw_to_quaternion(yaw_rad):
    """Convert a yaw angle (radians) to a quaternion (x, y, z, w)."""
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return (0.0, 0.0, sy, cy)  # x, y, z, w


class WaypointPatroller(Node):
    def __init__(self):
        super().__init__('waypoint_patroller')

        # Action client for Nav2 FollowWaypoints
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        # Define patrol waypoints: (x, y, theta_degrees)
        # These are valid positions within the provided map
        self._waypoints_raw = [
            (0.89, 1.13, 135.0),    # Waypoint 1
            (0.82, -0.95, -50.0),   # Waypoint 2
            (-0.76, -0.10, 0.0),    # Waypoint 3
        ]

        self._current_waypoint_index = 0

        self.get_logger().info('Waypoint Patroller initialized.')
        self.get_logger().info(f'Patrol route has {len(self._waypoints_raw)} waypoints.')

        # Wait for Nav2 action server, then send goal
        self._send_patrol_goal()

    def _create_pose_stamped(self, x, y, theta_deg):
        """Create a PoseStamped message from x, y, theta (degrees)."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # Convert theta from degrees to quaternion
        theta_rad = math.radians(theta_deg)
        q = yaw_to_quaternion(theta_rad)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return pose

    def _send_patrol_goal(self):
        """Send the list of waypoints to Nav2 FollowWaypoints action."""
        self.get_logger().info('Waiting for Nav2 FollowWaypoints action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Nav2 FollowWaypoints action server is available.')

        # Build the goal message
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = []

        for i, (x, y, theta) in enumerate(self._waypoints_raw):
            pose = self._create_pose_stamped(x, y, theta)
            goal_msg.poses.append(pose)
            self.get_logger().info(
                f'  Waypoint {i + 1}: x={x:.2f}, y={y:.2f}, theta={theta:.1f} deg'
            )

        self.get_logger().info('Sending patrol goal...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        self._send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Called when the action server accepts/rejects the goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Patrol goal was rejected by Nav2!')
            return

        self.get_logger().info('Patrol goal accepted! Starting navigation...')
        self.get_logger().info('Navigating to Waypoint 1...')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)

    def _feedback_callback(self, feedback_msg):
        """Called every time Nav2 provides feedback on progress."""
        feedback = feedback_msg.feedback
        current_wp = feedback.current_waypoint

        # Detect when we've moved to a new waypoint
        if current_wp != self._current_waypoint_index:
            # The previous waypoint was reached
            self.get_logger().info(f'Waypoint {self._current_waypoint_index + 1} Reached!')
            self._current_waypoint_index = current_wp
            self.get_logger().info(
                f'Navigating to Waypoint {self._current_waypoint_index + 1}...'
            )

    def _get_result_callback(self, future):
        """Called when all waypoints are completed or the action finishes."""
        result = future.result().result
        missed = result.missed_waypoints

        # The last waypoint reached notification
        self.get_logger().info(
            f'Waypoint {len(self._waypoints_raw)} Reached!'
        )

        if len(missed) == 0:
            self.get_logger().info('Patrol complete! All waypoints reached successfully.')
        else:
            self.get_logger().warn(
                f'Patrol finished with {len(missed)} missed waypoint(s): {list(missed)}'
            )

        # Shutdown after patrol is complete
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPatroller()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
