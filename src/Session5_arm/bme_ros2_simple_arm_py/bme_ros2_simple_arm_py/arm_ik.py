"""
Custom Inverse Kinematics node for mogi_arm (4-DOF).

Kinematic chain (from URDF):
  base_link
    └─ shoulder_pan_joint  (z offset: +0.05 m, rotates about Z)  → θ1
       └─ shoulder_lift_joint (z offset: +0.025 m, rotates about Y) → θ2
          └─ elbow_joint      (z offset: +0.200 m, rotates about Y) → θ3
             └─ wrist_joint   (z offset: +0.250 m, rotates about Y) → θ4
                └─ end_effector_link (z offset: +0.175 m, fixed)

Link lengths used in the planar IK:
  L1 = 0.200  (upper arm:  shoulder_lift → elbow)
  L2 = 0.250  (forearm:    elbow → wrist)
  L3 = 0.175  (wrist → end_effector, fixed offset along wrist z-axis)

Shoulder base height above base_link:
  d0 = 0.05 + 0.025 = 0.075 m

IK derivation (4-DOF planar + pan):
  Step 1 – Pan angle (θ1):
      θ1 = atan2(y, x)

  Step 2 – Planar reach and height:
      r  = sqrt(x² + y²)          (horizontal distance from arm axis)
      pz = z - d0                  (height above shoulder pivot)

  Step 3 – Subtract wrist offset to find wrist position:
      The wrist-to-EE vector is always along the wrist's local z-axis.
      For a down-facing end-effector (wrist pointing straight up), that
      direction is the same as the planar arm direction.
      We assume the EE approaches from above (wrist angle keeps EE level),
      so we subtract L3 from the vertical:
          wz = pz - L3
          wr = r

  Step 4 – 2R IK for shoulder_lift (θ2) and elbow (θ3):
      D = (wr² + wz²) / (2·L1·L2)   via law of cosines
      θ3 = atan2(±sqrt(1 - D²), D)  (elbow-up solution used)
      θ2 = atan2(wz, wr) - atan2(L2·sin(θ3), L1 + L2·cos(θ3))

  Step 5 – Wrist angle (θ4) to keep end-effector level (pointing down):
      θ4 = -(θ2 + θ3)
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math


# ── Link lengths from URDF ──────────────────────────────────────────────────
D0 = 0.075   # shoulder pivot height above base_link (0.05 + 0.025)
L1 = 0.200   # upper arm length  (shoulder_lift → elbow)
L2 = 0.250   # forearm length    (elbow → wrist)
L3 = 0.175   # wrist → end_effector (fixed)


def solve_ik(x: float, y: float, z: float):
    """
    Solve 4-DOF IK for target position (x, y, z) in base_link frame.
    Returns (theta1, theta2, theta3, theta4) in radians, or None if unreachable.

    Joint mapping:
      theta1 → shoulder_pan_joint
      theta2 → shoulder_lift_joint
      theta3 → elbow_joint
      theta4 → wrist_joint
    """

    # Step 1: Pan (base rotation about Z)
    theta1 = math.atan2(y, x)

    # Step 2: Planar reach and height above shoulder pivot
    r  = math.sqrt(x**2 + y**2)
    pz = z - D0

    # Step 3: Subtract wrist-to-EE offset (EE kept level, pointing upward)
    wr = r
    wz = pz - L3

    # Step 4: 2R planar IK
    D = (wr**2 + wz**2) / (2.0 * L1 * L2)

    if abs(D) > 1.0:
        return None  # target unreachable

    # Elbow-up solution
    theta3 = math.atan2(math.sqrt(1.0 - D**2), D)

    # Shoulder lift
    alpha = math.atan2(wz, wr)
    beta  = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta

    # Step 5: Wrist keeps EE horizontal (level)
    theta4 = -(theta2 + theta3)

    # Clamp to joint limits from URDF
    limits = [
        (-3.14,  3.14),   # shoulder_pan
        (-1.5708, 1.5708), # shoulder_lift
        (-2.3562, 2.3562), # elbow
        (-2.3562, 2.3562), # wrist
    ]
    angles = [theta1, theta2, theta3, theta4]
    for i, (angle, (lo, hi)) in enumerate(zip(angles, limits)):
        if not (lo <= angle <= hi):
            print(f'  [IK] Joint {i+1} angle {math.degrees(angle):.1f}° out of limits [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]')
            return None

    return theta1, theta2, theta3, theta4


class ArmIKNode(Node):
    def __init__(self):
        super().__init__('arm_ik_node')

        # Target position in base_link frame (metres)
        # This example reaches forward-right and slightly above the base
        self.declare_parameter('target_x', 0.30)
        self.declare_parameter('target_y', 0.20)
        self.declare_parameter('target_z', 0.40)

        x = self.get_parameter('target_x').value
        y = self.get_parameter('target_y').value
        z = self.get_parameter('target_z').value

        self._arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        self._gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )

        self.get_logger().info(
            f'Solving IK for target: x={x:.3f}, y={y:.3f}, z={z:.3f} m'
        )

        result = solve_ik(x, y, z)

        if result is None:
            self.get_logger().error(
                'Target position is unreachable! Check coordinates and joint limits.'
            )
            return

        theta1, theta2, theta3, theta4 = result

        self.get_logger().info('IK solution found:')
        self.get_logger().info(f'  shoulder_pan_joint  (θ1): {math.degrees(theta1):+.2f}°  ({theta1:+.4f} rad)')
        self.get_logger().info(f'  shoulder_lift_joint (θ2): {math.degrees(theta2):+.2f}°  ({theta2:+.4f} rad)')
        self.get_logger().info(f'  elbow_joint         (θ3): {math.degrees(theta3):+.2f}°  ({theta3:+.4f} rad)')
        self.get_logger().info(f'  wrist_joint         (θ4): {math.degrees(theta4):+.2f}°  ({theta4:+.4f} rad)')

        # Verify forward kinematics
        ee_x = (L1*math.cos(theta2) + L2*math.cos(theta2+theta3) + L3*math.cos(theta2+theta3+theta4)) * math.cos(theta1)
        ee_y = (L1*math.cos(theta2) + L2*math.cos(theta2+theta3) + L3*math.cos(theta2+theta3+theta4)) * math.sin(theta1)
        ee_z = D0 + L1*math.sin(theta2) + L2*math.sin(theta2+theta3) + L3*math.sin(theta2+theta3+theta4)
        self.get_logger().info(
            f'FK verification → EE at: x={ee_x:.3f}, y={ee_y:.3f}, z={ee_z:.3f}'
        )

        # Wait a moment for controllers to be ready, then publish
        self._angles = (theta1, theta2, theta3, theta4)
        self._timer = self.create_timer(2.0, self._publish_trajectory)

    def _publish_trajectory(self):
        """Publish once then cancel the timer."""
        self._timer.cancel()

        theta1, theta2, theta3, theta4 = self._angles

        # ── Arm trajectory ──────────────────────────────────────────────────
        arm_msg = JointTrajectory()
        arm_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_joint',
        ]

        point = JointTrajectoryPoint()
        point.positions = [theta1, theta2, theta3, theta4]
        point.velocities = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=3, nanosec=0)

        arm_msg.points = [point]
        self._arm_pub.publish(arm_msg)

        # ── Gripper — open for approach ─────────────────────────────────────
        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = ['left_finger_joint', 'right_finger_joint']

        gpoint = JointTrajectoryPoint()
        gpoint.positions = [0.03, 0.03]   # open
        gpoint.velocities = [0.0, 0.0]
        gpoint.time_from_start = Duration(sec=3, nanosec=0)

        gripper_msg.points = [gpoint]
        self._gripper_pub.publish(gripper_msg)

        self.get_logger().info(
            'Trajectory published to /arm_controller/joint_trajectory '
            'and /gripper_controller/joint_trajectory'
        )
        self.get_logger().info('Arm is moving to target position...')


def main(args=None):
    rclpy.init(args=args)
    node = ArmIKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
