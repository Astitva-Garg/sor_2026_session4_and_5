"""
Custom Inverse Kinematics node for mogi_arm (4-DOF).

Kinematic chain (from URDF) — all revolute joints rotate about Y axis
except shoulder_pan which rotates about Z:

  base_link (at world origin)
    └─ shoulder_pan_joint   z=+0.050, rot Z  → θ1  (yaw/pan)
       └─ shoulder_lift_joint z=+0.025, rot Y → θ2  (pitch up/down)
          upper_arm_link length along local-Z = 0.200 m
          └─ elbow_joint    z=+0.200, rot Y  → θ3
             forearm_link length along local-Z = 0.250 m
             └─ wrist_joint z=+0.250, rot Y  → θ4
                wrist_link + gripper_base + end_effector
                total fixed length along local-Z = 0.175 m  → L3

Because every joint rotates about Y, the arm lies in a vertical plane
whose azimuth is set by θ1.  Within that plane the joint angles are
PITCH angles — positive rotates the tip UPWARD (i.e. away from Z=0).

Forward kinematics in the vertical plane (elevation angle convention):
  Let φ2 = θ2,  φ3 = θ2+θ3,  φ4 = θ2+θ3+θ4

  horizontal reach from shoulder pivot:
    r_fk = L1·cos(φ2) + L2·cos(φ3) + L3·cos(φ4)

  vertical rise from shoulder pivot:
    z_fk = L1·sin(φ2) + L2·sin(φ3) + L3·sin(φ4)

  EE in base_link frame:
    x = r_fk · cos(θ1)
    y = r_fk · sin(θ1)
    z = D0 + z_fk          where D0 = 0.075 m (shoulder pivot height)

Inverse kinematics:
  1.  θ1 = atan2(y, x)
  2.  r  = sqrt(x²+y²),   pz = z − D0
  3.  Find wrist position by removing the L3 EE segment.
      We want EE level (φ4 = 0, meaning wrist points horizontally),
      so the L3 segment is purely horizontal:
          wr = r  − L3        (wrist horizontal reach)
          wz = pz             (wrist height = EE height for level EE)
  4.  2-R planar IK on (wr, wz) with links L1, L2:
          D  = (wr²+wz² − L1²−L2²) / (2·L1·L2)   [cosine rule]
          θ3 = atan2(+sqrt(1−D²), D)               [elbow-up]
          θ2 = atan2(wz, wr) − atan2(L2·sin θ3, L1+L2·cos θ3)
  5.  θ4 = −(θ2+θ3)   → keeps EE horizontal (φ4 = 0)
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

# ── Link parameters from URDF ────────────────────────────────────────────────
D0 = 0.075   # shoulder pivot height above base_link  (0.050 + 0.025)
L1 = 0.200   # upper arm   (shoulder_lift_joint → elbow_joint)
L2 = 0.250   # forearm     (elbow_joint → wrist_joint)
L3 = 0.175   # wrist-to-EE (fixed, along local Z after wrist_joint)


def solve_ik(x: float, y: float, z: float):
    """
    Solve 4-DOF IK.  Returns (θ1,θ2,θ3,θ4) in radians, or None.
    All angles are about the joint's local Y axis (except θ1 about Z).
    """

    # ── Step 1: pan angle ────────────────────────────────────────────────────
    theta1 = math.atan2(y, x)

    # ── Step 2: horizontal reach and height above shoulder pivot ─────────────
    r  = math.sqrt(x**2 + y**2)
    pz = z - D0

    # ── Step 3: wrist position (EE kept horizontal → L3 is horizontal) ───────
    wr = r - L3      # horizontal reach to wrist joint
    wz = pz          # wrist height equals EE height (EE level)

    print(f'  [IK] r={r:.3f}  pz={pz:.3f}  wr={wr:.3f}  wz={wz:.3f}')

    # ── Step 4: 2-R IK ───────────────────────────────────────────────────────
    dist2 = wr**2 + wz**2
    D = (dist2 - L1**2 - L2**2) / (2.0 * L1 * L2)   # cosine rule

    print(f'  [IK] dist={math.sqrt(dist2):.3f}  D={D:.4f}')

    if abs(D) > 1.0:
        print(f'  [IK] Unreachable: |D|={abs(D):.3f} > 1')
        return None

    theta3 = math.atan2(math.sqrt(1.0 - D**2), D)   # elbow-up

    alpha  = math.atan2(wz, wr)
    beta   = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta

    # ── Step 5: wrist keeps EE level ─────────────────────────────────────────
    theta4 = -(theta2 + theta3)

    print(f'  [IK] θ1={math.degrees(theta1):+.1f}°  θ2={math.degrees(theta2):+.1f}°'
          f'  θ3={math.degrees(theta3):+.1f}°  θ4={math.degrees(theta4):+.1f}°')

    # ── Forward-kinematics verification ──────────────────────────────────────
    phi2 = theta2
    phi3 = theta2 + theta3
    phi4 = theta2 + theta3 + theta4          # should be ≈ 0

    r_fk  = L1*math.cos(phi2) + L2*math.cos(phi3) + L3*math.cos(phi4)
    z_fk  = L1*math.sin(phi2) + L2*math.sin(phi3) + L3*math.sin(phi4)
    ee_x  = r_fk * math.cos(theta1)
    ee_y  = r_fk * math.sin(theta1)
    ee_z  = D0 + z_fk
    err   = math.sqrt((ee_x-x)**2 + (ee_y-y)**2 + (ee_z-z)**2)
    print(f'  [FK]  EE=({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f})  '
          f'target=({x:.3f}, {y:.3f}, {z:.3f})  err={err*1000:.1f} mm')

    # ── Joint-limit check ─────────────────────────────────────────────────────
    limits = [
        (-3.14,   3.14),     # shoulder_pan_joint
        (-1.5708, 1.5708),   # shoulder_lift_joint
        (-2.3562, 2.3562),   # elbow_joint
        (-2.3562, 2.3562),   # wrist_joint
    ]
    names  = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist']
    angles = [theta1, theta2, theta3, theta4]
    ok = all(lo <= a <= hi for a, (lo, hi) in zip(angles, limits))

    if not ok:
        # Try elbow-down solution
        theta3 = math.atan2(-math.sqrt(1.0 - D**2), D)
        beta   = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
        theta2 = alpha - beta
        theta4 = -(theta2 + theta3)
        print(f'  [IK] Trying elbow-down: θ2={math.degrees(theta2):+.1f}°'
              f'  θ3={math.degrees(theta3):+.1f}°  θ4={math.degrees(theta4):+.1f}°')
        angles = [theta1, theta2, theta3, theta4]

    for angle, (lo, hi), name in zip(angles, limits, names):
        if not (lo <= angle <= hi):
            print(f'  [IK] {name}: {math.degrees(angle):.1f}° outside '
                  f'[{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]')
            return None

    return theta1, theta2, theta3, theta4


class ArmIKNode(Node):
    def __init__(self):
        super().__init__('arm_ik_node')

        # Default target — reachable position in front of the arm
        # The arm's max horizontal reach = L1+L2+L3 = 0.625 m
        # Keep wr = r-L3 <= L1+L2 = 0.45 m  →  r <= 0.625 m
        self.declare_parameter('target_x', 0.30)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 0.20)

        x = self.get_parameter('target_x').value
        y = self.get_parameter('target_y').value
        z = self.get_parameter('target_z').value

        self._arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self._gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # Detach the green cylinder immediately so it doesn't follow the arm
        from std_msgs.msg import Empty
        self._detach_pub = self.create_publisher(Empty, '/green/detach', 10)

        self.get_logger().info(
            f'Solving IK for target: x={x:.3f}, y={y:.3f}, z={z:.3f} m')

        result = solve_ik(x, y, z)

        if result is None:
            self.get_logger().error(
                'Target is unreachable — check coordinates and joint limits.')
            return

        theta1, theta2, theta3, theta4 = result

        self.get_logger().info('IK solution:')
        self.get_logger().info(
            f'  shoulder_pan  θ1 = {math.degrees(theta1):+7.2f}°  ({theta1:+.4f} rad)')
        self.get_logger().info(
            f'  shoulder_lift θ2 = {math.degrees(theta2):+7.2f}°  ({theta2:+.4f} rad)')
        self.get_logger().info(
            f'  elbow         θ3 = {math.degrees(theta3):+7.2f}°  ({theta3:+.4f} rad)')
        self.get_logger().info(
            f'  wrist         θ4 = {math.degrees(theta4):+7.2f}°  ({theta4:+.4f} rad)')

        self._angles = (theta1, theta2, theta3, theta4)
        self._publish_count = 0

        # First detach the green object, then start publishing the trajectory
        self._detach_timer = self.create_timer(0.5, self._detach_and_start)

    def _detach_and_start(self):
        """Publish detach once, then start trajectory publishing."""
        self._detach_timer.cancel()
        from std_msgs.msg import Empty
        self._detach_pub.publish(Empty())
        self.get_logger().info('Detached green cylinder.')
        # Start publishing trajectory after a short delay
        self._timer = self.create_timer(0.5, self._publish_trajectory)

    def _publish_trajectory(self):
        """Publish trajectory 3 times to ensure controller receives it."""
        self._publish_count += 1

        theta1, theta2, theta3, theta4 = self._angles

        arm_msg = JointTrajectory()
        arm_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_joint',
        ]
        pt = JointTrajectoryPoint()
        pt.positions  = [theta1, theta2, theta3, theta4]
        pt.velocities = [0.0, 0.0, 0.0, 0.0]
        pt.time_from_start = Duration(sec=4, nanosec=0)
        arm_msg.points = [pt]
        self._arm_pub.publish(arm_msg)

        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = ['left_finger_joint', 'right_finger_joint']
        gpt = JointTrajectoryPoint()
        gpt.positions  = [0.03, 0.03]   # open gripper
        gpt.velocities = [0.0, 0.0]
        gpt.time_from_start = Duration(sec=2, nanosec=0)
        gripper_msg.points = [gpt]
        self._gripper_pub.publish(gripper_msg)

        self.get_logger().info(
            f'Trajectory sent (attempt {self._publish_count}/3)')

        if self._publish_count >= 3:
            self._timer.cancel()
            self.get_logger().info(
                'Arm moving to target — motion completes in ~4 seconds.')
            self._done_timer = self.create_timer(6.0, self._shutdown)

    def _shutdown(self):
        self._done_timer.cancel()
        self.get_logger().info('Motion complete. Shutting down.')
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = ArmIKNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
