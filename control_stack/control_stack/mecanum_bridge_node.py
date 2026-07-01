"""메카넘 역운동학 노드 (use_strafe 로 모드 분기).

차량 휠은 표준 메카넘 X-패턴 (C↗ D↖ A↖ B↗). 두 가지 모드 지원:

[use_strafe = False] — Differential drive (segment 기반 stop-and-go 용)
  linear.y 는 무시.
  v_left  = vx - wz * Ly
  v_right = vx + wz * Ly
  A=C=v_left, B=D=v_right

[use_strafe = True]  — Full mecanum (continuous control + Nav2 Omni 용)
  linear.y 사용. 표준 X-패턴 역운동학.
  L = Lx + Ly
  v_A (RL) = vx - vy - wz * L
  v_B (RR) = vx + vy + wz * L
  v_C (FL) = vx + vy - wz * L
  v_D (FR) = vx - vy + wz * L

출력: /wheel_targets (Float32MultiArray, [A,B,C,D] km/h)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


MPS_TO_KMH = 3.6


class MecanumBridgeNode(Node):
    def __init__(self):
        super().__init__('mecanum_bridge_node')

        self.declare_parameter('Lx', 0.091)
        self.declare_parameter('Ly', 0.110)
        self.declare_parameter('use_strafe', False)

        self.Lx = float(self.get_parameter('Lx').value)
        self.Ly = float(self.get_parameter('Ly').value)
        self.L = self.Lx + self.Ly
        self.use_strafe = bool(self.get_parameter('use_strafe').value)

        self.sub = self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/wheel_targets', 10)

        mode = 'mecanum (strafe ON)' if self.use_strafe else 'diff-drive (strafe OFF)'
        self.get_logger().info(
            f'mecanum_bridge_node ready ({mode}, Lx={self.Lx}, Ly={self.Ly}, L={self.L})'
        )

    def on_cmd_vel(self, msg: Twist) -> None:
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        if self.use_strafe:
            v_a = vx - vy - wz * self.L
            v_b = vx + vy + wz * self.L
            v_c = vx + vy - wz * self.L
            v_d = vx - vy + wz * self.L
        else:
            if abs(vy) > 1e-3:
                self.get_logger().warn(
                    f'/cmd_vel.linear.y={vy:.3f} ignored (diff-drive mode).',
                    throttle_duration_sec=2.0,
                )
            v_left = vx - wz * self.Ly
            v_right = vx + wz * self.Ly
            v_a = v_left
            v_c = v_left
            v_b = v_right
            v_d = v_right

        out = Float32MultiArray()
        out.data = [
            float(v_a * MPS_TO_KMH),    # A (Rear Left)
            float(v_b * MPS_TO_KMH),    # B (Rear Right)
            float(v_c * MPS_TO_KMH),    # C (Front Left)
            float(v_d * MPS_TO_KMH),    # D (Front Right)
        ]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MecanumBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
