"""안전 클램프 노드.

`input_topic` (기본 /cmd_vel_raw) 을 받아 안전 처리 후 /cmd_vel 로 재발행.

처리:
  1. vx, vy, wz 를 max_vx / max_vy / max_wz 로 하드 클램프.
  2. allow_simultaneous=false 일 때: vx 와 wz 가 동시 0 이 아니면 회전 우선 (vx=0).
     segment 모드(직진과 회전 분리)에서 사용.
  3. allow_simultaneous=true 일 때: vx + vy + wz 동시 사용 허용 (continuous control).

용도:
  - segment 모드: input_topic=/cmd_vel_raw, allow_simultaneous=false
  - continuous 모드: input_topic=/cmd_vel_nav (Nav2 controller remap), allow_simultaneous=true
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class MotionSequencerNode(Node):
    def __init__(self):
        super().__init__('motion_sequencer_node')

        self.declare_parameter('input_topic', '/cmd_vel_raw')
        self.declare_parameter('max_vx', 0.2)
        self.declare_parameter('max_vy', 0.2)
        self.declare_parameter('max_wz', 0.6)
        self.declare_parameter('allow_simultaneous', False)

        self.declare_parameter('traffic_light_stop', True)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_wz = float(self.get_parameter('max_wz').value)
        self.allow_sim = bool(self.get_parameter('allow_simultaneous').value)
        self.traffic_light_stop = bool(self.get_parameter('traffic_light_stop').value)

        self._tl_state: str = 'NONE'

        self.sub = self.create_subscription(Twist, self.input_topic, self.on_raw, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(String, '/traffic_light', self._on_traffic_light, 10)

        self.get_logger().info(
            f'motion_sequencer_node ready (input={self.input_topic}, '
            f'max_vx={self.max_vx}, max_vy={self.max_vy}, max_wz={self.max_wz}, '
            f'allow_simultaneous={self.allow_sim}, '
            f'traffic_light_stop={self.traffic_light_stop})'
        )

    def _on_traffic_light(self, msg: String) -> None:
        prev = self._tl_state
        self._tl_state = msg.data
        if self._tl_state != prev:
            self.get_logger().info(f'[traffic_light] {prev} → {self._tl_state}')

    def on_raw(self, msg: Twist) -> None:
        # 신호등 RED → vx/vy/wz 모두 0
        if self.traffic_light_stop and self._tl_state == 'RED':
            self.get_logger().info(
                'RED 신호등 — 정지',
                throttle_duration_sec=1.0,
            )
            self.pub.publish(Twist())
            return

        vx = _clamp(msg.linear.x, -self.max_vx, self.max_vx)
        vy = _clamp(msg.linear.y, -self.max_vy, self.max_vy)
        wz = _clamp(msg.angular.z, -self.max_wz, self.max_wz)

        if not self.allow_sim and abs(wz) > 1e-3 and abs(vx) > 1e-3:
            self.get_logger().warn(
                'simultaneous vx and wz — gating vx to 0 (rotation priority).',
                throttle_duration_sec=2.0,
            )
            vx = 0.0

        out = Twist()
        out.linear.x = vx
        out.linear.y = vy
        out.angular.z = wz
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MotionSequencerNode()
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
