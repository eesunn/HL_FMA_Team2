"""차선 보정 명령 노드.

/lane_offset (Float32) + /lane_recovery_enable (Bool) 구독 →
enable=True일 때만 /cmd_vel_recovery (Twist) 발행.

/traffic_light (String) 구독:
  "RED"  → 즉시 정지 (차선 추종 일시 중단)
  그 외  → 정상 차선 추종 재개

좌표계 규칙 (CLAUDE.md §3): X=전방+, Y=좌측+, ωz=CCW+
부호:  offset > 0 (차선 중심이 오른쪽) → angular.z < 0 (시계방향 = 우회전) → 중앙으로 복귀.
"""

import csv
import datetime
import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


class LaneRecoveryNode(Node):
    def __init__(self):
        super().__init__('lane_recovery_node')

        self.declare_parameter('base_speed', 0.2)
        self.declare_parameter('k_steer', 0.6)
        self.declare_parameter('dead_zone', 0.05)
        self.declare_parameter('max_wz', 0.8)
        self.declare_parameter('max_vx', 0.8)
        self.declare_parameter('cmd_rate_hz', 20.0)
        self.declare_parameter('offset_timeout_sec', 0.35)
        self.declare_parameter('require_lane_valid', True)
        self.declare_parameter('auto_enable', False)
        self.declare_parameter('output_topic', '/cmd_vel_recovery')
        self.declare_parameter('log_csv', True)
        self.declare_parameter('capture_dir', os.path.expanduser('~/capstone_ws/logs'))
        self.declare_parameter('keyboard_stop', True)
        self.declare_parameter('wz_rate_limit', 0.5)      # rad/s²
        self.declare_parameter('invalid_hold_frames', 4)
        self.declare_parameter('speed_offset_factor', 0.5)
        self.declare_parameter('traffic_light_stop', True)
        self.declare_parameter('box_area_threshold', 2000.0)  # px² (보정 필요)
        self.declare_parameter('use_box_area',       True)    # False = 구형 동작 (RED만으로 정지)

        self.base_speed          = float(self.get_parameter('base_speed').value)
        self.k_steer             = float(self.get_parameter('k_steer').value)
        self.dead_zone           = float(self.get_parameter('dead_zone').value)
        self.max_wz              = float(self.get_parameter('max_wz').value)
        self.max_vx              = float(self.get_parameter('max_vx').value)
        self.cmd_rate_hz         = float(self.get_parameter('cmd_rate_hz').value)
        self.offset_timeout_sec  = float(self.get_parameter('offset_timeout_sec').value)
        self.require_lane_valid  = bool(self.get_parameter('require_lane_valid').value)
        self.output_topic        = str(self.get_parameter('output_topic').value)
        self.log_csv             = bool(self.get_parameter('log_csv').value)
        self.capture_dir         = str(self.get_parameter('capture_dir').value)
        self.keyboard_stop       = bool(self.get_parameter('keyboard_stop').value)
        self.wz_rate_limit        = float(self.get_parameter('wz_rate_limit').value)
        self.invalid_hold_frames  = int(self.get_parameter('invalid_hold_frames').value)
        self.speed_offset_factor  = float(self.get_parameter('speed_offset_factor').value)
        self.traffic_light_stop   = bool(self.get_parameter('traffic_light_stop').value)
        self.box_area_threshold   = float(self.get_parameter('box_area_threshold').value)
        self.use_box_area         = bool(self.get_parameter('use_box_area').value)

        self._enabled             = bool(self.get_parameter('auto_enable').value)
        self._latest_offset: float | None = None
        self._last_offset_time: float | None = None
        self._lane_valid          = False
        self._invalid_count       = 0
        self._prev_wz             = 0.0
        self._old_termios         = None
        self._tl_state            = 'NONE'   # /traffic_light 최신 값
        self._tl_box_area: float  = 0.0      # /traffic_light_box_area 최신 값
        self._box_area_last_time: float | None = None

        self._csv_f = None
        self._csv_writer = None
        if self.log_csv:
            os.makedirs(self.capture_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(self.capture_dir, f'lane_recovery_{ts}.csv')
            self._csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_f)
            self._csv_writer.writerow([
                'ts_sec',
                'enabled',
                'lane_valid',
                'traffic_light',
                'box_area_px2',
                'offset',
                'offset_age_sec',
                'offset_fresh',
                'reason',
                'cmd_vx',
                'cmd_vy',
                'cmd_wz',
                'base_speed',
                'k_steer',
                'dead_zone',
                'max_wz',
            ])
            self.get_logger().info(f'[RecoveryLog] {csv_path}')

        _latching_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_offset = self.create_subscription(
            Float32, '/lane_offset', self._on_offset, 10
        )
        self.sub_valid = self.create_subscription(
            Bool, '/lane_valid', self._on_lane_valid, _latching_qos
        )
        self.sub_enable = self.create_subscription(
            Bool, '/lane_recovery_enable', self._on_enable, 10
        )
        self.sub_tl = self.create_subscription(
            String, '/traffic_light', self._on_traffic_light, 10
        )
        self.sub_box_area = self.create_subscription(
            Float32, '/traffic_light_box_area', self._on_box_area, 10
        )
        self.pub = self.create_publisher(Twist, self.output_topic, 10)

        period = 1.0 / max(1.0, self.cmd_rate_hz)
        self.timer = self.create_timer(period, self._on_timer)
        self.key_timer = None
        if self.keyboard_stop and sys.stdin.isatty():
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.key_timer = self.create_timer(0.05, self._poll_keyboard)

        self.get_logger().info(
            f'lane_recovery_node ready  base_speed={self.base_speed}'
            f'  max_vx={self.max_vx}  k_steer={self.k_steer}  dead_zone={self.dead_zone}'
            f'  max_wz={self.max_wz}  output={self.output_topic}'
            f'  auto_enable={self._enabled}  require_lane_valid={self.require_lane_valid}'
            f'  traffic_light_stop={self.traffic_light_stop}'
            f'  use_box_area={self.use_box_area}  box_area_threshold={self.box_area_threshold:.0f}px²'
        )
        if self.key_timer is not None:
            self.get_logger().info('keyboard stop ready: press SPACE to stop')

    def _on_enable(self, msg: Bool) -> None:
        self._enabled = msg.data
        if not self._enabled:
            self._publish_zero('disabled')
            self.get_logger().info('lane_recovery disabled', throttle_duration_sec=2.0)
        else:
            self.get_logger().info('lane_recovery enabled', throttle_duration_sec=2.0)

    def _on_traffic_light(self, msg: String) -> None:
        prev = self._tl_state
        self._tl_state = msg.data
        if prev != self._tl_state:
            self.get_logger().info(f'[TL] {prev} → {self._tl_state}')

    def _on_box_area(self, msg: Float32) -> None:
        self._tl_box_area = msg.data
        self._box_area_last_time = self._now_sec()

    def _box_area_is_fresh(self) -> bool:
        if self._box_area_last_time is None:
            return False
        return (self._now_sec() - self._box_area_last_time) <= self.offset_timeout_sec

    def _on_lane_valid(self, msg: Bool) -> None:
        self._lane_valid = bool(msg.data)
        if self._lane_valid:
            self._invalid_count = 0
        else:
            self._invalid_count += 1

    def _on_offset(self, msg: Float32) -> None:
        self._latest_offset = msg.data
        self._last_offset_time = self._now_sec()

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _offset_is_fresh(self) -> bool:
        if self._last_offset_time is None:
            return False
        return (self._now_sec() - self._last_offset_time) <= self.offset_timeout_sec

    def _on_timer(self) -> None:
        if not self._enabled:
            return

        # 신호등 RED + 박스 크기 기반 정지 판단 (다른 조건보다 우선)
        if self.traffic_light_stop and self._tl_state == 'RED':
            # use_box_area=True 이고 박스 면적 데이터가 신선하면 → 크기 임계값 비교
            # 데이터 미수신(traffic_light_detector 미실행) 시 → 안전 동작으로 즉시 정지
            should_stop = True
            if self.use_box_area and self._box_area_is_fresh():
                should_stop = (self._tl_box_area >= self.box_area_threshold)
            if should_stop:
                self._publish_zero('traffic_light_red_stop')
                self.get_logger().warn(
                    f'RED 정지 — box_area={self._tl_box_area:.0f}px²'
                    f'  thr={self.box_area_threshold:.0f}',
                    throttle_duration_sec=1.0,
                )
                return
            # RED지만 박스 크기 미달 → 아직 멀리 있음, 계속 주행
            self.get_logger().info(
                f'RED 접근 중 — box_area={self._tl_box_area:.0f}px²'
                f'  thr={self.box_area_threshold:.0f} (정지 미도달)',
                throttle_duration_sec=2.0,
            )

        if self._latest_offset is None or not self._offset_is_fresh():
            self._publish_zero('stale_or_missing_offset')
            self.get_logger().warn(
                'lane offset stale or missing — publishing zero Twist',
                throttle_duration_sec=1.0,
            )
            return

        if self.require_lane_valid and self._invalid_count >= self.invalid_hold_frames:
            self._publish_zero('lane_invalid')
            self.get_logger().warn(
                'lane_valid=false — publishing zero Twist',
                throttle_duration_sec=1.0,
            )
            return

        self._publish_cmd(self._latest_offset)

    def _poll_keyboard(self) -> None:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return
        ch = sys.stdin.read(1)
        if ch == ' ':
            self._enabled = False
            self._publish_zero('keyboard_space_stop')
            self.get_logger().warn('SPACE pressed — lane_recovery stopped')

    def _publish_cmd(self, offset: float) -> None:
        wz_target = 0.0
        if abs(offset) >= self.dead_zone:
            raw = -self.k_steer * offset
            wz_target = max(-self.max_wz, min(self.max_wz, raw))

        dt = 1.0 / max(1.0, self.cmd_rate_hz)
        max_delta = self.wz_rate_limit * dt
        wz = self._prev_wz + max(-max_delta, min(max_delta, wz_target - self._prev_wz))
        self._prev_wz = wz

        speed_factor = max(0.4, 1.0 - self.speed_offset_factor * abs(offset))

        cmd = Twist()
        cmd.linear.x = min(self.base_speed * speed_factor, self.max_vx)
        cmd.linear.y = 0.0
        cmd.angular.z = wz
        try:
            self.pub.publish(cmd)
        except Exception:
            return
        self._write_csv('drive', cmd)

    def _publish_zero(self, reason: str = 'zero') -> None:
        dt = 1.0 / max(1.0, self.cmd_rate_hz)
        max_delta = self.wz_rate_limit * dt
        self._prev_wz += max(-max_delta, min(max_delta, -self._prev_wz))

        cmd = Twist()
        cmd.angular.z = self._prev_wz
        try:
            self.pub.publish(cmd)
        except Exception:
            return
        self._write_csv(reason, cmd)

    def _write_csv(self, reason: str, cmd: Twist) -> None:
        if self._csv_writer is None:
            return
        now = self._now_sec()
        age = ''
        fresh = False
        if self._last_offset_time is not None:
            age = now - self._last_offset_time
            fresh = age <= self.offset_timeout_sec
        offset = '' if self._latest_offset is None else self._latest_offset
        self._csv_writer.writerow([
            now,
            int(self._enabled),
            int(self._lane_valid),
            self._tl_state,
            round(self._tl_box_area, 1),
            offset,
            age,
            int(fresh),
            reason,
            cmd.linear.x,
            cmd.linear.y,
            cmd.angular.z,
            self.base_speed,
            self.k_steer,
            self.dead_zone,
            self.max_wz,
        ])
        self._csv_f.flush()


def main(args=None):
    rclpy.init(args=args)
    node = LaneRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, '_old_termios', None) is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node._old_termios)
        if rclpy.ok():
            node._publish_zero('shutdown')
        if getattr(node, '_csv_f', None) is not None:
            node._csv_f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
