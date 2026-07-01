"""MotionSegment 시퀀스 실행 노드.

상태머신:
  IDLE       — segment 없음, cmd_vel 0
  ROTATING   — 현재 segment 가 ROTATE, target_yaw 도달까지 wz 발행
  STRAIGHT   — 현재 segment 가 STRAIGHT, distance 만큼 vx 발행
  SETTLE     — 한 segment 완료 후 정지 안정화 (stop_settle_time)

입력:
  /motion_segments      (MotionSegmentArray)
  TF map→base_link      (AMCL map→odom + EKF odom→base_link)

로봇 자세는 /plan 과 동일한 map 프레임에서 봐야 ROTATE 의 절대 target_yaw 가
일치한다. 그래서 /odometry/filtered(odom 프레임) 대신 TF map→base_link 를 쓴다.

출력:
  /cmd_vel_raw          (Twist) → motion_sequencer 가 클램프 후 /cmd_vel 로 전달
"""

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from capstone_msgs.msg import MotionSegment, MotionSegmentArray


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class State(Enum):
    IDLE = 0
    ROTATING = 1
    STRAIGHT = 2
    SETTLE = 3


class SegmentExecutorNode(Node):
    def __init__(self):
        super().__init__('segment_executor_node')

        self.declare_parameter('rotate_speed', 0.15)
        self.declare_parameter('straight_speed', 0.15)
        self.declare_parameter('rotate_tolerance_deg', 2.0)
        self.declare_parameter('straight_tolerance_m', 0.03)
        self.declare_parameter('stop_settle_time', 0.5)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('rotate_slowdown_deg', 15.0)
        self.declare_parameter('straight_slowdown_m', 0.15)
        self.declare_parameter('min_speed_floor', 0.05)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.rotate_speed = float(self.get_parameter('rotate_speed').value)
        self.straight_speed = float(self.get_parameter('straight_speed').value)
        self.rot_tol = math.radians(float(self.get_parameter('rotate_tolerance_deg').value))
        self.str_tol = float(self.get_parameter('straight_tolerance_m').value)
        self.settle_time = float(self.get_parameter('stop_settle_time').value)
        self.ctrl_rate = float(self.get_parameter('control_rate_hz').value)
        self.rot_slowdown = math.radians(float(self.get_parameter('rotate_slowdown_deg').value))
        self.str_slowdown = float(self.get_parameter('straight_slowdown_m').value)
        self.min_floor = float(self.get_parameter('min_speed_floor').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.segments = []
        self.seg_idx = 0
        self.state = State.IDLE
        self.start_x = 0.0
        self.start_y = 0.0
        self.settle_start = None

        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        self.have_pose = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_segs = self.create_subscription(
            MotionSegmentArray, '/motion_segments', self.on_segments, 10
        )
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        self.timer = self.create_timer(1.0 / self.ctrl_rate, self.tick)
        self.get_logger().info(
            f'segment_executor_node ready (pose from TF {self.map_frame}->{self.base_frame})'
        )

    def _update_pose(self) -> bool:
        """TF 로 map->base_link 자세를 갱신. 성공 시 True."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False
        self.cur_x = tf.transform.translation.x
        self.cur_y = tf.transform.translation.y
        self.cur_yaw = yaw_from_quat(tf.transform.rotation)
        self.have_pose = True
        return True

    def on_segments(self, msg: MotionSegmentArray) -> None:
        if not msg.segments:
            self.get_logger().info('empty segment array — going IDLE')
            self.segments = []
            self.seg_idx = 0
            self.state = State.IDLE
            return
        self.segments = list(msg.segments)
        self.seg_idx = 0
        self._update_pose()
        self._start_current_segment()
        self.get_logger().info(f'received {len(self.segments)} segments — starting')

    def _start_current_segment(self) -> None:
        if self.seg_idx >= len(self.segments):
            self.state = State.IDLE
            self.get_logger().info('all segments done')
            return
        seg = self.segments[self.seg_idx]
        self.start_x = self.cur_x
        self.start_y = self.cur_y
        if seg.type == MotionSegment.TYPE_ROTATE:
            self.state = State.ROTATING
            self.get_logger().info(
                f'seg[{self.seg_idx}] ROTATE → {math.degrees(seg.target_yaw):.1f} deg'
            )
        elif seg.type == MotionSegment.TYPE_STRAIGHT:
            self.state = State.STRAIGHT
            self.get_logger().info(
                f'seg[{self.seg_idx}] STRAIGHT distance={seg.distance:.3f} m'
            )
        else:
            self.get_logger().warn(f'unknown segment type {seg.type} — skipping')
            self._advance()

    def _advance(self) -> None:
        self.seg_idx += 1
        self.state = State.SETTLE
        self.settle_start = self.get_clock().now()

    def tick(self) -> None:
        cmd = Twist()
        pose_ok = self._update_pose()

        if self.state == State.IDLE:
            self.pub_cmd.publish(cmd)
            return

        # 실행 중인데 map->base_link TF 가 없으면 (AMCL 미수렴 등) 정지 유지.
        if not pose_ok:
            self.get_logger().warn(
                'no map->base_link TF — holding (2D Pose Estimate 했나?)',
                throttle_duration_sec=2.0,
            )
            self.pub_cmd.publish(cmd)
            return

        if self.state == State.SETTLE:
            elapsed = (self.get_clock().now() - self.settle_start).nanoseconds * 1e-9
            self.pub_cmd.publish(cmd)
            if elapsed >= self.settle_time:
                self._start_current_segment()
            return

        seg = self.segments[self.seg_idx]

        if self.state == State.ROTATING:
            err = normalize_angle(seg.target_yaw - self.cur_yaw)
            if abs(err) < self.rot_tol:
                self.pub_cmd.publish(cmd)
                self._advance()
                return
            speed = self.rotate_speed
            if abs(err) < self.rot_slowdown:
                speed = max(self.min_floor,
                            self.rotate_speed * abs(err) / self.rot_slowdown)
            cmd.angular.z = math.copysign(speed, err)
            self.pub_cmd.publish(cmd)
            return

        if self.state == State.STRAIGHT:
            traveled = math.hypot(self.cur_x - self.start_x,
                                  self.cur_y - self.start_y)
            target = abs(seg.distance)
            remaining = target - traveled
            if remaining < self.str_tol:
                self.pub_cmd.publish(cmd)
                self._advance()
                return
            speed = self.straight_speed
            if remaining < self.str_slowdown:
                speed = max(self.min_floor,
                            self.straight_speed * remaining / self.str_slowdown)
            cmd.linear.x = math.copysign(speed, seg.distance)
            self.pub_cmd.publish(cmd)
            return


def main(args=None):
    rclpy.init(args=args)
    node = SegmentExecutorNode()
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
