"""CAN 브리지 노드 (differential-drive odometry).

송신:
  /wheel_targets (Float32MultiArray, [A,B,C,D] km/h) → 0x300 SpeedCommand (50 ms 주기)
  시작 시 0x301 ControlMode = 3 (ROS2 Autonomous) 1회 송신
  종료 시 0x300 = 0 (안전 정지)

수신:
  0x200 Rear_Feedback  (5 ms)
  0x201 Front_Feedback (5 ms)
  0x400 DiagnosticInfo (50 ms, 디버그용)

발행:
  /odom (nav_msgs/Odometry) — differential drive 정운동학 + 누적 적분
                              vy 는 항상 0. EKF가 융합 후 TF 를 발행하므로
                              본 노드는 odom→base_link TF 를 발행하지 않는다.
"""

import math
import struct
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

import can


KMH_TO_MPS = 1.0 / 3.6
CAN_ID_REAR_FB = 0x200
CAN_ID_FRONT_FB = 0x201
CAN_ID_DIAG = 0x400
CAN_ID_SPEED_CMD = 0x300
CAN_ID_CTRL_MODE = 0x301


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class CanBridgeNode(Node):
    def __init__(self):
        super().__init__('can_bridge_node')

        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('cmd_send_rate_hz', 20.0)
        self.declare_parameter('initial_ctrl_mode', 3)
        self.declare_parameter('ctrl_mode_resend_hz', 1.0)
        self.declare_parameter('Lx', 0.091)
        self.declare_parameter('Ly', 0.110)
        self.declare_parameter('use_strafe', False)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.can_iface = str(self.get_parameter('can_interface').value)
        self.cmd_rate = float(self.get_parameter('cmd_send_rate_hz').value)
        self.initial_mode = int(self.get_parameter('initial_ctrl_mode').value)
        self.ctrl_mode_resend_hz = float(
            self.get_parameter('ctrl_mode_resend_hz').value
        )
        self.Lx = float(self.get_parameter('Lx').value)
        self.Ly = float(self.get_parameter('Ly').value)
        self.L = self.Lx + self.Ly
        self.use_strafe = bool(self.get_parameter('use_strafe').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        self.target_kmh = [0.0, 0.0, 0.0, 0.0]
        self.wheel_speed_mps = [0.0, 0.0, 0.0, 0.0]
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_odom_stamp = None
        self.lock = threading.Lock()

        self.bus = can.interface.Bus(channel=self.can_iface, bustype='socketcan')
        self.get_logger().info(f'CAN interface opened: {self.can_iface}')

        self._send_ctrl_mode(self.initial_mode)

        self.sub_targets = self.create_subscription(
            Float32MultiArray, '/wheel_targets', self.on_wheel_targets, 10
        )
        self.sub_ctrl_mode = self.create_subscription(
            Int32, '/control_mode', self.on_control_mode, 10
        )
        self.pub_odom = self.create_publisher(Odometry, '/odom', 50)

        period = 1.0 / self.cmd_rate
        self.cmd_timer = self.create_timer(period, self.send_speed_cmd)
        self.ctrl_mode_timer = None
        if self.ctrl_mode_resend_hz > 0.0 and self.initial_mode >= 0:
            ctrl_period = 1.0 / self.ctrl_mode_resend_hz
            self.ctrl_mode_timer = self.create_timer(
                ctrl_period, self.resend_ctrl_mode
            )

        self.rx_thread_stop = False
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        mode = 'mecanum' if self.use_strafe else 'diff-drive'
        self.get_logger().info(
            f'can_bridge_node ready ({mode}, cmd {self.cmd_rate:.1f} Hz, '
            f'ctrl_mode={self.initial_mode}, '
            f'ctrl_resend={self.ctrl_mode_resend_hz:.1f} Hz, '
            f'Lx={self.Lx}, Ly={self.Ly})'
        )

    def _send_ctrl_mode(self, mode: int) -> None:
        msg = can.Message(
            arbitration_id=CAN_ID_CTRL_MODE,
            data=[mode & 0xFF],
            is_extended_id=False,
        )
        try:
            self.bus.send(msg)
            self.get_logger().info(f'TX 0x301 ControlMode={mode}')
        except can.CanError as e:
            self.get_logger().warn(f'TX 0x301 failed: {e}')

    def resend_ctrl_mode(self) -> None:
        self._send_ctrl_mode(self.initial_mode)

    def on_control_mode(self, msg: Int32) -> None:
        mode = int(msg.data)
        self.initial_mode = mode
        self._send_ctrl_mode(mode)
        self.get_logger().info(f'/control_mode requested: {mode}')

    def on_wheel_targets(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 4:
            self.get_logger().warn(f'/wheel_targets size != 4 (got {len(msg.data)})')
            return
        with self.lock:
            self.target_kmh = [float(v) for v in msg.data]

    def send_speed_cmd(self) -> None:
        with self.lock:
            vals_kmh = list(self.target_kmh)
        raw = []
        for v in vals_kmh:
            s = int(round(v * 1000.0))
            s = max(-32768, min(32767, s))
            raw.append(s)
        data = struct.pack('<hhhh', *raw)
        msg = can.Message(
            arbitration_id=CAN_ID_SPEED_CMD,
            data=data,
            is_extended_id=False,
        )
        try:
            self.bus.send(msg)
        except can.CanError as e:
            self.get_logger().warn(f'TX 0x300 failed: {e}', throttle_duration_sec=1.0)

    def _rx_loop(self) -> None:
        while not self.rx_thread_stop:
            try:
                msg = self.bus.recv(timeout=0.1)
            except can.CanError as e:
                if self.rx_thread_stop:
                    break
                self.get_logger().warn(f'CAN recv error: {e}', throttle_duration_sec=1.0)
                continue
            if msg is None:
                continue
            if msg.arbitration_id == CAN_ID_REAR_FB and len(msg.data) >= 8:
                a_kmh, b_kmh, _a_diff, _b_diff = struct.unpack('<hhhh', bytes(msg.data[:8]))
                with self.lock:
                    self.wheel_speed_mps[0] = (a_kmh / 1000.0) * KMH_TO_MPS
                    self.wheel_speed_mps[1] = (b_kmh / 1000.0) * KMH_TO_MPS
                self._update_odom()
            elif msg.arbitration_id == CAN_ID_FRONT_FB and len(msg.data) >= 8:
                c_kmh, d_kmh, _c_diff, _d_diff = struct.unpack('<hhhh', bytes(msg.data[:8]))
                with self.lock:
                    self.wheel_speed_mps[2] = (c_kmh / 1000.0) * KMH_TO_MPS
                    self.wheel_speed_mps[3] = (d_kmh / 1000.0) * KMH_TO_MPS
            elif msg.arbitration_id == CAN_ID_DIAG and len(msg.data) >= 8:
                self.get_logger().debug(
                    f'DIAG PWM A/B/C/D = {msg.data[0]}/{msg.data[1]}/{msg.data[2]}/{msg.data[3]}'
                )

    def _update_odom(self) -> None:
        with self.lock:
            va, vb, vc, vd = self.wheel_speed_mps
            x, y, theta = self.x, self.y, self.theta
            last = self.last_odom_stamp

        if self.use_strafe:
            # Full mecanum 정운동학.
            vx = (va + vb + vc + vd) / 4.0
            vy = (-va + vb + vc - vd) / 4.0
            wz = (-va + vb - vc + vd) / (4.0 * self.L)
        else:
            # Differential drive 정운동학.
            v_left_avg = (va + vc) / 2.0
            v_right_avg = (vb + vd) / 2.0
            vx = (v_left_avg + v_right_avg) / 2.0
            vy = 0.0
            wz = (v_right_avg - v_left_avg) / (2.0 * self.Ly)

        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        if last is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.1, now_sec - last))

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx_world = (vx * cos_t - vy * sin_t) * dt
        dy_world = (vx * sin_t + vy * cos_t) * dt
        dtheta = wz * dt

        new_x = x + dx_world
        new_y = y + dy_world
        new_theta = math.atan2(math.sin(theta + dtheta), math.cos(theta + dtheta))

        with self.lock:
            self.x = new_x
            self.y = new_y
            self.theta = new_theta
            self.last_odom_stamp = now_sec

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = new_x
        odom.pose.pose.position.y = new_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quat(new_theta)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        # 슬립 환경 대응: vx 만 EKF에 신뢰시키고 yaw/wz 는 IMU 가 책임지도록
        # covariance 를 비대칭으로 셋팅한다. (실제 EKF 필터링은 ekf_params 의
        # *_config 마스크가 결정하지만, covariance 도 함께 표기해 둠.)
        odom.pose.covariance[0] = 0.05   # x
        odom.pose.covariance[7] = 0.05   # y
        odom.pose.covariance[35] = 1.0   # yaw (high — do not trust)
        odom.twist.covariance[0] = 0.05  # vx
        odom.twist.covariance[7] = 1.0   # vy (always 0)
        odom.twist.covariance[35] = 1.0  # wz (high — IMU 가 책임)
        self.pub_odom.publish(odom)

    def shutdown(self) -> None:
        self.rx_thread_stop = True
        try:
            stop_msg = can.Message(
                arbitration_id=CAN_ID_SPEED_CMD,
                data=b'\x00' * 8,
                is_extended_id=False,
            )
            self.bus.send(stop_msg)
            self.get_logger().info('TX 0x300 = 0 (safe stop on shutdown)')
        except Exception as e:
            self.get_logger().warn(f'shutdown stop TX failed: {e}')
        try:
            self.bus.shutdown()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CanBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
