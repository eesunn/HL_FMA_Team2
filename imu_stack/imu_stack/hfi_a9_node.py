"""HFI-A9 (TBA9) IMU 드라이버 노드.

USB 시리얼 (921600 bps) 로 IMU 와 통신하여 sensor_msgs/Imu 를 발행한다.

프로토콜 (raw 캡처로 확정, 2026-05-27):
  - 0xAA 0x55 sync + payload-len 표시 byte + msg id + a0 03 00 + timestamp + float32 LE 들
  - 0x14 (25B): Euler  [roll°, pitch°, yaw°]
  - 0x2c (49B): Sensor [gyro xyz, accel xyz (g), mag xyz]
  - yaw 는 compass 규약 (CW=+) 이라 ROS (CCW=+) 로 부호 반전해 발행.

발행:
  /imu/data (sensor_msgs/Imu) — orientation(roll/pitch/yaw), angular_velocity(gyro),
  linear_acceleration(accel, m/s²). 50 Hz 다운샘플.

미검증 항목:
  - gyro z 부호: yaw 와 같은 규약으로 임시 반전. 회전 테스트로 검증 후 확정 필요.
  - EKF 는 현재 yaw(orientation, differential) 만 사용하므로 gyro 부호는 /imu/data
    표시에만 영향. EKF 영향 없음.
"""

import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField

try:
    import serial
except ImportError:
    serial = None


# HFI-A9 프레임 구조 (관측으로 확정, 2026-05-27):
#   [0:2] aa 55 (sync)
#   [2]   payload length 표시 (0x14=20→frame 25B, 0x2c=44→frame 49B)
#   [3]   message id (0x23 / 0x29)
#   [4:7] a0 03 00 (상수 태그)
#   [7:11]  timestamp (uint32, 사용 안 함)
#   이후 float32 LE:
#     0x14 (Euler):  [11]roll° [15]pitch° [19]yaw°
#     0x2c (Sensor): [11]gx [15]gy [19]gz [23]ax [27]ay [31]az [35]mx [39]my [43]mz
GRAVITY = 9.80665
FRAME_LEN = {0x14: 25, 0x2c: 49}


def crc16_modbus(data: bytes) -> int:
    """Modbus CRC16 (HFI-A9 프레임 검증용). 벤더/친구 코드와 동일 알고리즘."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def yaw_to_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class HfiA9Node(Node):
    def __init__(self):
        super().__init__('hfi_a9_node')

        self.declare_parameter('port', '/dev/ttyUSB_IMU')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate_hz', 50.0)
        # yaw 소스: 'device' = IMU 내장 융합 yaw (느린 드리프트 있으나 안정. SLAM loop
        #           closure 가 보정). 'mag' = 지자계 atan2 — 현재 우리 유닛은 mag 가
        #           정지 중에도 출렁(모터 간섭/미캘리)이라 비활성. 캘리/재배치 후에만 사용.
        self.declare_parameter('yaw_source', 'device')
        self.declare_parameter('mag_lpf_alpha', 0.98)     # 친구 코드와 동일
        self.declare_parameter('invert_mag_yaw', False)   # CCW+ 안 맞으면 true

        self.port = str(self.get_parameter('port').value)
        self.baud = int(self.get_parameter('baudrate').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.pub_rate = float(self.get_parameter('publish_rate_hz').value)
        self.yaw_source = str(self.get_parameter('yaw_source').value)
        self.mag_alpha = float(self.get_parameter('mag_lpf_alpha').value)
        self.invert_mag_yaw = bool(self.get_parameter('invert_mag_yaw').value)
        self._mag_fc = None   # 벡터 LPF 상태 (cos) — ±180 wrap 안전
        self._mag_fs = None   # 벡터 LPF 상태 (sin)

        if serial is None:
            self.get_logger().error(
                'pyserial not installed. Run: pip install pyserial'
            )

        # CP210x USB-serial 의 "device reports readiness but returned no data" 같은
        # 일시적 오류를 throttle 하기 위한 카운터.
        self._zero_read_count = 0
        self._crc_fail = 0

        self.latest = {
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'gx': 0.0, 'gy': 0.0, 'gz': 0.0,
            'ax': 0.0, 'ay': 0.0, 'az': 0.0,
            'mx': 0.0, 'my': 0.0, 'mz': 0.0,
            'updated': False,
        }
        self.lock = threading.Lock()

        self.ser = None
        if serial is not None:
            try:
                # timeout 0.5 s — CP210x 의 select() 노이즈를 줄임.
                self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
                self.get_logger().info(f'opened {self.port} @ {self.baud}')
            except Exception as e:
                self.get_logger().error(f'failed to open {self.port}: {e}')

        self.pub = self.create_publisher(Imu, '/imu/data', 50)
        # raw 지자계 — 0x2c 프레임의 mx/my/mz. 단위 미상(벤더 미공개)이라 그대로 publish.
        # madgwick 융합은 방향만 쓰므로 단위 무관. 절대 heading 복구용 진단/입력.
        self.pub_mag = self.create_publisher(MagneticField, '/imu/mag', 50)

        self.rx_thread_stop = False
        if self.ser is not None:
            self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self.rx_thread.start()

        self.timer = self.create_timer(1.0 / self.pub_rate, self._publish)
        self.get_logger().info(
            f'hfi_a9_node ready (publish {self.pub_rate:.1f} Hz, frame={self.frame_id})'
        )

    def _rx_loop(self) -> None:
        buf = bytearray()
        while not self.rx_thread_stop:
            try:
                data = self.ser.read(64)
            except serial.SerialException as e:
                if self.rx_thread_stop:
                    break
                # CP210x 의 "device reports readiness but returned no data" 등
                # 일시적 오류는 짧게 sleep 후 재시도. 누적해서 많이 나오면 한 번 알림.
                self._zero_read_count += 1
                if self._zero_read_count % 50 == 1:
                    self.get_logger().warn(
                        f'serial read error (#{self._zero_read_count}): {e}'
                    )
                time.sleep(0.02)
                continue
            except Exception as e:
                if self.rx_thread_stop:
                    break
                self.get_logger().warn(f'unexpected serial error: {e}',
                                       throttle_duration_sec=2.0)
                time.sleep(0.05)
                continue
            if not data:
                continue
            self._zero_read_count = 0
            buf.extend(data)
            # 프레임 단위로 잘라서 파싱.
            while True:
                consumed = self._parse_frame(buf)
                if consumed <= 0:
                    break
                del buf[:consumed]
            # 버퍼 폭주 방지.
            if len(buf) > 4096:
                del buf[:-1024]

    def _parse_frame(self, buf: bytearray) -> int:
        """버퍼에서 한 프레임을 파싱하고 사용한 바이트 수를 반환 (0 = 더 받아야 함)."""
        if len(buf) < 3:
            return 0
        # sync 정렬.
        if not (buf[0] == 0xAA and buf[1] == 0x55):
            i = buf.find(b'\xaa\x55')
            if i < 0:
                return max(0, len(buf) - 1)   # 마지막 0xAA 가능성만 남김
            return i
        flen = FRAME_LEN.get(buf[2])
        if flen is None:
            return 1   # 알 수 없는 타입 → 1바이트 버리고 재동기
        if len(buf) < flen:
            return 0   # 프레임 완성될 때까지 대기
        frame = bytes(buf[:flen])

        # CRC 검증 — 통과한 프레임만 사용. 실패=오정렬/노이즈 → 1바이트 버리고 재동기.
        # (이게 없으면 우연한 aa55 에 오정렬돼 mag 등 프레임 뒷부분이 garbage 로 읽힘)
        crc = crc16_modbus(frame[2:flen - 2])
        chk = frame[flen - 2:flen]
        if (((crc & 0xff) << 8) | (crc >> 8)) != ((chk[0] << 8) | chk[1]):
            self._crc_fail += 1
            if self._crc_fail % 200 == 1:
                self.get_logger().warn(
                    f'CRC fail x{self._crc_fail} (프레임 오정렬/노이즈 폐기 중)'
                )
            return 1

        mtype = frame[2]
        if mtype == 0x14:
            roll = struct.unpack_from('<f', frame, 11)[0]
            pitch = struct.unpack_from('<f', frame, 15)[0]
            yaw_deg = struct.unpack_from('<f', frame, 19)[0]
            with self.lock:
                self.latest['roll'] = math.radians(roll)
                self.latest['pitch'] = math.radians(pitch)
                # 센서는 CW=+ (compass). ROS 는 CCW=+ → 부호 반전.
                self.latest['yaw'] = -math.radians(yaw_deg)
                self.latest['updated'] = True
        elif mtype == 0x2c:
            gx = struct.unpack_from('<f', frame, 11)[0]
            gy = struct.unpack_from('<f', frame, 15)[0]
            gz = struct.unpack_from('<f', frame, 19)[0]
            ax = struct.unpack_from('<f', frame, 23)[0]
            ay = struct.unpack_from('<f', frame, 27)[0]
            az = struct.unpack_from('<f', frame, 31)[0]
            mx = struct.unpack_from('<f', frame, 35)[0]
            my = struct.unpack_from('<f', frame, 39)[0]
            mz = struct.unpack_from('<f', frame, 43)[0]
            with self.lock:
                self.latest['gx'] = gx
                self.latest['gy'] = gy
                # yaw 와 같은 부호 규약으로 맞춤 (CW→CCW). 실제 부호는 회전 테스트로 검증 필요.
                self.latest['gz'] = -gz
                self.latest['ax'] = ax * GRAVITY
                self.latest['ay'] = ay * GRAVITY
                self.latest['az'] = az * GRAVITY
                # raw 지자계 (단위 미상). 축 정렬/부호는 융합 단계에서 검증 후 확정.
                self.latest['mx'] = mx
                self.latest['my'] = my
                self.latest['mz'] = mz
        return flen

    def _compute_yaw(self, d) -> float:
        """yaw_source 에 따라 heading 결정.

        'mag'   : 지자계 atan2(my, mx) → 절대·drift-free heading (친구 해법).
                  ±180 경계 wrap 을 안전하게 처리하려고 각도 직접 LPF 가 아니라
                  cos/sin 벡터를 LPF 한 뒤 atan2 (친구 코드의 wrap 버그 보완).
                  ⚠ tilt 보정 없음(평지 가정) — 경사(hill)에선 오차 가능.
        'device': IMU 내장 융합 yaw (드리프트 관찰됨 → 비권장).
        """
        if self.yaw_source != 'mag':
            return d['yaw']
        mx, my = d['mx'], d['my']
        if mx == 0.0 and my == 0.0:               # mag 아직 없음
            return (math.atan2(self._mag_fs, self._mag_fc)
                    if self._mag_fc is not None else d['yaw'])
        raw = math.atan2(my, mx)
        if self.invert_mag_yaw:
            raw = -raw
        c, s = math.cos(raw), math.sin(raw)
        if self._mag_fc is None:
            self._mag_fc, self._mag_fs = c, s
        else:
            a = self.mag_alpha
            self._mag_fc = a * self._mag_fc + (1.0 - a) * c
            self._mag_fs = a * self._mag_fs + (1.0 - a) * s
        return math.atan2(self._mag_fs, self._mag_fc)

    def _publish(self) -> None:
        with self.lock:
            d = dict(self.latest)
            self.latest['updated'] = False

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        yaw = self._compute_yaw(d)
        qx, qy, qz, qw = yaw_to_quat(yaw, d['pitch'], d['roll'])
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.angular_velocity.x = d['gx']
        msg.angular_velocity.y = d['gy']
        msg.angular_velocity.z = d['gz']
        msg.linear_acceleration.x = d['ax']
        msg.linear_acceleration.y = d['ay']
        msg.linear_acceleration.z = d['az']
        # 9축 칼만이 내장이므로 비교적 작은 covariance 적용.
        msg.orientation_covariance[0] = 0.01
        msg.orientation_covariance[4] = 0.01
        msg.orientation_covariance[8] = 0.01
        msg.angular_velocity_covariance[0] = 0.005
        msg.angular_velocity_covariance[4] = 0.005
        msg.angular_velocity_covariance[8] = 0.005
        msg.linear_acceleration_covariance[0] = 0.05
        msg.linear_acceleration_covariance[4] = 0.05
        msg.linear_acceleration_covariance[8] = 0.05
        self.pub.publish(msg)

        mag = MagneticField()
        mag.header.stamp = msg.header.stamp
        mag.header.frame_id = self.frame_id
        mag.magnetic_field.x = d['mx']
        mag.magnetic_field.y = d['my']
        mag.magnetic_field.z = d['mz']
        # 단위/캘리 미확정 → covariance 0 (unknown) 으로 둠.
        self.pub_mag.publish(mag)

    def shutdown(self) -> None:
        self.rx_thread_stop = True
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = HfiA9Node()
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
