"""자율 주행 로깅 노드 — 분석용 CSV 출력.

매 launch 마다 새 session 디렉토리(`<log_root>/<session_name>/`)를 만들고,
맵 파일(pgm/yaml)도 같이 복사해 자체 완결 패키지로 저장한다. 이후 zip 으로
Windows 로 옮겨 MATLAB 으로 시각화.

생성 파일
---------
  map_info.csv     : resolution, origin_x, origin_y, width, height, image (단일 행)
  goals.csv        : run_id, t, x, y, yaw                            (목표 받을 때마다)
  plans.csv        : run_id, point_idx, x, y                          (각 plan 의 전체 점)
  trajectory.csv   : t, run_id, x, y, yaw                             (TF 자세 주기 로그)
  arrivals.csv     : run_id, t, x, y, yaw, target_x, target_y,
                      target_yaw, error_xy, error_yaw_deg            (도달 시 1행)

세션 흐름
---------
  /goal_pose 수신 → run_id++ → goals.csv 한 줄
  /plan 수신       → 그 run_id 로 plans.csv 에 모든 점
  timer 10 Hz      → TF map→base_link 조회 → trajectory.csv
                      + 도달 검사: |xy|<tol 이고 |yaw|<tol 이며 1s 유지 → arrivals.csv
"""

import csv
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException


def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class NavigationLoggerNode(Node):
    def __init__(self):
        super().__init__('navigation_logger_node')

        self.declare_parameter('log_root_dir', os.path.expanduser('~/capstone_ws/logs'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('map_yaml_path',
                                os.path.expanduser('~/capstone_ws/maps/course_map.yaml'))
        self.declare_parameter('pose_log_rate_hz', 10.0)
        self.declare_parameter('arrival_tolerance_xy_m', 0.25)
        self.declare_parameter('arrival_tolerance_yaw_deg', 30.0)
        self.declare_parameter('arrival_hold_seconds', 1.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        log_root = str(self.get_parameter('log_root_dir').value)
        session_name = str(self.get_parameter('session_name').value).strip()
        if not session_name:
            session_name = 'session_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.session_dir = Path(log_root) / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.map_yaml_path = str(self.get_parameter('map_yaml_path').value)
        self.pose_rate = float(self.get_parameter('pose_log_rate_hz').value)
        self.tol_xy = float(self.get_parameter('arrival_tolerance_xy_m').value)
        self.tol_yaw = math.radians(
            float(self.get_parameter('arrival_tolerance_yaw_deg').value)
        )
        self.hold_sec = float(self.get_parameter('arrival_hold_seconds').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        # 맵 복사 (자체완결 위해)
        self._copy_map_files()

        # CSV 핸들들 (header 한 번 씀)
        self.f_goals = self._open_csv('goals.csv',
                                       ['run_id', 't', 'x', 'y', 'yaw'])
        self.f_plans = self._open_csv('plans.csv',
                                       ['run_id', 'point_idx', 'x', 'y'])
        self.f_traj = self._open_csv('trajectory.csv',
                                      ['t', 'run_id', 'x', 'y', 'yaw'])
        self.f_arr = self._open_csv('arrivals.csv',
                                     ['run_id', 't', 'x', 'y', 'yaw',
                                      'target_x', 'target_y', 'target_yaw',
                                      'error_xy', 'error_yaw_deg'])
        self.f_wp = self._open_csv('waypoints.csv', ['idx', 'x', 'y', 'yaw'])
        self.map_info_written = False
        self.wp_logged = False

        # 상태
        self.run_id = 0
        self.cur_goal = None       # (x, y, yaw)
        self.plan_logged = False
        self.arrival_logged = False
        self.in_tol_since = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 구독
        transient_local = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.sub_goal = self.create_subscription(
            PoseStamped, '/goal_pose', self.on_goal, 10)
        self.sub_plan = self.create_subscription(
            NavPath, '/plan', self.on_plan, 10)
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.on_map, transient_local)
        self.sub_waypoints = self.create_subscription(
            PoseArray, '/waypoints', self.on_waypoints, transient_local)

        self.timer = self.create_timer(1.0 / self.pose_rate, self.tick)

        self.get_logger().info(
            f'navigation_logger_node ready — logging to {self.session_dir}'
        )

    # ---------- helpers ----------
    def _open_csv(self, name, header):
        path = self.session_dir / name
        f = open(path, 'w', newline='')
        w = csv.writer(f)
        w.writerow(header)
        f.flush()
        return (f, w)

    def _write(self, handle, row):
        f, w = handle
        w.writerow(row)
        f.flush()

    def _copy_map_files(self):
        """map_yaml_path 와 동일 디렉토리의 pgm 을 세션 디렉토리에 복사."""
        try:
            yaml_p = Path(self.map_yaml_path)
            if not yaml_p.is_file():
                self.get_logger().warn(f'map yaml not found: {yaml_p}')
                return
            shutil.copy(yaml_p, self.session_dir / yaml_p.name)
            # yaml 파싱해서 image 필드 추출
            image_name = None
            with open(yaml_p) as f:
                for line in f:
                    if line.strip().startswith('image:'):
                        image_name = line.split(':', 1)[1].strip()
                        break
            if image_name:
                pgm_src = yaml_p.parent / image_name
                if pgm_src.is_file():
                    shutil.copy(pgm_src, self.session_dir / pgm_src.name)
                    self.get_logger().info(f'copied map: {pgm_src.name}')
                else:
                    self.get_logger().warn(f'map image not found: {pgm_src}')
        except Exception as e:
            self.get_logger().warn(f'_copy_map_files failed: {e}')

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- callbacks ----------
    def on_map(self, msg: OccupancyGrid):
        if self.map_info_written:
            return
        info = msg.info
        image_name = ''
        try:
            yaml_p = Path(self.map_yaml_path)
            if yaml_p.is_file():
                with open(yaml_p) as f:
                    for line in f:
                        if line.strip().startswith('image:'):
                            image_name = line.split(':', 1)[1].strip()
                            break
        except Exception:
            pass
        path = self.session_dir / 'map_info.csv'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['resolution', 'origin_x', 'origin_y',
                        'width', 'height', 'image'])
            w.writerow([info.resolution, info.origin.position.x,
                        info.origin.position.y, info.width, info.height,
                        image_name])
        self.map_info_written = True
        self.get_logger().info('map_info.csv written')

    def on_goal(self, msg: PoseStamped):
        # 이전 run 에 도달 안 했어도 새 run 시작 (이전은 그냥 미도달로 남음)
        self.run_id += 1
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = yaw_from_quat(msg.pose.orientation)
        self.cur_goal = (x, y, yaw)
        self.plan_logged = False
        self.arrival_logged = False
        self.in_tol_since = None
        self._write(self.f_goals, [self.run_id, self._now_sec(), x, y, yaw])
        self.get_logger().info(
            f'run #{self.run_id} goal: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)'
        )

    def on_waypoints(self, msg: PoseArray):
        # s_waypoint_runner 가 latched 로 보낸 목표 웨이포인트 → waypoints.csv (1회).
        if self.wp_logged:
            return
        for i, p in enumerate(msg.poses):
            yaw = yaw_from_quat(p.orientation)
            self._write(self.f_wp, [i, p.position.x, p.position.y, yaw])
        self.wp_logged = True
        self.get_logger().info(f'waypoints.csv: {len(msg.poses)} points logged')

    def on_plan(self, msg: NavPath):
        if self.run_id == 0 or self.plan_logged:
            return
        for i, p in enumerate(msg.poses):
            self._write(self.f_plans, [self.run_id, i,
                                       p.pose.position.x, p.pose.position.y])
        self.plan_logged = True
        self.get_logger().info(
            f'run #{self.run_id} plan: {len(msg.poses)} points logged'
        )

    def _get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        return (tf.transform.translation.x,
                tf.transform.translation.y,
                yaw_from_quat(tf.transform.rotation))

    def tick(self):
        pose = self._get_pose()
        if pose is None:
            return
        x, y, yaw = pose
        t = self._now_sec()
        self._write(self.f_traj, [t, self.run_id, x, y, yaw])

        # 도달 검사
        if self.cur_goal is None or self.arrival_logged:
            return
        gx, gy, gyaw = self.cur_goal
        err_xy = math.hypot(x - gx, y - gy)
        err_yaw = abs(normalize_angle(yaw - gyaw))
        if err_xy <= self.tol_xy and err_yaw <= self.tol_yaw:
            if self.in_tol_since is None:
                self.in_tol_since = t
            elif (t - self.in_tol_since) >= self.hold_sec:
                self._write(self.f_arr, [
                    self.run_id, t, x, y, yaw, gx, gy, gyaw,
                    err_xy, math.degrees(err_yaw),
                ])
                self.arrival_logged = True
                self.get_logger().info(
                    f'run #{self.run_id} ARRIVED: err_xy={err_xy:.3f}m, '
                    f'err_yaw={math.degrees(err_yaw):.1f}°'
                )
        else:
            self.in_tol_since = None

    def shutdown(self):
        for h in (self.f_goals, self.f_plans, self.f_traj, self.f_arr, self.f_wp):
            try:
                h[0].close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = NavigationLoggerNode()
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
