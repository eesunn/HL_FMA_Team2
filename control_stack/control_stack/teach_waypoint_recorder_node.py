"""Teach-and-repeat waypoint recorder.

Drive the vehicle manually (e.g. TC275 PS2 joystick, ControlMode 2) while AMCL
localization is running. This node reads the live TF map->base_link (the robot's
real pose in the map frame) and auto-records a waypoint every time the robot has
moved a set distance (or turned a set angle). No RViz clicking needed, so the
recorded points always match where the vehicle physically is.

Output format is identical to pose_waypoint_recorder, so the printed
`waypoints:="x,y,yaw;..."` string can be fed straight into s_waypoint_runner for
NavigateThroughPoses replay in the continuous launch.

Inputs:
  TF map->base_link   (from AMCL map->odom + EKF odom->base_link)
Outputs:
  /waypoint_markers   (MarkerArray) — numbered arrows for RViz
  stdout              — waypoints param string + ready-to-run command
  file                — same string saved to <output_dir>/<session>.txt
"""

import math
import os
from datetime import datetime

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class TeachWaypointRecorderNode(Node):
    def __init__(self):
        super().__init__('teach_waypoint_recorder')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('record_distance_m', 0.20)
        self.declare_parameter('record_heading_deg', 25.0)  # 0 disables angle trigger
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('max_points', 0)             # 0 = unlimited
        self.declare_parameter('precision', 3)
        self.declare_parameter('auto_save', True)
        self.declare_parameter(
            'output_dir', os.path.expanduser('~/capstone_ws/waypoints'))

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.rec_dist = float(self.get_parameter('record_distance_m').value)
        self.rec_yaw = math.radians(
            float(self.get_parameter('record_heading_deg').value))
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.precision = int(self.get_parameter('precision').value)
        self.auto_save = bool(self.get_parameter('auto_save').value)
        self.output_dir = str(self.get_parameter('output_dir').value)

        self.poses = []  # list of (x, y, yaw)

        if self.auto_save:
            os.makedirs(self.output_dir, exist_ok=True)
            session = 'teach_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.save_path = os.path.join(self.output_dir, session + '.txt')
        else:
            self.save_path = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        marker_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/waypoint_markers', marker_qos)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)
        self._warned_no_tf = False

        self.get_logger().info(
            f'teach_waypoint_recorder ready — drive the robot; recording a '
            f'waypoint every {self.rec_dist:.2f} m'
            + (f' or {math.degrees(self.rec_yaw):.0f} deg turn'
               if self.rec_yaw > 0 else '')
            + f' (TF {self.map_frame}->{self.base_frame}).'
        )
        if self.save_path:
            self.get_logger().info(f'saving to {self.save_path}')

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
                yaw_from_quaternion(tf.transform.rotation))

    def tick(self) -> None:
        pose = self._get_pose()
        if pose is None:
            if not self._warned_no_tf:
                self.get_logger().warn(
                    'no map->base_link TF yet — did AMCL converge? '
                    '(set 2D Pose Estimate in RViz)',
                    throttle_duration_sec=2.0,
                )
            return
        self._warned_no_tf = False
        x, y, yaw = pose

        if not self.poses:
            self._record(x, y, yaw, reason='start')
            return

        lx, ly, lyaw = self.poses[-1]
        moved = math.hypot(x - lx, y - ly)
        turned = abs(normalize_angle(yaw - lyaw))
        if moved >= self.rec_dist or (self.rec_yaw > 0 and turned >= self.rec_yaw):
            self._record(x, y, yaw,
                         reason=f'd={moved:.2f}m, turn={math.degrees(turned):.0f}deg')

    def _record(self, x: float, y: float, yaw: float, reason: str) -> None:
        self.poses.append((x, y, yaw))
        idx = len(self.poses)
        self.get_logger().info(
            f'P{idx} recorded [{reason}]: '
            f'x={x:.{self.precision}f}, y={y:.{self.precision}f}, '
            f'yaw={yaw:.{self.precision}f} rad ({math.degrees(yaw):.1f} deg)'
        )
        self.publish_markers()
        self.print_current_waypoints()
        if self.save_path:
            self.save_to_file()

        if self.max_points > 0 and len(self.poses) >= self.max_points:
            self.get_logger().info('max_points reached — stopping recorder')
            rclpy.shutdown()

    # --- output helpers (format matches pose_waypoint_recorder) ---
    def format_pose(self, x: float, y: float, yaw: float) -> str:
        return (f'{x:.{self.precision}f},'
                f'{y:.{self.precision}f},'
                f'{yaw:.{self.precision}f}')

    def waypoints_param(self) -> str:
        return ';'.join(self.format_pose(x, y, yaw) for x, y, yaw in self.poses)

    def print_current_waypoints(self) -> None:
        param = self.waypoints_param()
        print('')
        print('--- current waypoints parameter with yaw ---')
        print(f'waypoints:="{param}"')
        print('')
        print('--- run this after recording is done ---')
        print('ros2 run control_stack s_waypoint_runner '
              f'--ros-args -p waypoints:="{param}"')
        print('')

    def save_to_file(self) -> None:
        param = self.waypoints_param()
        try:
            with open(self.save_path, 'w') as f:
                f.write('# teach-recorded waypoints (map frame)\n')
                f.write(f'# points: {len(self.poses)}\n\n')
                f.write(f'waypoints:="{param}"\n\n')
                f.write('ros2 run control_stack s_waypoint_runner '
                        f'--ros-args -p waypoints:="{param}"\n')
        except OSError as e:
            self.get_logger().warn(f'save failed: {e}', throttle_duration_sec=5.0)

    def publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for idx, (x, y, yaw) in enumerate(self.poses, start=1):
            qz = math.sin(yaw * 0.5)
            qw = math.cos(yaw * 0.5)

            arrow = Marker()
            arrow.header.frame_id = self.map_frame
            arrow.header.stamp = now
            arrow.ns = 's_waypoints_arrow'
            arrow.id = idx
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = 0.05
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x = 0.28
            arrow.scale.y = 0.055
            arrow.scale.z = 0.055
            arrow.color.r = 0.0
            arrow.color.g = 0.85
            arrow.color.b = 1.0
            arrow.color.a = 1.0
            markers.markers.append(arrow)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = now
            label.ns = 's_waypoints_label'
            label.id = idx
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color.r = 1.0
            label.color.g = 0.95
            label.color.b = 0.0
            label.color.a = 1.0
            label.text = str(idx)
            markers.markers.append(label)

        self.marker_pub.publish(markers)

    def on_shutdown(self) -> None:
        if not self.poses:
            self.get_logger().info('no waypoints recorded')
            return
        if self.save_path:
            self.save_to_file()
        self.get_logger().info(
            f'recorded {len(self.poses)} waypoints'
            + (f' → {self.save_path}' if self.save_path else ''))
        self.print_current_waypoints()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TeachWaypointRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.on_shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
