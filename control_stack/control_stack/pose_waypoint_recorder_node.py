"""Record RViz 2D pose clicks as x/y/yaw waypoint coordinates."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class PoseWaypointRecorderNode(Node):
    def __init__(self):
        super().__init__('pose_waypoint_recorder')

        self.declare_parameter('topic', '/waypoint_pose')
        self.declare_parameter('max_points', 0)
        self.declare_parameter('precision', 3)

        self.topic = str(self.get_parameter('topic').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.precision = int(self.get_parameter('precision').value)
        self.poses = []

        marker_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/waypoint_markers',
            marker_qos,
        )

        self.sub = self.create_subscription(
            PoseStamped,
            self.topic,
            self.on_pose,
            10,
        )

        self.get_logger().info(
            f'listening on {self.topic}. Use a 2D pose tool and click-drag '
            'each waypoint in driving order.'
        )
        if self.max_points > 0:
            self.get_logger().info(f'will stop after {self.max_points} poses')

    def on_pose(self, msg: PoseStamped) -> None:
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.orientation)
        self.poses.append((x, y, yaw))

        idx = len(self.poses)
        self.get_logger().info(
            f'P{idx}: frame={msg.header.frame_id}, '
            f'x={x:.{self.precision}f}, y={y:.{self.precision}f}, '
            f'yaw={yaw:.{self.precision}f} rad ({math.degrees(yaw):.1f} deg)'
        )
        self.publish_markers(msg.header.frame_id)
        self.print_current_waypoints()

        if self.max_points > 0 and len(self.poses) >= self.max_points:
            self.get_logger().info('max_points reached')
            rclpy.shutdown()

    def format_pose(self, x: float, y: float, yaw: float) -> str:
        return (
            f'{x:.{self.precision}f},'
            f'{y:.{self.precision}f},'
            f'{yaw:.{self.precision}f}'
        )

    def publish_markers(self, frame_id: str) -> None:
        frame = frame_id or 'map'
        now = self.get_clock().now().to_msg()

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for idx, (x, y, yaw) in enumerate(self.poses, start=1):
            qz = math.sin(yaw * 0.5)
            qw = math.cos(yaw * 0.5)

            arrow = Marker()
            arrow.header.frame_id = frame
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
            label.header.frame_id = frame
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

    def print_current_waypoints(self) -> None:
        param = ';'.join(self.format_pose(x, y, yaw) for x, y, yaw in self.poses)
        code_lines = [
            'DEFAULT_WAYPOINTS = [',
            *[
                (
                    f'    ({x:.{self.precision}f}, '
                    f'{y:.{self.precision}f}, '
                    f'{yaw:.{self.precision}f}),'
                )
                for x, y, yaw in self.poses
            ],
            ']',
        ]

        print('')
        print('--- current waypoints parameter with yaw ---')
        print(f'waypoints:="{param}"')
        print('')
        print('--- run this after recording is done ---')
        print(
            'ros2 run control_stack s_waypoint_runner '
            f'--ros-args -p waypoints:="{param}"'
        )
        print('')
        print('--- current DEFAULT_WAYPOINTS block ---')
        print('\n'.join(code_lines))
        print('')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseWaypointRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
