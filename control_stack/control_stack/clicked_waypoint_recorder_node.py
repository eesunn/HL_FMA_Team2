"""Record RViz Publish Point clicks as waypoint coordinates."""

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


class ClickedWaypointRecorderNode(Node):
    def __init__(self):
        super().__init__('clicked_waypoint_recorder')

        self.declare_parameter('topic', '/clicked_point')
        self.declare_parameter('max_points', 0)
        self.declare_parameter('precision', 3)

        self.topic = str(self.get_parameter('topic').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.precision = int(self.get_parameter('precision').value)
        self.points = []

        self.sub = self.create_subscription(
            PointStamped,
            self.topic,
            self.on_point,
            10,
        )

        self.get_logger().info(
            f'listening on {self.topic}. Use RViz Publish Point and click '
            'waypoints in driving order.'
        )
        if self.max_points > 0:
            self.get_logger().info(f'will stop after {self.max_points} points')

    def on_point(self, msg: PointStamped) -> None:
        x = msg.point.x
        y = msg.point.y
        self.points.append((x, y))

        idx = len(self.points)
        self.get_logger().info(
            f'P{idx}: frame={msg.header.frame_id}, '
            f'x={x:.{self.precision}f}, y={y:.{self.precision}f}'
        )
        self.print_current_waypoints()

        if self.max_points > 0 and len(self.points) >= self.max_points:
            self.get_logger().info('max_points reached')
            rclpy.shutdown()

    def format_point(self, x: float, y: float) -> str:
        return f'{x:.{self.precision}f},{y:.{self.precision}f}'

    def print_current_waypoints(self) -> None:
        param = ';'.join(self.format_point(x, y) for x, y in self.points)
        code_lines = [
            'DEFAULT_WAYPOINTS = [',
            *[
                f'    ({x:.{self.precision}f}, {y:.{self.precision}f}),'
                for x, y in self.points
            ],
            ']',
        ]

        print('')
        print('--- current waypoints parameter ---')
        print(f'waypoints:="{param}"')
        print('')
        print('--- current DEFAULT_WAYPOINTS block ---')
        print('\n'.join(code_lines))
        print('')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ClickedWaypointRecorderNode()
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
