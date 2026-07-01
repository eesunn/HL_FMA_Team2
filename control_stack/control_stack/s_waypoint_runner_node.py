"""Send an S-curve waypoint sequence to Nav2 continuous navigation.

This node is intended for lidar_navigation_continuous.launch.py. It sends a
NavigateThroughPoses action goal, so Nav2 plans through all waypoints and the
MPPI controller follows the path continuously.
"""

import math
from typing import List, Optional, Sequence, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateThroughPoses
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy


# Replace these sample map-frame coordinates with points measured in RViz.
# Each item can be (x, y) or (x, y, yaw_rad). If yaw is omitted, it is computed
# from the direction to the next waypoint.
DEFAULT_WAYPOINTS = [
    (0.50, 0.00),
    (1.00, 0.45),
    (1.50, -0.45),
    (2.00, 0.45),
    (2.50, 0.00),
]


WaypointInput = Tuple[float, float, Optional[float]]
Waypoint = Tuple[float, float, float]


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def direction_yaw(points: Sequence[WaypointInput], idx: int) -> float:
    x, y, _ = points[idx]
    if idx < len(points) - 1:
        nx, ny, _ = points[idx + 1]
        return math.atan2(ny - y, nx - x)
    if idx > 0:
        px, py, _ = points[idx - 1]
        return math.atan2(y - py, x - px)
    return 0.0


def normalize_waypoint(item: Sequence[float]) -> WaypointInput:
    if len(item) == 2:
        return float(item[0]), float(item[1]), None
    if len(item) == 3:
        return float(item[0]), float(item[1]), float(item[2])
    raise ValueError('waypoint must be (x, y) or (x, y, yaw_rad)')


def parse_waypoint_string(raw: str) -> List[WaypointInput]:
    """Parse 'x,y;x,y;y,yaw' style waypoint parameter strings."""
    waypoints: List[WaypointInput] = []
    for chunk in raw.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = [float(v.strip()) for v in chunk.split(',')]
        waypoints.append(normalize_waypoint(values))
    return waypoints


def resolve_yaws(points: Sequence[WaypointInput], yaw_mode: str) -> List[Waypoint]:
    if yaw_mode not in ('final_recorded', 'recorded', 'path_tangent'):
        raise ValueError(
            'yaw_mode must be one of final_recorded, recorded, path_tangent'
        )

    resolved = []
    last_idx = len(points) - 1
    for idx, (x, y, yaw) in enumerate(points):
        if yaw_mode == 'path_tangent':
            out_yaw = direction_yaw(points, idx)
        elif yaw_mode == 'recorded':
            out_yaw = direction_yaw(points, idx) if yaw is None else yaw
        else:
            use_recorded_final = idx == last_idx and yaw is not None
            out_yaw = yaw if use_recorded_final else direction_yaw(points, idx)
        resolved.append((x, y, out_yaw))
    return resolved


def status_name(status: int) -> str:
    names = {
        GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
        GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
        GoalStatus.STATUS_EXECUTING: 'EXECUTING',
        GoalStatus.STATUS_CANCELING: 'CANCELING',
        GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
        GoalStatus.STATUS_CANCELED: 'CANCELED',
        GoalStatus.STATUS_ABORTED: 'ABORTED',
    }
    return names.get(status, f'UNRECOGNIZED({status})')


class SWaypointRunnerNode(Node):
    def __init__(self):
        super().__init__('s_waypoint_runner')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('action_name', 'navigate_through_poses')
        self.declare_parameter('waypoints', '')
        self.declare_parameter('yaw_mode', 'final_recorded')
        self.declare_parameter('start_delay_sec', 2.0)
        self.declare_parameter('publish_final_goal_pose', False)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.action_name = str(self.get_parameter('action_name').value)
        self.yaw_mode = str(self.get_parameter('yaw_mode').value)
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.publish_final_goal_pose = bool(
            self.get_parameter('publish_final_goal_pose').value
        )

        raw_waypoints = str(self.get_parameter('waypoints').value).strip()
        if raw_waypoints:
            waypoint_inputs = parse_waypoint_string(raw_waypoints)
        else:
            waypoint_inputs = [normalize_waypoint(p) for p in DEFAULT_WAYPOINTS]

        if len(waypoint_inputs) < 2:
            raise ValueError('at least two waypoints are required')

        self.waypoints = resolve_yaws(waypoint_inputs, self.yaw_mode)
        self.client = ActionClient(
            self,
            NavigateThroughPoses,
            self.action_name,
        )
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # 목표 웨이포인트를 latched 로 발행 → navigation_logger 가 waypoints.csv 로 기록.
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', latched)
        self._publish_waypoint_array()

        self.start_timer = self.create_timer(
            max(self.start_delay_sec, 0.1),
            self._start_once,
        )

        self.get_logger().info(
            f'S waypoint runner ready: {len(self.waypoints)} waypoints, '
            f'frame={self.frame_id}, action={self.action_name}, '
            f'yaw_mode={self.yaw_mode}'
        )
        for idx, (x, y, yaw) in enumerate(self.waypoints, start=1):
            self.get_logger().info(
                f'  P{idx}: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f} deg'
            )

    def _publish_waypoint_array(self) -> None:
        arr = PoseArray()
        arr.header.frame_id = self.frame_id
        arr.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in self.waypoints:
            p = Pose()
            p.position.x = x
            p.position.y = y
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            p.orientation.x = qx
            p.orientation.y = qy
            p.orientation.z = qz
            p.orientation.w = qw
            arr.poses.append(p)
        self.waypoints_pub.publish(arr)
        self.get_logger().info(
            f'published {len(arr.poses)} waypoints on /waypoints (latched)'
        )

    def make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _start_once(self) -> None:
        self.start_timer.cancel()

        self.get_logger().info('waiting for NavigateThroughPoses action server...')
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'action server not available: {self.action_name}'
            )
            rclpy.shutdown()
            return

        poses = [self.make_pose(x, y, yaw) for x, y, yaw in self.waypoints]

        if self.publish_final_goal_pose:
            self.goal_pose_pub.publish(poses[-1])
            self.get_logger().warn(
                'published final waypoint to /goal_pose. In continuous mode this '
                'can also trigger NavigateToPose, so keep this parameter false '
                'unless you specifically need it.'
            )

        goal = NavigateThroughPoses.Goal()
        goal.poses = poses

        self.get_logger().info(f'sending {len(poses)} waypoints')
        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.on_feedback,
        )
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('S waypoint goal rejected')
            rclpy.shutdown()
            return

        self.get_logger().info('S waypoint goal accepted')
        goal_handle.get_result_async().add_done_callback(self.on_result)

    def on_feedback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'remaining poses={feedback.number_of_poses_remaining}, '
            f'distance={feedback.distance_remaining:.2f} m',
            throttle_duration_sec=2.0,
        )

    def on_result(self, future) -> None:
        result = future.result()
        name = status_name(result.status)
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'S waypoint navigation finished: {name}')
        else:
            self.get_logger().error(
                f'S waypoint navigation did not succeed: {name} ({result.status})'
            )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SWaypointRunnerNode()
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
