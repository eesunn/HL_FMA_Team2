"""goal_pose → planner 직접 호출 → /plan 발행.

bt_navigator / controller_server 를 쓰지 않고, planner_server 의 ComputePathToPose
액션을 직접 호출해 경로만 뽑아 /plan 으로 발행한다. 이후 path_segmenter 가 소비.

흐름:
  RViz "2D Goal Pose" → /goal_pose → (본 노드) → compute_path_to_pose 액션
                      → 결과 path → /plan → path_segmenter → segment_executor
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose


class GoalToPlanNode(Node):
    def __init__(self):
        super().__init__('goal_to_plan_node')

        self.declare_parameter('planner_id', 'GridBased')
        self.planner_id = str(self.get_parameter('planner_id').value)

        self.client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.sub = self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.pub = self.create_publisher(Path, '/plan', 10)

        self.get_logger().info('goal_to_plan_node ready (waiting for /goal_pose)')

    def on_goal(self, msg: PoseStamped) -> None:
        if not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('compute_path_to_pose action server not available')
            return
        goal = ComputePathToPose.Goal()
        goal.goal = msg
        goal.planner_id = self.planner_id
        goal.use_start = False   # 현재 로봇 위치(TF)를 시작점으로 사용
        self.get_logger().info(
            f'requesting plan to ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )
        self.client.send_goal_async(goal).add_done_callback(self._on_response)

    def _on_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('plan goal rejected by planner_server')
            return
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        result = future.result().result
        path = result.path
        n = len(path.poses)
        if n == 0:
            self.get_logger().warn('planner returned empty path (목표 도달 불가?)')
            return
        self.pub.publish(path)
        self.get_logger().info(f'published /plan with {n} poses')


def main(args=None):
    rclpy.init(args=args)
    node = GoalToPlanNode()
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
