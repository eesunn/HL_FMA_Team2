"""Path → MotionSegment 분해 노드.

Nav2 planner가 발행한 /plan (nav_msgs/Path) 을 받아
[ROTATE → STRAIGHT → ROTATE → STRAIGHT ...] segment 시퀀스로 분해한다.

알고리즘 (간단):
  1. plan.poses 의 (x,y) 점들을 가져온다.
  2. 시작점부터 순회하면서 다음 segment 후보 방향을 누적해 본다.
     인접 두 점 사이 방향이 straight_merge_threshold_deg 이내면 같은 직선으로 묶고
     그것보다 크면 segment 를 끊는다.
  3. 각 직선 구간 진입 전, 그 방향으로 ROTATE segment 를 먼저 넣는다 (heading
     차이가 heading_threshold_deg 보다 클 때만).
  4. 마지막에 plan 의 종점 yaw 로 맞추는 ROTATE segment 를 한 번 더 넣는다 (선택).

좌표계:
  /plan 은 보통 map 프레임이므로, segment 의 target_yaw 는 동일하게 map 프레임
  기준 절대 yaw 가 된다. segment_executor 는 /odometry/filtered (odom 프레임)
  를 보지만, 정상 상태에선 odom 과 map 의 yaw 차이가 거의 없으므로 (AMCL 이
  보정) 실용상 그대로 사용한다.

  완전 정확하려면 TF 로 map→odom 변환을 적용해야 하지만 현재 단계에선 생략.
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path

from capstone_msgs.msg import MotionSegment, MotionSegmentArray


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class PathSegmenterNode(Node):
    def __init__(self):
        super().__init__('path_segmenter_node')

        self.declare_parameter('heading_threshold_deg', 5.0)
        self.declare_parameter('straight_merge_threshold_deg', 3.0)
        self.declare_parameter('min_segment_distance_m', 0.05)

        self.heading_threshold = math.radians(
            float(self.get_parameter('heading_threshold_deg').value)
        )
        self.merge_threshold = math.radians(
            float(self.get_parameter('straight_merge_threshold_deg').value)
        )
        self.min_seg_dist = float(self.get_parameter('min_segment_distance_m').value)

        self.sub_plan = self.create_subscription(
            Path, '/plan', self.on_plan, 10
        )
        self.pub_segs = self.create_publisher(
            MotionSegmentArray, '/motion_segments', 10
        )

        self.last_seq_count = 0
        self.get_logger().info('path_segmenter_node ready')

    def on_plan(self, msg: Path) -> None:
        poses = msg.poses
        if len(poses) < 2:
            return

        pts = [(p.pose.position.x, p.pose.position.y) for p in poses]

        # 인접 점들의 yaw 후보 계산.
        leg_yaws = []
        for i in range(len(pts) - 1):
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
            if math.hypot(dx, dy) < 1e-4:
                leg_yaws.append(None)
            else:
                leg_yaws.append(math.atan2(dy, dx))

        # leg 들을 같은 방향끼리 묶어서 구간(run) 단위로 합친다.
        runs = []  # list of (yaw, total_distance)
        current_yaw = None
        current_start_idx = 0
        for i, y in enumerate(leg_yaws):
            if y is None:
                continue
            if current_yaw is None:
                current_yaw = y
                current_start_idx = i
                continue
            if abs(normalize_angle(y - current_yaw)) <= self.merge_threshold:
                # 같은 run.
                continue
            # 끊김. 이전 run 종료.
            dist = self._run_distance(pts, current_start_idx, i)
            runs.append((current_yaw, dist))
            current_yaw = y
            current_start_idx = i
        if current_yaw is not None:
            dist = self._run_distance(pts, current_start_idx, len(leg_yaws))
            runs.append((current_yaw, dist))

        # min_seg_dist 이하 run 은 제거 (잡음).
        runs = [(y, d) for (y, d) in runs if d >= self.min_seg_dist]
        if not runs:
            return

        # ROTATE → STRAIGHT 시퀀스 생성.
        # 시작 heading 은 plan 의 첫 pose orientation 을 쓴다.
        prev_yaw = self._pose_yaw(poses[0])
        segments = []
        for (target_yaw, dist) in runs:
            if abs(normalize_angle(target_yaw - prev_yaw)) > self.heading_threshold:
                rot = MotionSegment()
                rot.type = MotionSegment.TYPE_ROTATE
                rot.target_yaw = float(target_yaw)
                rot.distance = 0.0
                segments.append(rot)
            st = MotionSegment()
            st.type = MotionSegment.TYPE_STRAIGHT
            st.target_yaw = float(target_yaw)
            st.distance = float(dist)
            segments.append(st)
            prev_yaw = target_yaw

        # 마지막에 plan 종점 yaw 로 정렬.
        final_yaw = self._pose_yaw(poses[-1])
        if abs(normalize_angle(final_yaw - prev_yaw)) > self.heading_threshold:
            rot = MotionSegment()
            rot.type = MotionSegment.TYPE_ROTATE
            rot.target_yaw = float(final_yaw)
            rot.distance = 0.0
            segments.append(rot)

        out = MotionSegmentArray()
        out.header = msg.header
        out.segments = segments
        self.pub_segs.publish(out)

        self.last_seq_count += 1
        self.get_logger().info(
            f'plan #{self.last_seq_count}: poses={len(poses)} → segments={len(segments)} '
            f'(runs={len(runs)})'
        )

    @staticmethod
    def _run_distance(pts, start, end) -> float:
        d = 0.0
        for j in range(start, end):
            d += math.hypot(pts[j + 1][0] - pts[j][0],
                            pts[j + 1][1] - pts[j][1])
        return d

    @staticmethod
    def _pose_yaw(pose_stamped) -> float:
        q = pose_stamped.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = PathSegmenterNode()
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
