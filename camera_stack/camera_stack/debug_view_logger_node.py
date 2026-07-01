"""Debug image logger for camera mission evidence.

Subscribes to visual debug image topics such as /lane_debug and
/traffic_light_debug, then records each stream as an mp4 file.  Optional PNG
snapshots can also be saved for slides or reports.
"""

import csv
import datetime
import os
from dataclasses import dataclass

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def _split_param(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def _safe_name(topic: str) -> str:
    name = topic.strip('/').replace('/', '_')
    return name if name else 'image'


@dataclass
class StreamState:
    label: str
    topic: str
    writer: cv2.VideoWriter | None = None
    frame_no: int = 0
    last_snapshot_ts: float = -1.0
    video_path: str = ''


class DebugViewLoggerNode(Node):
    def __init__(self):
        super().__init__('debug_view_logger_node')

        self.declare_parameter('topics', '/lane_debug,/traffic_light_debug')
        self.declare_parameter('labels', 'lane,traffic_light')
        self.declare_parameter('save_dir', os.path.expanduser('~/capstone_ws/logs'))
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('show_window', False)
        self.declare_parameter('write_video', True)
        self.declare_parameter('write_frames', False)
        self.declare_parameter('snapshot_every_sec', 1.0)

        topics = _split_param(str(self.get_parameter('topics').value))
        labels = _split_param(str(self.get_parameter('labels').value))
        if not topics:
            raise RuntimeError('topics 파라미터가 비어 있습니다.')
        if len(labels) < len(topics):
            labels += [_safe_name(topic) for topic in topics[len(labels):]]

        self.save_root = str(self.get_parameter('save_dir').value)
        self.fps = float(self.get_parameter('fps').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.write_video = bool(self.get_parameter('write_video').value)
        self.write_frames = bool(self.get_parameter('write_frames').value)
        self.snapshot_every_sec = float(
            self.get_parameter('snapshot_every_sec').value
        )

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_root, f'debug_view_{ts}')
        os.makedirs(self.session_dir, exist_ok=True)
        self.frames_dir = os.path.join(self.session_dir, 'frames')
        if self.write_frames:
            os.makedirs(self.frames_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._streams: dict[str, StreamState] = {}
        self._csv_path = os.path.join(self.session_dir, 'index.csv')
        self._csv_f = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow([
            'topic',
            'label',
            'frame_no',
            'ros_ts_sec',
            'image_stamp_sec',
            'image_age_ms',
            'width',
            'height',
            'video_path',
            'snapshot_path',
        ])
        self._csv_f.flush()

        for topic, label in zip(topics, labels):
            state = StreamState(label=label, topic=topic)
            self._streams[topic] = state
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self._on_image(topic, msg),
                10,
            )

        self.get_logger().info(
            f'debug_view_logger_node ready  dir={self.session_dir}  '
            f'topics={topics}  video={self.write_video}  frames={self.write_frames}'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp_sec(self, msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _ensure_writer(self, state: StreamState, width: int, height: int) -> None:
        if not self.write_video or state.writer is not None:
            return
        video_path = os.path.join(self.session_dir, f'{state.label}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'VideoWriter open 실패: {video_path}')
        state.writer = writer
        state.video_path = video_path
        self.get_logger().info(f'[{state.label}] video logging: {video_path}')

    def _snapshot_path(
        self,
        state: StreamState,
        frame,
        ros_ts_sec: float,
    ) -> str:
        if not self.write_frames:
            return ''
        if (
            state.last_snapshot_ts >= 0.0
            and ros_ts_sec - state.last_snapshot_ts < self.snapshot_every_sec
        ):
            return ''
        label_dir = os.path.join(self.frames_dir, state.label)
        os.makedirs(label_dir, exist_ok=True)
        path = os.path.join(label_dir, f'{state.label}_{state.frame_no:06d}.png')
        cv2.imwrite(path, frame)
        state.last_snapshot_ts = ros_ts_sec
        return path

    def _on_image(self, topic: str, msg: Image) -> None:
        state = self._streams[topic]
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        ros_ts_sec = self._now_sec()
        image_stamp_sec = self._stamp_sec(msg)
        age_ms = (
            (ros_ts_sec - image_stamp_sec) * 1000.0
            if image_stamp_sec > 0.0 else ''
        )

        self._ensure_writer(state, width, height)
        if state.writer is not None:
            state.writer.write(frame)

        snapshot = self._snapshot_path(state, frame, ros_ts_sec)

        self._csv.writerow([
            topic,
            state.label,
            state.frame_no,
            round(ros_ts_sec, 4),
            round(image_stamp_sec, 4) if image_stamp_sec > 0.0 else '',
            round(age_ms, 2) if age_ms != '' else '',
            width,
            height,
            state.video_path,
            snapshot,
        ])
        if state.frame_no % 30 == 0:
            self._csv_f.flush()

        if self.show_window:
            cv2.imshow(state.label, frame)
            cv2.waitKey(1)

        state.frame_no += 1

    def destroy_node(self) -> None:
        for state in self._streams.values():
            if state.writer is not None:
                state.writer.release()
                state.writer = None
        if self._csv_f is not None:
            self._csv_f.flush()
            self._csv_f.close()
            self._csv_f = None
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DebugViewLoggerNode()
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
