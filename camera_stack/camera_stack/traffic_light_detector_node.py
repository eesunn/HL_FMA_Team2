"""신호등 인식 노드 (PC 추론, ML 전용).

/camera/image_raw 구독 → ultralytics YOLO .pt 추론 → voting 상태 결정.

출력 토픽:
  /traffic_light          : "RED" / "GREEN" / "NONE"
  /traffic_light_box_area : EMA 평활화된 RED 박스 넓이 (px²) — 근접도 판단용
  /traffic_light_debug    : 디버그 이미지

박스 크기 기반 정지 원리:
  차량이 신호등에 접근할수록 YOLO 바운딩박스 면적이 증가.
  EMA(alpha=box_area_ema_alpha)로 노이즈를 평활화하고,
  RED 박스가 없는 프레임에서는 box_area_decay로 서서히 감소.
  lane_recovery_node에서 'RED AND box_area >= threshold'일 때 정지.

CSV 로그 (~/capstone_ws/logs/traffic_light_YYYYMMDD_HHMMSS.csv):
  ts_sec            : 수신 타임스탬프 (초)
  raw_detect        : 프레임 단위 YOLO 감지 결과 (RED/GREEN/NONE)
  state             : voting 후 최종 상태
  state_changed     : 이전 프레임 대비 상태 변경 여부 (0/1)
  max_red_conf      : 프레임 내 RED 클래스 최대 신뢰도
  max_green_conf    : 프레임 내 GREEN 클래스 최대 신뢰도
  conf_thresh       : 적용된 신뢰도 임계값
  n_boxes           : 감지된 총 박스 수
  n_red_boxes       : RED 박스 수
  n_green_boxes     : GREEN 박스 수
  vote_r            : 버퍼 내 RED 투표 수
  vote_g            : 버퍼 내 GREEN 투표 수
  vote_n            : 버퍼 내 NONE 투표 수
  vote_total        : 버퍼 내 총 투표 수
  vote_buf_size     : 버퍼 최대 크기
  red_vote_ratio    : RED 판정 기준 비율
  green_vote_ratio  : GREEN 판정 기준 비율
  box_area_ema_px2  : EMA 평활화된 최대 RED 박스 면적 (px²)
"""

import csv
import datetime
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from ultralytics import YOLO

_CSV_HEADER = [
    'ts_sec',
    'frame_no',
    'fps',
    'image_age_ms',
    'callback_ms',
    'infer_ms',
    'raw_detect',
    'state',
    'state_changed',
    'max_red_conf',
    'max_green_conf',
    'conf_thresh',
    'n_boxes',
    'n_red_boxes',
    'n_green_boxes',
    'max_red_area_px2',
    'max_green_area_px2',
    'max_box_area_px2',
    'vote_r',
    'vote_g',
    'vote_n',
    'vote_total',
    'vote_buf_size',
    'red_vote_ratio',
    'green_vote_ratio',
    'box_area_ema_px2',
]


class TrafficLightDetectorNode(Node):
    RED   = 'RED'
    GREEN = 'GREEN'
    NONE  = 'NONE'

    # class index in the .pt model (confirm with model.names)
    _CLS_GREEN = 0
    _CLS_RED   = 1

    def __init__(self):
        super().__init__('traffic_light_detector_node')

        _default_pt = os.path.join(
            get_package_share_directory('camera_stack'),
            'models', 'traffic_light_best.pt',
        )

        self.declare_parameter('model_path',           _default_pt)
        self.declare_parameter('show_window',           False)
        self.declare_parameter('conf_thresh',           0.50)
        self.declare_parameter('vote_buffer_size',      10)
        self.declare_parameter('min_vote_samples',      3)
        self.declare_parameter('red_vote_ratio',        0.5)
        self.declare_parameter('green_vote_ratio',      0.5)
        self.declare_parameter('input_topic',           '/camera/image_raw')
        self.declare_parameter('log_csv',               True)
        self.declare_parameter('capture_dir',           os.path.expanduser('~/capstone_ws/logs'))
        # 박스 크기 기반 정지용 EMA 파라미터
        self.declare_parameter('box_area_ema_alpha',    0.3)   # EMA 평활 계수 (0~1, 클수록 빠름)
        self.declare_parameter('box_area_decay',        0.9)   # RED 박스 없을 때 프레임당 감쇠율

        model_path                = str(self.get_parameter('model_path').value)
        self.show_window          = bool(self.get_parameter('show_window').value)
        self.conf_thresh          = float(self.get_parameter('conf_thresh').value)
        self.vote_buffer_size     = max(1, int(self.get_parameter('vote_buffer_size').value))
        self.min_vote_samples     = max(1, int(self.get_parameter('min_vote_samples').value))
        self.red_vote_ratio       = float(self.get_parameter('red_vote_ratio').value)
        self.green_vote_ratio     = float(self.get_parameter('green_vote_ratio').value)
        input_topic               = str(self.get_parameter('input_topic').value)
        log_csv                   = bool(self.get_parameter('log_csv').value)
        capture_dir               = str(self.get_parameter('capture_dir').value)
        self._box_area_ema_alpha  = float(self.get_parameter('box_area_ema_alpha').value)
        self._box_area_decay      = float(self.get_parameter('box_area_decay').value)

        if not os.path.isfile(model_path):
            raise RuntimeError(
                f'model_path 파일이 없습니다: "{model_path}"\n'
                f'  --ros-args -p model_path:=/path/to/best.pt 로 지정하세요.'
            )

        self._model = YOLO(model_path)
        self.get_logger().info(
            f'YOLO 모델 로드 완료: {os.path.basename(model_path)}'
            f'  classes={self._model.names}'
        )

        # 상태 머신 + voting
        self._state      = self.NONE
        self._prev_state = self.NONE
        self._vote_history = deque(maxlen=self.vote_buffer_size)
        self._vote_counts  = {self.RED: 0, self.GREEN: 0, self.NONE: 0}

        # EMA 박스 면적 (박스 크기 기반 정지용)
        self._ema_box_area: float = 0.0

        # 디버그 캐시
        self._dbg_scores: tuple = (0.0, 0.0)   # (red_conf, green_conf)
        self._dbg_raw: str      = self.NONE
        self._dbg_boxes: list   = []            # [(x1,y1,x2,y2,cls,conf), ...]
        self._frame_no: int     = 0
        self._last_ts_sec: float | None = None
        self._last_fps: float   = 0.0
        self._last_infer_ms: float = 0.0

        self._bridge = CvBridge()

        # CSV 로거
        self._csv_f      = None
        self._csv_writer = None
        if log_csv:
            os.makedirs(capture_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(capture_dir, f'traffic_light_{ts}.csv')
            self._csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_f)
            self._csv_writer.writerow(_CSV_HEADER)
            self.get_logger().info(f'[TLLog] {csv_path}')

        # 퍼블리셔
        self.pub_state    = self.create_publisher(String,  '/traffic_light',          10)
        self.pub_box_area = self.create_publisher(Float32, '/traffic_light_box_area', 10)
        self.pub_debug    = self.create_publisher(Image,   '/traffic_light_debug',    10)

        # 구독
        self.create_subscription(Image, input_topic, self._on_image, 10)

        self.get_logger().info(
            f'traffic_light_detector_node ready'
            f'  input={input_topic}'
            f'  conf>={self.conf_thresh:.2f}'
            f'  vote_buf={self.vote_buffer_size}'
            f'  min_samples={self.min_vote_samples}'
            f'  red_ratio>={self.red_vote_ratio:.2f}'
            f'  log_csv={log_csv}'
        )

    # ------------------------------------------------------------------
    # 이미지 콜백
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        cb_t0 = time.perf_counter()
        ts_sec = self.get_clock().now().nanoseconds * 1e-9
        msg_stamp_sec = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        image_age_ms = (
            (ts_sec - msg_stamp_sec) * 1000.0
            if msg_stamp_sec > 0.0 else None
        )
        if self._last_ts_sec is not None:
            dt = ts_sec - self._last_ts_sec
            if dt > 0.0:
                self._last_fps = 1.0 / dt
        self._last_ts_sec = ts_sec

        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        raw, n_boxes, n_red, n_green, max_red_area, max_green_area = self._detect(bgr)
        self._prev_state = self._state
        state = self._update_state(raw)
        state_changed = int(state != self._prev_state)

        # EMA 박스 면적 갱신
        if n_red > 0 and max_red_area > 0.0:
            if self._ema_box_area == 0.0:
                self._ema_box_area = max_red_area
            else:
                self._ema_box_area = (
                    self._box_area_ema_alpha * max_red_area
                    + (1.0 - self._box_area_ema_alpha) * self._ema_box_area
                )
        else:
            self._ema_box_area *= self._box_area_decay

        # /traffic_light_box_area 발행
        area_msg = Float32()
        area_msg.data = float(self._ema_box_area)
        self.pub_box_area.publish(area_msg)

        out_msg = String()
        out_msg.data = state
        self.pub_state.publish(out_msg)

        self._dbg_raw = raw
        self._publish_debug(bgr, state)

        if self.show_window:
            self._update_display(bgr, state)

        callback_ms = (time.perf_counter() - cb_t0) * 1000.0
        self._write_csv(
            ts_sec, image_age_ms, callback_ms,
            raw, state, state_changed,
            n_boxes, n_red, n_green,
            max_red_area, max_green_area,
        )
        self._frame_no += 1

    # ------------------------------------------------------------------
    # YOLO 추론
    # ------------------------------------------------------------------

    def _detect(self, bgr: np.ndarray) -> tuple:
        """YOLO 추론 → 상태/박스 수/최대 색상별 박스 면적."""
        infer_t0 = time.perf_counter()
        results = self._model(bgr, conf=self.conf_thresh, verbose=False)
        self._last_infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        max_red      = 0.0
        max_green    = 0.0
        n_red        = 0
        n_green      = 0
        max_red_area = 0.0
        max_green_area = 0.0
        boxes_out    = []

        for box in results[0].boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            boxes_out.append((x1, y1, x2, y2, cls, conf))
            if cls == self._CLS_RED:
                max_red = max(max_red, conf)
                n_red  += 1
                area = float((x2 - x1) * (y2 - y1))
                if area > max_red_area:
                    max_red_area = area
            elif cls == self._CLS_GREEN:
                max_green = max(max_green, conf)
                n_green  += 1
                area = float((x2 - x1) * (y2 - y1))
                if area > max_green_area:
                    max_green_area = area

        self._dbg_scores = (max_red, max_green)
        self._dbg_boxes  = boxes_out

        if max_red >= self.conf_thresh:
            raw = self.RED
        elif max_green >= self.conf_thresh:
            raw = self.GREEN
        else:
            raw = self.NONE

        return raw, len(boxes_out), n_red, n_green, max_red_area, max_green_area

    # ------------------------------------------------------------------
    # Voting 상태 결정
    # ------------------------------------------------------------------

    def _update_state(self, detected: str) -> str:
        self._vote_history.append(detected)
        self._vote_counts = {
            self.RED:   sum(1 for v in self._vote_history if v == self.RED),
            self.GREEN: sum(1 for v in self._vote_history if v == self.GREEN),
            self.NONE:  sum(1 for v in self._vote_history if v == self.NONE),
        }

        total = len(self._vote_history)
        if total < self.min_vote_samples:
            return self._state

        r = self._vote_counts[self.RED]   / total
        g = self._vote_counts[self.GREEN] / total

        if r >= self.red_vote_ratio:
            self._state = self.RED
        elif g >= self.green_vote_ratio:
            self._state = self.GREEN
        else:
            self._state = self.NONE

        return self._state

    # ------------------------------------------------------------------
    # CSV 로깅
    # ------------------------------------------------------------------

    def _write_csv(
        self,
        ts_sec: float,
        image_age_ms: float | None,
        callback_ms: float,
        raw: str,
        state: str,
        state_changed: int,
        n_boxes: int,
        n_red: int,
        n_green: int,
        max_red_area: float,
        max_green_area: float,
    ) -> None:
        if self._csv_writer is None:
            return
        red_s, grn_s = self._dbg_scores
        total = len(self._vote_history)
        self._csv_writer.writerow([
            round(ts_sec, 4),
            self._frame_no,
            round(self._last_fps, 2),
            round(image_age_ms, 2) if image_age_ms is not None else '',
            round(callback_ms, 2),
            round(self._last_infer_ms, 2),
            raw,
            state,
            state_changed,
            round(red_s, 4),
            round(grn_s, 4),
            self.conf_thresh,
            n_boxes,
            n_red,
            n_green,
            round(max_red_area, 1),
            round(max_green_area, 1),
            round(max(max_red_area, max_green_area), 1),
            self._vote_counts[self.RED],
            self._vote_counts[self.GREEN],
            self._vote_counts[self.NONE],
            total,
            self.vote_buffer_size,
            self.red_vote_ratio,
            self.green_vote_ratio,
            round(self._ema_box_area, 1),
        ])
        self._csv_f.flush()

    # ------------------------------------------------------------------
    # 디버그 이미지 발행
    # ------------------------------------------------------------------

    def _publish_debug(self, bgr: np.ndarray, state: str) -> None:
        vis = self._draw_debug(bgr, state)
        msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(msg)

    def _draw_debug(self, bgr: np.ndarray, state: str) -> np.ndarray:
        vis = bgr.copy()
        _COLOR = {self.RED: (0, 0, 255), self.GREEN: (0, 220, 0), self.NONE: (130, 130, 130)}
        sc = _COLOR.get(state, (255, 255, 255))

        # 박스 그리기
        for x1, y1, x2, y2, cls, conf in self._dbg_boxes:
            color = _COLOR[self.RED] if cls == self._CLS_RED else _COLOR[self.GREEN]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f'{"RED" if cls == self._CLS_RED else "GREEN"} {conf:.2f}'
            cv2.putText(vis, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 상태 HUD
        cv2.rectangle(vis, (0, 0), (320, 30), (0, 0, 0), -1)
        cv2.putText(vis, f'{state}  raw:{self._dbg_raw}', (4, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

        red_s, grn_s = self._dbg_scores
        cv2.putText(vis, f'R:{red_s:.2f}  G:{grn_s:.2f}  thr:{self.conf_thresh:.2f}',
                    (4, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        vote_label = (
            f'R:{self._vote_counts[self.RED]} '
            f'G:{self._vote_counts[self.GREEN]} '
            f'N:{self._vote_counts[self.NONE]} '
            f'/{self.vote_buffer_size}'
        )
        cv2.putText(vis, vote_label, (4, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        area_color = (0, 0, 255) if self._ema_box_area > 0.0 else (130, 130, 130)
        cv2.putText(vis, f'boxEMA:{self._ema_box_area:.0f}px2', (4, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, area_color, 1)

        return vis

    # ------------------------------------------------------------------
    # 로컬 윈도우 표시
    # ------------------------------------------------------------------

    def _update_display(self, bgr: np.ndarray, state: str) -> None:
        vis = self._draw_debug(bgr, state)
        cv2.imshow('TrafficLight', vis)
        if cv2.waitKey(1) & 0xFF == 27:
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        if self._csv_f is not None:
            self._csv_f.flush()
            self._csv_f.close()
            self._csv_f = None
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
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
