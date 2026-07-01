"""신호등 인식 노드 (OAK-D 전용).

인식 방법:
  HSV 모드 : blob_path="" (기본)
  ML  모드  : YOLOv8n-detect blob
              출력 output0 shape = [4+nc, num_anchors] (nc=2)
              class 0 = green light, class 1 = red light

출력 토픽:
  /traffic_light       : "RED" / "GREEN" / "NONE"
  /traffic_light_debug : 디버그 이미지 (항상 발행)
  /cmd_vel_traffic     : RED 시에만 Twist() 발행 (현재 twist_mux 미연결)
"""

import os
from collections import deque

import cv2
import depthai as dai
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class TrafficLightDetectorNode(Node):
    IMG_W = 640
    IMG_H = 224

    # HSV 범위 (OpenCV: Hue 0-180)
    _RED_L1 = np.array([  0, 100,  80], dtype=np.uint8)
    _RED_H1 = np.array([ 10, 255, 255], dtype=np.uint8)
    _RED_L2 = np.array([160, 100,  80], dtype=np.uint8)
    _RED_H2 = np.array([180, 255, 255], dtype=np.uint8)
    _GRN_L  = np.array([ 38,  80,  80], dtype=np.uint8)
    _GRN_H  = np.array([ 90, 255, 255], dtype=np.uint8)

    RED   = 'RED'
    GREEN = 'GREEN'
    NONE  = 'NONE'

    _CLS_GREEN = 0
    _CLS_RED   = 1

    def __init__(self):
        super().__init__('traffic_light_detector_node')

        self.declare_parameter('blob_path',        '')
        self.declare_parameter('show_window',       False)
        self.declare_parameter('conf_thresh',       0.50)
        self.declare_parameter('roi_top_ratio',     0.0)
        self.declare_parameter('roi_bottom_ratio',  0.55)
        self.declare_parameter('min_color_area',    200)
        self.declare_parameter('vote_buffer_size',  10)
        self.declare_parameter('min_vote_samples',  3)
        self.declare_parameter('red_vote_ratio',    0.5)
        self.declare_parameter('green_vote_ratio',  0.5)

        blob_path             = str(self.get_parameter('blob_path').value)
        self.show_window      = bool(self.get_parameter('show_window').value)
        self.conf_thresh      = float(self.get_parameter('conf_thresh').value)
        self.roi_top_ratio    = float(self.get_parameter('roi_top_ratio').value)
        self.roi_bottom_ratio = float(self.get_parameter('roi_bottom_ratio').value)
        self.min_color_area   = int(self.get_parameter('min_color_area').value)
        self.vote_buffer_size = max(1, int(self.get_parameter('vote_buffer_size').value))
        self.min_vote_samples = max(1, int(self.get_parameter('min_vote_samples').value))
        self.red_vote_ratio   = float(self.get_parameter('red_vote_ratio').value)
        self.green_vote_ratio = float(self.get_parameter('green_vote_ratio').value)

        self._roi_top    = int(self.IMG_H * self.roi_top_ratio)
        self._roi_bottom = int(self.IMG_H * self.roi_bottom_ratio)

        # 상태 머신
        self._state        = self.NONE
        self._vote_history = deque(maxlen=self.vote_buffer_size)
        self._vote_counts  = {self.RED: 0, self.GREEN: 0, self.NONE: 0}

        # 디버그 캐시
        self._dbg_scores: tuple       = (0.0, 0.0)
        self._dbg_masks: tuple | None = None
        self._dbg_raw: str            = self.NONE

        self._use_ml = bool(blob_path and os.path.isfile(blob_path))

        # 퍼블리셔
        self.pub_state = self.create_publisher(String, '/traffic_light',       10)
        self.pub_debug = self.create_publisher(Image,  '/traffic_light_debug', 10)
        self.pub_cmd   = self.create_publisher(Twist,  '/cmd_vel_traffic',     10)
        self._bridge   = CvBridge()

        self._init_device(blob_path)
        self.create_timer(0.033, self._process_cb)

        self.get_logger().info(
            f'traffic_light_detector_node ready'
            f'  mode={"ML:" + os.path.basename(blob_path) if self._use_ml else "HSV"}'
            f'  roi_y={self._roi_top}-{self._roi_bottom}'
            f'  vote_buf={self.vote_buffer_size}'
            f'  min_samples={self.min_vote_samples}'
            f'  red_ratio>={self.red_vote_ratio:.2f}'
        )

    # ------------------------------------------------------------------
    # depthai 초기화
    # ------------------------------------------------------------------

    def _init_device(self, blob_path: str) -> None:
        self.pipeline = dai.Pipeline(dai.Device())

        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            (self.IMG_W, self.IMG_H),
            type=dai.ImgFrame.Type.BGR888p,
            fps=25,
        )

        if self._use_ml:
            nn = self.pipeline.create(dai.node.NeuralNetwork)
            nn.setBlobPath(blob_path)
            nn.setNumInferenceThreads(2)
            nn.input.setBlocking(False)
            cam_out.link(nn.input)
            self.nn_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)

        self.rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
        self.pipeline.start()

    # ------------------------------------------------------------------
    # 타이머 콜백
    # ------------------------------------------------------------------

    def _process_cb(self) -> None:
        frame_msg = self.rgb_queue.tryGet()
        if frame_msg is None:
            return
        bgr: np.ndarray = frame_msg.getCvFrame()

        raw = None
        if self._use_ml:
            nn_msg = self.nn_queue.tryGet()
            if nn_msg is not None:
                raw = self._detect_ml(nn_msg)
        if raw is None:
            raw = self._detect_hsv(bgr)

        state = self._update_state(raw)

        msg_state = String()
        msg_state.data = state
        self.pub_state.publish(msg_state)

        if state == self.RED:
            self.pub_cmd.publish(Twist())

        self._dbg_raw = raw
        self._publish_debug(bgr, state)

        if self.show_window:
            self._update_display(bgr, state)

    # ------------------------------------------------------------------
    # ML 감지
    # ------------------------------------------------------------------

    def _detect_ml(self, nn_msg) -> str:
        try:
            names = nn_msg.getAllLayerNames()
            tensor = np.array(nn_msg.getTensor(names[0]), dtype=np.float32)
            if not hasattr(self, '_ml_shape_logged'):
                self.get_logger().info(f'NN layers: {names}  shape: {tensor.shape}')
                self._ml_shape_logged = True
            raw = tensor.flatten()
        except Exception as e:
            self.get_logger().warn(f'ML 출력 파싱 실패: {e}', throttle_duration_sec=2.0)
            return self.NONE

        nc = 2
        num_anchors = raw.size // (4 + nc)
        if num_anchors == 0:
            return self.NONE

        preds      = raw.reshape(4 + nc, num_anchors)
        cls_logits = preds[4:, :]
        cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_logits, -20.0, 20.0)))
        max_green  = float(np.max(cls_scores[self._CLS_GREEN]))
        max_red    = float(np.max(cls_scores[self._CLS_RED]))
        self._dbg_scores = (max_red, max_green)

        if max_red >= self.conf_thresh:
            return self.RED
        if max_green >= self.conf_thresh:
            return self.GREEN
        return self.NONE

    # ------------------------------------------------------------------
    # HSV 감지
    # ------------------------------------------------------------------

    def _detect_hsv(self, bgr: np.ndarray) -> str:
        roi = bgr[self._roi_top:self._roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask = (
            cv2.inRange(hsv, self._RED_L1, self._RED_H1)
            | cv2.inRange(hsv, self._RED_L2, self._RED_H2)
        )
        grn_mask = cv2.inRange(hsv, self._GRN_L, self._GRN_H)

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        grn_mask = cv2.morphologyEx(grn_mask, cv2.MORPH_OPEN, kernel)

        r_area = self._max_contour_area(red_mask)
        g_area = self._max_contour_area(grn_mask)

        self._dbg_masks  = (red_mask, grn_mask)
        total = max(r_area + g_area, 1.0)
        self._dbg_scores = (r_area / total, g_area / total)

        if r_area >= self.min_color_area and r_area >= g_area:
            return self.RED
        if g_area >= self.min_color_area:
            return self.GREEN
        return self.NONE

    @staticmethod
    def _max_contour_area(mask: np.ndarray) -> float:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return float(max((cv2.contourArea(c) for c in cnts), default=0.0))

    # ------------------------------------------------------------------
    # Voting 상태 결정
    # ------------------------------------------------------------------

    def _update_state(self, detected: str | None) -> str:
        if detected is not None:
            self._vote_history.append(detected)

        self._vote_counts = {
            self.RED:   sum(1 for v in self._vote_history if v == self.RED),
            self.GREEN: sum(1 for v in self._vote_history if v == self.GREEN),
            self.NONE:  sum(1 for v in self._vote_history if v == self.NONE),
        }

        total = len(self._vote_history)
        if total < self.min_vote_samples:
            return self._state

        r, g = (self._vote_counts[k] / total for k in (self.RED, self.GREEN))

        if r >= self.red_vote_ratio:
            self._state = self.RED
        elif g >= self.green_vote_ratio:
            self._state = self.GREEN
        else:
            self._state = self.NONE

        return self._state

    # ------------------------------------------------------------------
    # 디버그 이미지 발행
    # ------------------------------------------------------------------

    def _publish_debug(self, bgr: np.ndarray, state: str) -> None:
        vis = bgr.copy()

        cv2.rectangle(vis, (0, self._roi_top), (self.IMG_W - 1, self._roi_bottom),
                      (180, 180, 180), 1)

        if not self._use_ml and self._dbg_masks is not None:
            red_m, grn_m = self._dbg_masks
            overlay = np.zeros_like(vis)
            s = slice(self._roi_top, self._roi_bottom)
            overlay[s][red_m > 0] = (0, 0, 200)
            overlay[s][grn_m > 0] = (0, 200, 0)
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        _COLOR = {self.RED: (0, 0, 255), self.GREEN: (0, 220, 0), self.NONE: (130, 130, 130)}
        sc = _COLOR.get(state, (255, 255, 255))

        cv2.rectangle(vis, (0, 0), (260, 30), (0, 0, 0), -1)
        cv2.putText(vis, f'{state}  raw:{self._dbg_raw}', (4, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

        red_s, grn_s = self._dbg_scores
        if self._use_ml:
            label = f'R:{red_s:.2f}  G:{grn_s:.2f}  thr:{self.conf_thresh:.2f}'
        else:
            label = f'R_area:{red_s*100:.0f}%  G_area:{grn_s*100:.0f}%'
        cv2.putText(vis, label, (4, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        vote_label = (
            f'R:{self._vote_counts[self.RED]} '
            f'G:{self._vote_counts[self.GREEN]} '
            f'N:{self._vote_counts[self.NONE]} '
            f'/{self.vote_buffer_size} '
            f'need>={self.red_vote_ratio:.0%}'
        )
        cv2.putText(vis, vote_label, (4, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(msg)

    # ------------------------------------------------------------------
    # 로컬 윈도우 표시
    # ------------------------------------------------------------------

    def _update_display(self, bgr: np.ndarray, state: str) -> None:
        _COLOR = {self.RED: (0, 0, 255), self.GREEN: (0, 220, 0), self.NONE: (130, 130, 130)}
        disp = bgr.copy()
        cv2.rectangle(disp, (0, self._roi_top), (self.IMG_W - 1, self._roi_bottom),
                      (180, 180, 180), 2)
        cv2.putText(disp, state, (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, _COLOR.get(state, (255, 255, 255)), 3)

        r = self._vote_counts[self.RED]
        g = self._vote_counts[self.GREEN]
        n = self._vote_counts[self.NONE]
        cv2.putText(disp,
                    f'R:{r} G:{g} N:{n} / {self.vote_buffer_size}  raw:{self._dbg_raw}',
                    (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        if self._dbg_masks is not None:
            red_m, grn_m = self._dbg_masks
            overlay = np.zeros_like(disp)
            s = slice(self._roi_top, self._roi_bottom)
            overlay[s][red_m > 0] = (0, 0, 180)
            overlay[s][grn_m > 0] = (0, 180, 0)
            disp = cv2.addWeighted(disp, 0.7, overlay, 0.3, 0)

        cv2.imshow('TrafficLight', disp)
        if cv2.waitKey(1) & 0xFF == 27:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
