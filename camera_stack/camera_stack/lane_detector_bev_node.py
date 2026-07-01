"""차선 인식 노드 — Bird's Eye View 원근 보정 버전.

OAK-D 온디바이스 ImageManip으로 저고도(17~19 cm) 카메라 영상을
탑뷰로 변환한 뒤 YOLOv8-seg 추론 → /lane_offset, /lane_offset_m, /lane_valid 발행.

4점 warp 파라미터 (픽셀 좌표, 640×224 기준):
  warp_tl_x/y : 카메라 영상에서 도로 사다리꼴 좌상단
  warp_tr_x/y : 우상단
  warp_br_x/y : 우하단
  warp_bl_x/y : 좌하단
→ 이 사다리꼴이 640×224 전체로 펼쳐집니다.

show_window=true 로 실행하면 두 창이 표시됩니다:
  [원본] : 카메라 원본 + 사다리꼴 영역 표시
  [BEV]  : 변환된 탑뷰 + NN 검출 결과 오버레이
"""

import os

import cv2
import depthai as dai
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32

from camera_stack.diag_logger import DiagLogger


class LaneDetectorBEVNode(Node):
    IMG_W = 640
    IMG_H = 224

    def __init__(self):
        super().__init__('lane_detector_bev_node')

        # ── 기존 파라미터 ──────────────────────────────────────────────
        self.declare_parameter('blob_path', '')
        self.declare_parameter('conf_thresh', 0.35)
        self.declare_parameter('mask_thresh', 0.45)
        self.declare_parameter('roi_start_ratio', 0.55)
        self.declare_parameter('noise_cut_ratio', 0.20)
        self.declare_parameter('road_half_w', 80)
        self.declare_parameter('gap_thresh', 3)
        self.declare_parameter('show_window', False)

        # ── 신규 파라미터 ──────────────────────────────────────────────
        self.declare_parameter('xm_per_pix', 0.0)          # m/px (가로 방향)
        self.declare_parameter('lane_width_m', 0.0)        # 차선 전체 폭 m (0이면 road_half_w 유지)
        self.declare_parameter('log_diag', False)           # True=CSV 진단 기록
        self.declare_parameter('capture_dir', '/tmp/lane_diag')

        # ── BEV warp 4점 (픽셀 좌표) ────────────────────────────────────
        #
        #  원본 영상 640×224
        #  ┌─────────────────────────────────┐
        #  │       TL───────TR               │
        #  │      /             \            │
        #  │     /               \           │
        #  │   BL─────────────────BR         │
        #  └─────────────────────────────────┘
        #
        self.declare_parameter('warp_tl_x', 220.0)   # 좌상단 X
        self.declare_parameter('warp_tl_y',  80.0)   # 좌상단 Y
        self.declare_parameter('warp_tr_x', 420.0)   # 우상단 X
        self.declare_parameter('warp_tr_y',  80.0)   # 우상단 Y
        self.declare_parameter('warp_br_x', 620.0)   # 우하단 X
        self.declare_parameter('warp_br_y', 210.0)   # 우하단 Y
        self.declare_parameter('warp_bl_x',  20.0)   # 좌하단 X
        self.declare_parameter('warp_bl_y', 210.0)   # 좌하단 Y

        blob_path = str(self.get_parameter('blob_path').value)
        self.conf_thresh      = float(self.get_parameter('conf_thresh').value)
        self.mask_thresh      = float(self.get_parameter('mask_thresh').value)
        self.roi_start_ratio  = float(self.get_parameter('roi_start_ratio').value)
        self.noise_cut_ratio  = float(self.get_parameter('noise_cut_ratio').value)
        self.road_half_w      = int(self.get_parameter('road_half_w').value)
        self.gap_thresh       = int(self.get_parameter('gap_thresh').value)
        self.show_window      = bool(self.get_parameter('show_window').value)
        self.xm_per_pix       = float(self.get_parameter('xm_per_pix').value)
        self.lane_width_m     = float(self.get_parameter('lane_width_m').value)

        # lane_width_m가 지정된 경우 pixels 단위 half-width 재계산
        if self.lane_width_m > 0 and self.xm_per_pix > 0:
            self.road_half_w = int(self.lane_width_m / 2.0 / self.xm_per_pix)

        self.warp_src = np.float32([
            [self.get_parameter('warp_tl_x').value, self.get_parameter('warp_tl_y').value],
            [self.get_parameter('warp_tr_x').value, self.get_parameter('warp_tr_y').value],
            [self.get_parameter('warp_br_x').value, self.get_parameter('warp_br_y').value],
            [self.get_parameter('warp_bl_x').value, self.get_parameter('warp_bl_y').value],
        ])
        self.warp_dst = np.float32([
            [0,           0          ],
            [self.IMG_W,  0          ],
            [self.IMG_W,  self.IMG_H ],
            [0,           self.IMG_H ],
        ])
        # PC 시각화용 OpenCV Homography 행렬
        self.M_display = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)

        # ROI row_start (DiagLogger용 캐시)
        _rs = int(self.IMG_H * self.roi_start_ratio)
        _nc = int((self.IMG_H - _rs) * self.noise_cut_ratio)
        self._row_start = _rs + _nc

        if not blob_path or not os.path.isfile(blob_path):
            raise FileNotFoundError(
                f'blob_path 파라미터를 설정하세요. 현재값: {blob_path!r}'
            )

        # ── 퍼블리셔 ──────────────────────────────────────────────────
        _latching_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub          = self.create_publisher(Float32, '/lane_offset',   10)
        self.pub_offset_m = self.create_publisher(Float32, '/lane_offset_m', 10)
        self.pub_valid    = self.create_publisher(Bool,    '/lane_valid',    _latching_qos)

        # ── DiagLogger ─────────────────────────────────────────────────
        log_diag    = bool(self.get_parameter('log_diag').value)
        capture_dir = str(self.get_parameter('capture_dir').value)
        self._diag  = DiagLogger(enabled=log_diag, save_dir=capture_dir,
                                 xm_per_pix=self.xm_per_pix)

        # DiagLogger에 넘길 프레임별 상태 (side-effect from _lane_center_x)
        self._vis_lc_pts      = 0
        self._vis_rc_pts      = 0
        self._vis_center_mode = 'NONE'

        self._init_device(blob_path)
        self.create_timer(0.033, self._process_cb)

        self.get_logger().info(
            f'lane_detector_bev_node ready  blob={os.path.basename(blob_path)}'
            f'  TL=({self.warp_src[0,0]:.0f},{self.warp_src[0,1]:.0f})'
            f'  TR=({self.warp_src[1,0]:.0f},{self.warp_src[1,1]:.0f})'
            f'  BR=({self.warp_src[2,0]:.0f},{self.warp_src[2,1]:.0f})'
            f'  BL=({self.warp_src[3,0]:.0f},{self.warp_src[3,1]:.0f})'
            f'  xm_per_pix={self.xm_per_pix:.6f}'
            f'  road_half_w={self.road_half_w}px'
        )

    # ------------------------------------------------------------------
    # depthai 파이프라인 초기화
    # ------------------------------------------------------------------

    def _init_device(self, blob_path: str) -> None:
        self.pipeline = dai.Pipeline(dai.Device())

        # 카메라
        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            (self.IMG_W, self.IMG_H),
            type=dai.ImgFrame.Type.BGR888p,
            fps=25,
        )

        # ── BEV 원근 변환 (온디바이스) ────────────────────────────────
        manip = self.pipeline.create(dai.node.ImageManip)
        src_pts = [dai.Point2f(float(x), float(y)) for x, y in self.warp_src]
        dst_pts = [dai.Point2f(float(x), float(y)) for x, y in self.warp_dst]
        manip.initialConfig.addTransformFourPoints(src_pts, dst_pts, False)
        manip.initialConfig.setOutputSize(self.IMG_W, self.IMG_H)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        cam_out.link(manip.inputImage)
        # ─────────────────────────────────────────────────────────────

        # NN — 변환된 BEV 이미지로 추론
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(blob_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        manip.out.link(nn.input)

        self.nn_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
        if self.show_window:
            # 원본 프레임 (사다리꼴 표시용)
            self.rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()

    # ------------------------------------------------------------------
    # 메인 처리 콜백
    # ------------------------------------------------------------------

    def _process_cb(self) -> None:
        in_nn = self.nn_queue.tryGet()
        if in_nn is None:
            if self.show_window:
                self._update_display(None, None)
            return

        binary = self._parse_yolov8seg(in_nn)
        if binary is not None:
            center_x = self._lane_center_x(binary)
        else:
            center_x = None
            self._vis_lc_pts      = 0
            self._vis_rc_pts      = 0
            self._vis_center_mode = 'NONE'

        offset = self._normalize_offset(center_x)

        # /lane_offset (기존, 유지)
        if offset is not None:
            msg = Float32()
            msg.data = offset
            self.pub.publish(msg)

        # /lane_offset_m (신규)
        lane_offset_m = None
        if center_x is not None and self.xm_per_pix > 0:
            lane_offset_m = (center_x - self.IMG_W / 2.0) * self.xm_per_pix
            msg_m = Float32()
            msg_m.data = float(lane_offset_m)
            self.pub_offset_m.publish(msg_m)

        # /lane_valid (신규)
        lane_valid = center_x is not None
        msg_valid = Bool()
        msg_valid.data = lane_valid
        self.pub_valid.publish(msg_valid)

        # DiagLogger
        now = self.get_clock().now().nanoseconds * 1e-9
        self._diag.write_row(
            ts_sec=now,
            center_mode=self._vis_center_mode,
            lc_status='ACCEPT' if self._vis_lc_pts > 0 else 'MISS',
            rc_status='ACCEPT' if self._vis_rc_pts > 0 else 'MISS',
            lc_pts=self._vis_lc_pts,
            rc_pts=self._vis_rc_pts,
            center_x_px=center_x,
            lane_offset_norm=offset,
            lane_offset_m=lane_offset_m,
            lane_valid=lane_valid,
            epsi_rad=None,
            lc_coef=None, rc_coef=None, center_coef=None, half_width_coef=None,
            lc_residual=None, rc_residual=None,
            center_jumped=False,
            row_start=self._row_start,
            bev_enabled=True,
        )

        if self.show_window:
            self._update_display(binary, center_x)

    # ------------------------------------------------------------------
    # YOLOv8-seg 디코딩 (기존과 동일)
    # ------------------------------------------------------------------

    def _parse_yolov8seg(self, in_nn) -> np.ndarray | None:
        try:
            names = sorted(in_nn.getAllLayerNames())
            if len(names) < 2:
                return None

            raw0 = np.array(in_nn.getTensor(names[0])).flatten()
            raw1 = np.array(in_nn.getTensor(names[1])).flatten()

            H4 = self.IMG_H // 4
            W4 = self.IMG_W // 4
            expected0 = 37 * 2940
            expected1 = 32 * H4 * W4

            if raw0.size == expected1 and raw1.size == expected0:
                raw0, raw1 = raw1, raw0

            if raw0.size != expected0 or raw1.size != expected1:
                self.get_logger().warn(
                    f'출력 크기 불일치: {raw0.size} / {raw1.size}',
                    throttle_duration_sec=5.0,
                )
                return None

            preds  = raw0.reshape(37, 2940).T
            protos = raw1.reshape(32, H4, W4)

            scores = preds[:, 4]
            keep   = scores > self.conf_thresh
            if not np.any(keep):
                return None

            mask_coeffs = preds[keep, 5:]
            proto_flat  = protos.reshape(32, H4 * W4)
            masks_raw   = (mask_coeffs @ proto_flat).reshape(-1, H4, W4)
            masks       = 1.0 / (1.0 + np.exp(-masks_raw))
            combined    = np.max(masks, axis=0)

            binary_small = (combined > self.mask_thresh).astype(np.uint8)
            return cv2.resize(binary_small, (self.IMG_W, self.IMG_H),
                              interpolation=cv2.INTER_NEAREST)

        except Exception as exc:
            self.get_logger().warn(
                f'parse_yolov8seg 오류: {exc}', throttle_duration_sec=2.0
            )
            return None

    # ------------------------------------------------------------------
    # 도로 중심 X 계산 — 픽셀 카운트/모드 추적 추가
    # ------------------------------------------------------------------

    def _lane_center_x(self, binary: np.ndarray) -> float | None:
        H, W   = binary.shape
        cx_img = W / 2.0

        roi_start = int(H * self.roi_start_ratio)
        noise_cut = int((H - roi_start) * self.noise_cut_ratio)
        row_start = roi_start + noise_cut

        l_wsum, l_wtot = 0.0, 0.0
        r_wsum, r_wtot = 0.0, 0.0
        lc_total = 0
        rc_total = 0

        for row in range(row_start, H):
            cols = np.where(binary[row] > 0)[0]
            if cols.size < self.gap_thresh:
                continue

            w = float(row * row)
            left_cols  = cols[cols < cx_img]
            right_cols = cols[cols >= cx_img]

            if left_cols.size:
                l_wsum += w * float(np.mean(left_cols))
                l_wtot += w
                lc_total += left_cols.size
            if right_cols.size:
                r_wsum += w * float(np.mean(right_cols))
                r_wtot += w
                rc_total += right_cols.size

        self._vis_lc_pts = lc_total
        self._vis_rc_pts = rc_total

        lx = (l_wsum / l_wtot) if l_wtot > 0.0 else None
        rx = (r_wsum / r_wtot) if r_wtot > 0.0 else None

        if lx is not None and rx is not None:
            self._vis_center_mode = 'BOTH'
            return (lx + rx) / 2.0
        if lx is not None:
            self._vis_center_mode = 'LEFT'
            return lx + self.road_half_w
        if rx is not None:
            self._vis_center_mode = 'RIGHT'
            return rx - self.road_half_w
        self._vis_center_mode = 'NONE'
        return None

    # ------------------------------------------------------------------
    # 정규화
    # ------------------------------------------------------------------

    def _normalize_offset(self, center_x: float | None) -> float | None:
        if center_x is None:
            return None
        offset = (center_x - self.IMG_W / 2.0) / (self.IMG_W / 2.0)
        return float(np.clip(offset, -1.0, 1.0))

    # ------------------------------------------------------------------
    # 디버그 표시 (show_window=True 전용)
    # ------------------------------------------------------------------

    def _update_display(self, binary: np.ndarray | None, cx: float | None) -> None:
        raw_frame = None
        if hasattr(self, 'rgb_queue'):
            in_rgb = self.rgb_queue.tryGet()
            if in_rgb is not None:
                raw_frame = in_rgb.getCvFrame()

        # ── 창 1: 원본 + 사다리꼴 ─────────────────────────────────────
        if raw_frame is not None:
            orig_vis = raw_frame.copy()
            pts = self.warp_src.astype(np.int32)
            cv2.polylines(orig_vis, [pts[[0, 1, 2, 3]]], isClosed=True,
                          color=(0, 200, 255), thickness=2)
            labels = ['TL', 'TR', 'BR', 'BL']
            for i, (px, py) in enumerate(pts):
                cv2.circle(orig_vis, (px, py), 5, (0, 200, 255), -1)
                cv2.putText(orig_vis, labels[i], (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            cv2.imshow('original (trapezoid)', orig_vis)

        # ── 창 2: BEV 탑뷰 + NN 오버레이 ─────────────────────────────
        if raw_frame is not None:
            bev_frame = cv2.warpPerspective(
                raw_frame, self.M_display, (self.IMG_W, self.IMG_H)
            )
        else:
            bev_frame = None

        if bev_frame is not None:
            bev_vis = bev_frame.copy()
            if binary is not None:
                overlay = np.zeros_like(bev_vis)
                overlay[binary > 0] = (0, 255, 0)
                bev_vis = cv2.addWeighted(bev_vis, 0.7, overlay, 0.3, 0)
        elif binary is not None:
            bev_vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        else:
            if cv2.waitKey(1) == 27:
                self.get_logger().info('ESC 입력 — 노드 종료')
                rclpy.shutdown()
            return

        if cx is not None:
            cv2.line(bev_vis, (int(cx), 0), (int(cx), self.IMG_H - 1),
                     (0, 255, 0), 2)
        cv2.line(bev_vis, (self.IMG_W // 2, 0), (self.IMG_W // 2, self.IMG_H - 1),
                 (0, 0, 255), 1)

        # HUD: offset_m 표시
        if cx is not None and self.xm_per_pix > 0:
            off_m = (cx - self.IMG_W / 2.0) * self.xm_per_pix
            cv2.putText(bev_vis, f'off={off_m:.3f}m  {self._vis_center_mode}',
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

        cv2.imshow('BEV lane detection', bev_vis)
        if cv2.waitKey(1) == 27:
            self.get_logger().info('ESC 입력 — 노드 종료')
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._diag.close()
        if hasattr(self, 'pipeline'):
            self.pipeline = None
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorBEVNode()
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
