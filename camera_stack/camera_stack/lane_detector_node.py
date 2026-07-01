"""차선 인식 노드.

OAK-D Pro depthai 파이프라인으로 YOLOv8-seg blob 추론 →
차선 이진 마스크 → 도로 중심 X → 정규화 오프셋 /lane_offset 발행.

/lane_offset 부호 규칙: 차선 중심이 차량 기준 오른쪽 = +, 왼쪽 = -.
미검출 시 발행 침묵 (NaN 발행 금지).
"""

import csv
import datetime
import math
import os
import time

import cv2
import depthai as dai
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool, Float32

from camera_stack.diag_logger import DiagLogger


class LaneDetectorNode(Node):
    IMG_W = 640
    IMG_H = 224

    def __init__(self):
        super().__init__('lane_detector_node')

        _default_blob = os.path.join(
            get_package_share_directory('camera_stack'),
            'models', 'best_openvino_2022.1_6shave.blob'
        )
        self.declare_parameter('blob_path', _default_blob)
        self.declare_parameter('conf_thresh', 0.40)
        self.declare_parameter('mask_thresh', 0.45)
        self.declare_parameter('roi_start_ratio', 0.55)
        # 원본 마스크를 도로 평면으로 펴는 BEV 변환
        self.declare_parameter('use_bev', True)
        self.declare_parameter('bev_src_top_y_ratio', 0.620)
        self.declare_parameter('bev_src_top_left_ratio', 0.320)
        self.declare_parameter('bev_src_top_right_ratio', 0.680)
        self.declare_parameter('bev_src_bottom_left_ratio', 0.050)
        self.declare_parameter('bev_src_bottom_right_ratio', 0.950)
        self.declare_parameter('bev_dst_left_ratio', 0.250)
        self.declare_parameter('bev_dst_right_ratio', 0.750)
        self.declare_parameter('bev_scan_start_ratio', 0.05)
        self.declare_parameter('road_half_w', 80)
        self.declare_parameter('gap_thresh', 1)
        self.declare_parameter('show_window', False)
        self.declare_parameter('capture_dir', os.path.expanduser('~/capstone_ws/logs'))
        # 진단 CSV 로깅
        self.declare_parameter('log_diag', True)
        self.declare_parameter('log_path', False)   # True=경로 좌표 CSV 저장
        self.declare_parameter('xm_per_pix', 0.0)   # BEV 캘리브 후 설정 (px→meter)
        # 시간 필터 파라미터
        self.declare_parameter('ema_alpha', 0.5)   # EMA 반응속도 (0=완전 평균, 1=필터 없음)
        self.declare_parameter('hold_sec', 1.5)    # 미감지 시 마지막 값 유지 시간(초)
        self.declare_parameter('hold_decay', 0.97)  # HOLD 중 offset 프레임별 감쇠율
        # blob 형태 필터 파라미터
        self.declare_parameter('max_blob_ratio', 0.15)   # 이미지 면적 대비 최대 blob 크기 (벽 제거)
        self.declare_parameter('min_blob_area',  100)    # 최소 blob 픽셀 수 (노이즈 제거)
        self.declare_parameter('min_aspect_ratio', 1.5)  # 최소 가로세로 비율 (뭉툭한 반사광 제거)
        # 다항식 피팅 + 헤딩 오차 파라미터
        self.declare_parameter('min_poly_pts', 8)    # 다항식 피팅 최소 점 수
        self.declare_parameter('poly_degree', 2)      # 1=직선(주행 안정) / 2=곡률 실험용
        self.declare_parameter('epsi_weight', 0.3)    # 헤딩 오차 가중치 (0=사용 안 함, 직선로에서 권장)
        # 다항식 EMA 필터 파라미터 (A 필터)
        self.declare_parameter('poly_alpha', 0.25)    # 다항식 EMA 알파 (ema_alpha보다 느리게)
        self.declare_parameter('poly_hold_sec', 0.8)  # 차선 미감지 시 다항식 유지 시간(초)
        # row run 선택 + robust fitting 파라미터
        self.declare_parameter('min_run_width', 2)         # 행 내 차선 후보의 최소 연속 픽셀 폭
        self.declare_parameter('max_run_width', 45)        # 너무 두꺼운 run은 벽/콘/반사로 보고 제외
        self.declare_parameter('bev_edge_margin_px', 15)   # BEV 좌우 가장자리 mask 제거 폭
        self.declare_parameter('lane_search_margin', 45.0)  # 행간 최대 탐색 거리(px)
        self.declare_parameter('min_row_span', 12)          # 피팅 포인트가 차지해야 하는 최소 세로 범위(px)
        self.declare_parameter('fit_residual_px', 13.0)     # 1차 피팅 후 inlier 허용 잔차(px)
        self.declare_parameter('min_inlier_ratio', 0.45)   # 최종 피팅에 필요한 최소 inlier 비율
        self.declare_parameter('max_poly_slope', 8.0)      # ROI 내 다항식 기울기 최대 절댓값
        self.declare_parameter('max_valid_epsi_deg', 100.0)     # 주행에 쓸 최대 중심선 헤딩각
        self.declare_parameter('max_valid_path_span_px', 320.0) # 중심 경로 상하단 x 변화 허용치
        self.declare_parameter('poly_reject_delta', 50.0)  # 이전 EMA 대비 ROI 내 최대 위치 변화량(px)
        self.declare_parameter('min_lane_width', 40.0)     # ROI 내 최소 좌우 차선 폭(px)
        self.declare_parameter('max_lane_width', 680.0)    # ROI 중·하단 최대 좌우 차선 폭(px)
        self.declare_parameter('lane_width_top_min_px', 220.0)
        self.declare_parameter('lane_width_top_max_px', 430.0)
        self.declare_parameter('lane_width_mid_min_px', 320.0)
        self.declare_parameter('lane_width_mid_max_px', 460.0)
        self.declare_parameter('lane_width_bot_min_px', 360.0)
        self.declare_parameter('lane_width_bot_max_px', 540.0)
        self.declare_parameter('half_width_update_max_epsi_deg', 30.0)
        # 단일 차선 중심 복원 파라미터
        self.declare_parameter('single_lane_half_width_top', 203.0)
        self.declare_parameter('single_lane_half_width_bottom', 203.0)
        self.declare_parameter('half_width_alpha', 0.15)
        self.declare_parameter('single_lane_max_center_jump', 150.0)
        self.declare_parameter('single_lane_center_margin', 80.0)
        self.declare_parameter('single_lane_path_start_ratio', 0.55)
        self.declare_parameter('control_row_ratio', 0.72)
        # IMU 헤딩 교차 검증
        self.declare_parameter('use_imu_heading', True)
        self.declare_parameter('imu_epsi_suppress_deg', 5.0)

        blob_path = str(self.get_parameter('blob_path').value)
        self.conf_thresh       = float(self.get_parameter('conf_thresh').value)
        self.mask_thresh       = float(self.get_parameter('mask_thresh').value)
        self.roi_start_ratio   = float(self.get_parameter('roi_start_ratio').value)
        self.use_bev           = bool(self.get_parameter('use_bev').value)
        self.bev_src_top_y_ratio = float(
            self.get_parameter('bev_src_top_y_ratio').value
        )
        self.bev_src_top_left_ratio = float(
            self.get_parameter('bev_src_top_left_ratio').value
        )
        self.bev_src_top_right_ratio = float(
            self.get_parameter('bev_src_top_right_ratio').value
        )
        self.bev_src_bottom_left_ratio = float(
            self.get_parameter('bev_src_bottom_left_ratio').value
        )
        self.bev_src_bottom_right_ratio = float(
            self.get_parameter('bev_src_bottom_right_ratio').value
        )
        self.bev_dst_left_ratio = float(
            self.get_parameter('bev_dst_left_ratio').value
        )
        self.bev_dst_right_ratio = float(
            self.get_parameter('bev_dst_right_ratio').value
        )
        self.bev_scan_start_ratio = float(
            self.get_parameter('bev_scan_start_ratio').value
        )
        self.road_half_w       = int(self.get_parameter('road_half_w').value)
        self.gap_thresh        = int(self.get_parameter('gap_thresh').value)
        self.show_window       = bool(self.get_parameter('show_window').value)
        self.ema_alpha         = float(self.get_parameter('ema_alpha').value)
        self.hold_sec          = float(self.get_parameter('hold_sec').value)
        self.hold_decay        = float(self.get_parameter('hold_decay').value)
        self.max_blob_ratio    = float(self.get_parameter('max_blob_ratio').value)
        self.min_blob_area     = int(self.get_parameter('min_blob_area').value)
        self.min_aspect_ratio  = float(self.get_parameter('min_aspect_ratio').value)
        self.min_poly_pts      = int(self.get_parameter('min_poly_pts').value)
        self.poly_degree       = int(self.get_parameter('poly_degree').value)
        self.epsi_weight       = float(self.get_parameter('epsi_weight').value)
        self.poly_alpha        = float(self.get_parameter('poly_alpha').value)
        self.poly_hold_sec     = float(self.get_parameter('poly_hold_sec').value)
        self.min_run_width     = int(self.get_parameter('min_run_width').value)
        self.max_run_width     = int(self.get_parameter('max_run_width').value)
        self.bev_edge_margin_px = int(
            self.get_parameter('bev_edge_margin_px').value
        )
        self.lane_search_margin = float(self.get_parameter('lane_search_margin').value)
        self.min_row_span      = int(self.get_parameter('min_row_span').value)
        self.fit_residual_px   = float(self.get_parameter('fit_residual_px').value)
        self.min_inlier_ratio  = float(self.get_parameter('min_inlier_ratio').value)
        self.max_poly_slope    = float(self.get_parameter('max_poly_slope').value)
        self.max_valid_epsi_deg = float(
            self.get_parameter('max_valid_epsi_deg').value
        )
        self.max_valid_path_span_px = float(
            self.get_parameter('max_valid_path_span_px').value
        )
        self.poly_reject_delta = float(self.get_parameter('poly_reject_delta').value)
        self.min_lane_width    = float(self.get_parameter('min_lane_width').value)
        self.max_lane_width    = float(self.get_parameter('max_lane_width').value)
        self.lane_width_top_min_px = float(
            self.get_parameter('lane_width_top_min_px').value
        )
        self.lane_width_top_max_px = float(
            self.get_parameter('lane_width_top_max_px').value
        )
        self.lane_width_mid_min_px = float(
            self.get_parameter('lane_width_mid_min_px').value
        )
        self.lane_width_mid_max_px = float(
            self.get_parameter('lane_width_mid_max_px').value
        )
        self.lane_width_bot_min_px = float(
            self.get_parameter('lane_width_bot_min_px').value
        )
        self.lane_width_bot_max_px = float(
            self.get_parameter('lane_width_bot_max_px').value
        )
        self.half_width_update_max_epsi_deg = float(
            self.get_parameter('half_width_update_max_epsi_deg').value
        )
        self.single_lane_half_width_top = float(
            self.get_parameter('single_lane_half_width_top').value
        )
        self.single_lane_half_width_bottom = float(
            self.get_parameter('single_lane_half_width_bottom').value
        )
        self.half_width_alpha = float(
            self.get_parameter('half_width_alpha').value
        )
        self.single_lane_max_center_jump = float(
            self.get_parameter('single_lane_max_center_jump').value
        )
        self.single_lane_center_margin = float(
            self.get_parameter('single_lane_center_margin').value
        )
        self.single_lane_path_start_ratio = float(
            self.get_parameter('single_lane_path_start_ratio').value
        )
        self.control_row_ratio = float(
            self.get_parameter('control_row_ratio').value
        )
        self.use_imu_heading       = bool(self.get_parameter('use_imu_heading').value)
        self.imu_epsi_suppress_deg = float(
            self.get_parameter('imu_epsi_suppress_deg').value
        )
        self.capture_dir       = str(self.get_parameter('capture_dir').value)
        log_diag               = bool(self.get_parameter('log_diag').value)
        self._xm_per_pix       = float(self.get_parameter('xm_per_pix').value)
        xm_per_pix             = self._xm_per_pix
        os.makedirs(self.capture_dir, exist_ok=True)
        self._capture_count    = 0
        self._diag = DiagLogger(enabled=log_diag, save_dir=self.capture_dir,
                                xm_per_pix=xm_per_pix)

        # 경로 좌표 CSV
        log_path = bool(self.get_parameter('log_path').value)
        self._path_f      = None
        self._path_writer = None
        self._path_frame_no = 0
        if log_path:
            ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path_csv = os.path.join(self.capture_dir, f'lane_path_{ts_str}.csv')
            self._path_f = open(path_csv, 'w', newline='', encoding='utf-8')
            self._path_writer = csv.writer(self._path_f)
            self._path_writer.writerow([
                'ts_sec', 'frame_no', 'wp_idx',
                'row_px', 'x_px', 'dx_m',
                'curvature_r_m', 'heading_deg',
            ])
            self._path_f.flush()
            self.get_logger().info(f'[PathLog] {path_csv}')

        self._last_curvature_r = None   # HUD 표시용
        self._init_bev_transform()

        # 시간 필터 상태 (C 필터: scalar EMA)
        self._smooth_offset    = None   # EMA 누적값
        self._last_detect_time = 0.0    # 마지막 감지 시각
        # A 필터: 좌/우 차선 다항식 EMA 상태
        self._smooth_lc_coef   = None   # 왼쪽 차선 EMA 누적 다항식
        self._smooth_rc_coef   = None   # 오른쪽 차선 EMA 누적 다항식
        self._smooth_half_width_coef = self._default_half_width_coef()
        self._half_width_learned = False
        self._half_width_last_seen = 0.0
        self._lc_last_seen     = 0.0    # 왼쪽 차선 마지막 감지 시각
        self._rc_last_seen     = 0.0    # 오른쪽 차선 마지막 감지 시각
        self._prev_center_x    = None   # 동적 좌/우 분리 기준점

        # IMU 헤딩 상태
        self._imu_delta_yaw  = 0.0    # 마지막 유효 감지 이후 누적 회전량 (rad)
        self._imu_angular_z  = 0.0    # 최신 yaw rate (rad/s, CCW+)
        self._imu_last_t     = None   # 마지막 IMU 수신 시각 (time.time())
        self._imu_epsi_suppressed = False  # 이번 프레임에서 IMU가 epsi reject 억제했는지

        if not blob_path or not os.path.isfile(blob_path):
            raise FileNotFoundError(
                f'blob_path 파라미터를 설정하세요. 현재값: {blob_path!r}'
            )

        _latching_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub           = self.create_publisher(Float32, '/lane_offset',       10)
        self.pub_offset_m  = self.create_publisher(Float32, '/lane_offset_m',   10)
        self.pub_valid     = self.create_publisher(Bool,    '/lane_valid',       _latching_qos)
        self.pub_curvature = self.create_publisher(Float32, '/lane_curvature',   10)
        self.debug_pub     = self.create_publisher(Image,   '/lane_debug',       10)
        self.pub_raw_image = self.create_publisher(Image,   '/camera/image_raw', 10)
        self._bridge   = CvBridge()
        self._last_rgb_frame: np.ndarray | None = None

        if self.use_imu_heading:
            self.create_subscription(Imu, '/imu/data', self._on_imu, 10)

        # 디스플레이용 피팅 결과 캐시
        self._vis_left_pts    = []
        self._vis_right_pts   = []
        self._vis_lc_coef     = None
        self._vis_rc_coef     = None
        self._vis_center_coef = None   # 차량이 추종할 중심 경로 다항식
        self._vis_raw_lc_coef = None
        self._vis_raw_rc_coef = None
        self._vis_left_candidates = []
        self._vis_right_candidates = []
        self._vis_left_rejected = []
        self._vis_right_rejected = []
        self._vis_left_status = 'MISS'
        self._vis_right_status = 'MISS'
        self._vis_left_residual = None
        self._vis_right_residual = None
        self._vis_center_mode = 'NONE'
        # 진단 로그 신규 필드
        self._vis_reject_reason    = ''
        self._vis_left_total_pts   = 0
        self._vis_right_total_pts  = 0
        self._vis_left_row_span    = 0.0
        self._vis_right_row_span   = 0.0
        self._vis_left_mean_run_w  = 0.0
        self._vis_right_mean_run_w = 0.0
        self._vis_lane_width_top   = None
        self._vis_lane_width_mid   = None
        self._vis_lane_width_bot   = None
        self._vis_center_jump_px   = 0.0
        self._vis_raw_mask_px      = 0
        self._vis_filtered_mask_px = 0
        self._vis_bev_mask_px      = 0
        self._vis_path_x_span      = None
        self._vis_lane_valid       = False
        self._vis_lane_state       = 'MISS'

        self._init_device(blob_path)
        # ~30Hz — Nav2 20Hz보다 빠르게 폴링
        self.create_timer(0.033, self._process_cb)

        self.get_logger().info(
            f'lane_detector_node ready  blob={os.path.basename(blob_path)}'
            f'  conf={self.conf_thresh}  mask={self.mask_thresh}'
            f'  bev={self.use_bev}'
            f'  single_width={self.single_lane_half_width_top:.0f}'
            f'->{self.single_lane_half_width_bottom:.0f}px'
        )

    def _init_bev_transform(self) -> None:
        """정규화 파라미터로 원본↔BEV perspective transform을 준비한다."""
        W = float(self.IMG_W - 1)
        H = float(self.IMG_H - 1)
        top_y = float(np.clip(self.bev_src_top_y_ratio, 0.0, 0.99)) * H

        src = np.float32([
            [np.clip(self.bev_src_top_left_ratio, 0.0, 1.0) * W, top_y],
            [np.clip(self.bev_src_top_right_ratio, 0.0, 1.0) * W, top_y],
            [np.clip(self.bev_src_bottom_right_ratio, 0.0, 1.0) * W, H],
            [np.clip(self.bev_src_bottom_left_ratio, 0.0, 1.0) * W, H],
        ])
        dst = np.float32([
            [np.clip(self.bev_dst_left_ratio, 0.0, 1.0) * W, 0.0],
            [np.clip(self.bev_dst_right_ratio, 0.0, 1.0) * W, 0.0],
            [np.clip(self.bev_dst_right_ratio, 0.0, 1.0) * W, H],
            [np.clip(self.bev_dst_left_ratio, 0.0, 1.0) * W, H],
        ])

        if not (
            src[0, 0] < src[1, 0]
            and src[3, 0] < src[2, 0]
            and dst[0, 0] < dst[1, 0]
        ):
            raise ValueError('BEV 좌/우 점 순서가 잘못되었습니다.')

        self._bev_src_points = src
        self._bev_matrix = cv2.getPerspectiveTransform(src, dst)

    def _to_bev(self, image: np.ndarray, interpolation: int) -> np.ndarray:
        if not self.use_bev:
            return image
        return cv2.warpPerspective(
            image,
            self._bev_matrix,
            (self.IMG_W, self.IMG_H),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _apply_bev_mask_filters(self, binary: np.ndarray) -> np.ndarray:
        """BEV 변환 뒤 차선으로 쓰기 어려운 외곽 mask를 제거한다."""
        if not self.use_bev or binary is None:
            return binary
        margin = int(np.clip(self.bev_edge_margin_px, 0, self.IMG_W // 3))
        if margin <= 0:
            return binary
        filtered = binary.copy()
        filtered[:, :margin] = 0
        filtered[:, self.IMG_W - margin:] = 0
        return filtered

    def _default_half_width_coef(self) -> np.ndarray:
        """스캔 상단/하단 반폭을 연결한 기본 단일 차선 폭 모델."""
        scan_ratio = self.bev_scan_start_ratio if self.use_bev else self.roi_start_ratio
        row_top = float(int(self.IMG_H * scan_ratio))
        row_bottom = float(self.IMG_H - 1)
        row_span = max(row_bottom - row_top, 1.0)
        slope = (
            self.single_lane_half_width_bottom
            - self.single_lane_half_width_top
        ) / row_span
        intercept = self.single_lane_half_width_top - slope * row_top
        return np.array([0.0, slope, intercept], dtype=float)

    # ------------------------------------------------------------------
    # IMU 콜백
    # ------------------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        """yaw rate 누적 → _imu_delta_yaw.

        마지막 유효 차선 감지 이후 차량이 얼마나 회전했는지 추적한다.
        _process_cb()에서 offset이 갱신될 때마다 리셋된다.
        """
        now = time.time()
        if self._imu_last_t is not None:
            dt = now - self._imu_last_t
            if 0.0 < dt < 0.1:
                self._imu_delta_yaw += msg.angular_velocity.z * dt
        self._imu_last_t = now
        self._imu_angular_z = msg.angular_velocity.z

    # ------------------------------------------------------------------
    # depthai 초기화
    # ------------------------------------------------------------------

    def _init_device(self, blob_path: str) -> None:
        # depthai 3.x: Device를 Pipeline에 주입, start()로 실행
        self.pipeline = dai.Pipeline(dai.Device())

        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            (self.IMG_W, self.IMG_H),
            type=dai.ImgFrame.Type.BGR888p,
            fps=25,
        )

        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(blob_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        cam_out.link(nn.input)

        # createOutputQueue는 반드시 pipeline.start() 이전에 호출
        self.nn_queue  = nn.out.createOutputQueue(maxSize=4, blocking=False)
        self.rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

        self.pipeline.start()

    # ------------------------------------------------------------------
    # 메인 처리 (타이머 콜백)
    # ------------------------------------------------------------------

    def _process_cb(self) -> None:
        # RGB 프레임을 먼저 소비해 큐를 비우고 /camera/image_raw 발행
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb is not None:
            self._last_rgb_frame = in_rgb.getCvFrame()
            img_msg = self._bridge.cv2_to_imgmsg(self._last_rgb_frame, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            self.pub_raw_image.publish(img_msg)

        in_nn = self.nn_queue.tryGet()
        if in_nn is None:
            # NN 결과 없을 때 이전 프레임 박스가 잔류하지 않도록 초기화
            self._vis_anchors = []
            self._vis_max_conf = 0.0
            self._vis_nn_pass_count = 0
            if self.show_window:
                self._update_display(None, None)
            return

        raw_binary = self._parse_yolov8seg(in_nn)
        binary = (
            self._to_bev(raw_binary, cv2.INTER_NEAREST)
            if raw_binary is not None else None
        )
        if binary is not None:
            binary = (binary > 0).astype(np.uint8)
            self._vis_bev_mask_px = int(binary.sum())
            binary = self._apply_bev_mask_filters(binary)
            self._vis_filtered_mask_px = int(binary.sum())
        else:
            self._vis_bev_mask_px = 0
            self._vis_filtered_mask_px = 0
            self._vis_center_coef = None
            self._vis_center_mode = 'NONE'
            self._vis_left_status = 'MISS'
            self._vis_right_status = 'MISS'
            self._vis_reject_reason = 'no_mask'
        center_x, epsi = (
            self._lane_center_poly(binary)
            if binary is not None else (None, None)
        )
        path_x_span = self._compute_path_x_span()
        lane_valid = self._evaluate_lane_valid(center_x, epsi, path_x_span)
        offset = self._compute_offset(center_x, epsi) if lane_valid else None
        self._last_epsi    = epsi   # 디스플레이용

        now = time.time()

        if offset is not None:
            # 감지 성공 → EMA 평활화
            if self._smooth_offset is None:
                self._smooth_offset = offset
            else:
                self._smooth_offset = (self.ema_alpha * offset
                                       + (1.0 - self.ema_alpha) * self._smooth_offset)
            self._last_detect_time = now
            self._imu_delta_yaw = 0.0   # 유효 감지마다 누적 회전량 리셋

        # 발행: 감지됐거나 hold_sec 이내 미감지면 마지막 평활 값 사용
        if self._smooth_offset is not None:
            if offset is not None or (now - self._last_detect_time) < self.hold_sec:
                if offset is None:
                    # 직진 중(IMU yaw rate < 0.05 rad/s)이면 decay 억제
                    # IMU 미수신(_imu_last_t is None)이면 기존 decay 그대로 적용
                    imu_straight = (
                        self.use_imu_heading
                        and self._imu_last_t is not None
                        and abs(self._imu_angular_z) < 0.05
                    )
                    if not imu_straight:
                        decay = float(np.clip(self.hold_decay, 0.0, 1.0))
                        self._smooth_offset *= decay
                msg = Float32()
                msg.data = float(self._smooth_offset)
                self.pub.publish(msg)
                raw_str    = f'{offset:.3f}' if offset is not None else 'MISS'
                epsi_str   = f'{math.degrees(epsi):.1f}°' if epsi is not None else '-'
                state_str  = self._vis_lane_state
                self.get_logger().info(
                    f'[offset] raw={raw_str}  smooth={self._smooth_offset:.3f}'
                    f'  epsi={epsi_str}  {state_str}',
                    throttle_duration_sec=0.5,
                )
            else:
                # hold 시간 초과 → 상태 초기화
                self._smooth_offset = None

        # ── /lane_offset_m, /lane_valid, /lane_curvature 발행 ─────────
        lane_offset_m = None
        if self._smooth_offset is not None and self._xm_per_pix > 0:
            lane_offset_m = float(self._smooth_offset) * (self.IMG_W / 2.0) * self._xm_per_pix
            msg_m = Float32()
            msg_m.data = lane_offset_m
            self.pub_offset_m.publish(msg_m)
        msg_valid = Bool()
        msg_valid.data = lane_valid
        self.pub_valid.publish(msg_valid)

        # /lane_curvature: poly_degree=2일 때만 유효한 값, 1차면 1e6(직선)
        row_eval = float(self.IMG_H - 1)
        curv_r = self._compute_curvature_m(self._vis_center_coef, row_eval)
        self._last_curvature_r = curv_r
        msg_curv = Float32()
        msg_curv.data = float(min(curv_r, 1e6))
        self.pub_curvature.publish(msg_curv)

        # path_x_span: center poly가 scan 상단~하단에서 x축으로 이동한 거리(px)
        _vis_path_x_span = path_x_span

        # 경로 좌표 CSV
        self._write_path_csv(now, center_x, curv_r, epsi, lane_valid)

        self._publish_debug(binary, center_x)
        if self.show_window:
            self._update_display(binary, center_x)

        # ── 진단 CSV 로깅 ─────────────────────────────────────────────
        self._diag.write_row(
            ts_sec           = now,
            center_mode      = self._vis_center_mode,
            lc_status        = self._vis_left_status,
            rc_status        = self._vis_right_status,
            lc_pts           = len(self._vis_left_pts),
            rc_pts           = len(self._vis_right_pts),
            center_x_px      = center_x,
            lane_offset_norm = self._smooth_offset,
            lane_offset_m    = lane_offset_m,
            lane_valid       = lane_valid,
            epsi_rad         = epsi,
            lc_coef          = self._vis_lc_coef,
            rc_coef          = self._vis_rc_coef,
            center_coef      = self._vis_center_coef,
            half_width_coef  = self._smooth_half_width_coef,
            lc_residual      = self._vis_left_residual,
            rc_residual      = self._vis_right_residual,
            center_jumped    = (self._vis_center_mode == 'REJECT'),
            row_start        = getattr(self, '_vis_row_start', 0),
            bev_enabled      = self.use_bev,
            # 신규
            reject_reason    = self._vis_reject_reason,
            left_total_pts   = self._vis_left_total_pts,
            right_total_pts  = self._vis_right_total_pts,
            left_row_span_px = self._vis_left_row_span,
            right_row_span_px= self._vis_right_row_span,
            left_mean_run_w  = self._vis_left_mean_run_w,
            right_mean_run_w = self._vis_right_mean_run_w,
            path_x_span_px   = _vis_path_x_span,
            curvature_r_m    = float(min(curv_r, 1e6)),
            raw_mask_px      = self._vis_raw_mask_px,
            bev_mask_px      = self._vis_bev_mask_px,
            filtered_mask_px = self._vis_filtered_mask_px,
            lane_width_top_px= self._vis_lane_width_top,
            lane_width_mid_px= self._vis_lane_width_mid,
            lane_width_bot_px= self._vis_lane_width_bot,
            center_jump_px   = self._vis_center_jump_px,
            # IMU
            imu_fresh            = (self._imu_last_t is not None),
            imu_angular_z_rads   = self._imu_angular_z,
            imu_delta_yaw_deg    = math.degrees(self._imu_delta_yaw),
            imu_epsi_suppressed  = self._imu_epsi_suppressed,
            # NN 신뢰도
            nn_max_conf   = getattr(self, '_vis_max_conf', 0.0),
            nn_pass_count = getattr(self, '_vis_nn_pass_count', 0),
        )

    # ------------------------------------------------------------------
    # YOLOv8-seg 디코딩
    # ------------------------------------------------------------------

    def _parse_yolov8seg(self, in_nn) -> np.ndarray | None:
        """YOLOv8-seg NNData → 이진 차선 마스크 (IMG_H × IMG_W).

        output0: (37 × 2940) — [cx, cy, w, h, conf, mask_coeff×32]
        output1: (32 × H4 × W4) — mask prototypes  H4=56, W4=160
        """
        self._vis_raw_mask_px = 0
        self._vis_filtered_mask_px = 0
        try:
            names = sorted(in_nn.getAllLayerNames())
            if len(names) < 2:
                self.get_logger().warn('NN 출력 레이어 < 2', throttle_duration_sec=5.0)
                return None

            raw0 = np.array(in_nn.getTensor(names[0])).flatten()
            raw1 = np.array(in_nn.getTensor(names[1])).flatten()

            H4 = self.IMG_H // 4   # 56
            W4 = self.IMG_W // 4   # 160
            expected0 = 37 * 2940
            expected1 = 32 * H4 * W4

            # 정렬 순서가 틀렸을 경우 교체
            if raw0.size == expected1 and raw1.size == expected0:
                raw0, raw1 = raw1, raw0

            if raw0.size != expected0 or raw1.size != expected1:
                self.get_logger().warn(
                    f'출력 크기 불일치: output0={raw0.size}(expect {expected0})'
                    f'  output1={raw1.size}(expect {expected1})',
                    throttle_duration_sec=5.0,
                )
                return None

            preds = raw0.reshape(37, 2940).T      # (2940, 37)
            protos = raw1.reshape(32, H4, W4)     # (32, 56, 160)

            scores = preds[:, 4]
            keep = scores > self.conf_thresh
            self.get_logger().info(
                f'[NN] 최대신뢰도={scores.max():.3f}  '
                f'통과앵커={int(keep.sum())}개  '
                f'(threshold={self.conf_thresh})',
                throttle_duration_sec=1.0,
            )

            # 시각화용 앵커 정보 저장 (show_window=True 시 _update_display에서 사용)
            self._vis_max_conf = float(scores.max())
            self._vis_nn_pass_count = int(keep.sum())
            self._vis_anchors = []

            if not np.any(keep):
                return None

            for pred in preds[keep]:
                self._vis_anchors.append((
                    float(pred[0]), float(pred[1]),   # cx, cy (픽셀)
                    float(pred[2]), float(pred[3]),   # w,  h  (픽셀)
                    float(pred[4]),                   # conf
                ))

            mask_coeffs = preds[keep, 5:]                        # (N, 32)
            proto_flat = protos.reshape(32, H4 * W4)             # (32, 56×160)
            masks_raw = (mask_coeffs @ proto_flat).reshape(-1, H4, W4)  # (N, 56, 160)
            masks = 1.0 / (1.0 + np.exp(-masks_raw))            # sigmoid
            combined = np.max(masks, axis=0)                     # (56, 160)

            binary_small = (combined > self.mask_thresh).astype(np.uint8)
            binary = cv2.resize(binary_small, (self.IMG_W, self.IMG_H),
                                interpolation=cv2.INTER_NEAREST)

            # ROI 바깥(상단) 제거: 계산에 안 쓰이는 영역의 노이즈를 시각적으로도 숨김
            roi_cut = int(self.IMG_H * self.roi_start_ratio)
            binary[:roi_cut, :] = 0

            self._vis_raw_mask_px = int(binary.sum())   # blob 필터 전 픽셀 수

            # blob 형태 필터: 벽·반사광 제거
            binary = self._filter_blobs(binary)

            self._vis_filtered_mask_px = int(binary.sum())   # blob 필터 후 픽셀 수

            # blob 필터 후 실제로 살아남은 픽셀이 없으면 박스도 제거
            if binary.sum() == 0:
                self._vis_anchors = []

            return binary

        except Exception as exc:
            self.get_logger().warn(
                f'parse_yolov8seg 오류: {exc}', throttle_duration_sec=2.0
            )
            return None

    # ------------------------------------------------------------------
    # blob 형태 필터 (벽·반사광 후처리)
    # ------------------------------------------------------------------

    def _filter_blobs(self, binary: np.ndarray) -> np.ndarray:
        """연결된 덩어리(blob) 단위로 벽·노이즈·반사광 제거.

        제거 조건:
          1. 이미지 면적의 max_blob_ratio 이상 → 벽·대형 반사
          2. min_blob_area 픽셀 미만          → 점 노이즈
          3. 가로세로 비율 min_aspect_ratio 미만 → 둥근 반사광 패치
        차선은 가늘고 길어서 세 조건 모두 통과합니다.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        img_area = binary.shape[0] * binary.shape[1]
        filtered = np.zeros_like(binary)

        for i in range(1, num_labels):   # 0 = 배경
            area = int(stats[i, cv2.CC_STAT_AREA])
            w    = int(stats[i, cv2.CC_STAT_WIDTH])
            h    = int(stats[i, cv2.CC_STAT_HEIGHT])

            if area > img_area * self.max_blob_ratio:   # 너무 큰 blob → 벽
                continue
            if area < self.min_blob_area:               # 너무 작은 blob → 노이즈
                continue
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect < self.min_aspect_ratio:          # 뭉툭한 blob → 반사광
                continue

            filtered[labels == i] = 1

        return filtered

    # ------------------------------------------------------------------
    # 도로 중심 X + 헤딩 오차 계산 (2차 다항식 피팅)
    # ------------------------------------------------------------------

    def _lane_center_poly(self, binary: np.ndarray) -> tuple:
        """이진 마스크 → (center_x, epsi).

        행별 연속 픽셀 run 중 예상 차선에 가까운 후보를 추적하고,
        residual 기반 outlier 제거 후 다항식을 피팅한다.

        center_x : 도로 중심 X (픽셀)
        epsi     : 차선 중심선 기울기 각도 (rad).  수직 = 0,
                   우측으로 기울 때 > 0 (시계방향 보정 필요).

        Case A: 양쪽 모두 → center = (left + right)/2
        Case B: 좌측만    → center = left + 최근 y별 반폭 모델
        Case C: 우측만    → center = right - 최근 y별 반폭 모델
        Case D: 없음      → (None, None)
        """
        H, W = binary.shape
        # 이전 프레임 중심을 분리 기준으로 사용 (동적 split) — 초기에는 화면 중앙
        split_x = self._prev_center_x if self._prev_center_x is not None else W / 2.0

        scan_ratio = self.bev_scan_start_ratio if self.use_bev else self.roi_start_ratio
        roi_start = int(H * scan_ratio)
        row_start = roi_start

        def poly_eval(coef, row):
            return coef[0] * row**2 + coef[1] * row + coef[2]

        def row_runs(row):
            cols = np.where(binary[row] > 0)[0]
            if cols.size < self.gap_thresh:
                return []
            groups = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
            return [
                (float((group[0] + group[-1]) / 2.0), int(group.size))
                for group in groups
                if self.min_run_width <= group.size <= self.max_run_width
            ]

        def collect_lane(side, smooth_coef):
            points = []
            run_widths = []
            candidates = []
            rejected = []
            tracked_x = None

            # 차량에 가까운 하단에서 시작해 위쪽으로 같은 run을 추적한다.
            for row in range(H - 1, row_start - 1, -1):
                runs = row_runs(row)
                if side == 'left':
                    runs = [run for run in runs if run[0] < split_x]
                else:
                    runs = [run for run in runs if run[0] >= split_x]

                candidates.extend((row, center) for center, _ in runs)
                if not runs:
                    continue

                if smooth_coef is not None:
                    expected_x = poly_eval(smooth_coef, row)
                    nearby = [
                        run for run in runs
                        if abs(run[0] - expected_x) <= self.lane_search_margin
                    ]
                    if not nearby:
                        rejected.extend((row, center) for center, _ in runs)
                        continue
                    selected = min(nearby, key=lambda run: abs(run[0] - expected_x))
                elif tracked_x is not None:
                    nearby = [
                        run for run in runs
                        if abs(run[0] - tracked_x) <= self.lane_search_margin
                    ]
                    if not nearby:
                        rejected.extend((row, center) for center, _ in runs)
                        continue
                    selected = min(nearby, key=lambda run: abs(run[0] - tracked_x))
                else:
                    # 초기 프레임은 중앙 반사보다 바깥쪽 실제 차선을 우선한다.
                    selected = min(runs, key=lambda run: run[0]) if side == 'left' \
                        else max(runs, key=lambda run: run[0])

                tracked_x = selected[0]
                points.append((row, tracked_x))
                run_widths.append(selected[1])
                rejected.extend(
                    (row, center) for center, _ in runs if center != tracked_x
                )

            points.reverse()
            candidates.sort()
            rejected.sort()
            return points, candidates, rejected, run_widths

        left_points, left_candidates, left_rejected, left_run_widths = collect_lane(
            'left', self._smooth_lc_coef
        )
        right_points, right_candidates, right_rejected, right_run_widths = collect_lane(
            'right', self._smooth_rc_coef
        )

        def fit(points, side):
            stats = {
                'residual': None,
                'status': 'MISS',
                'inliers': [],
                'rejected': [],
                'raw_coef': None,
                'row_min': None,
                'row_max': None,
            }
            if len(points) < self.min_poly_pts:
                stats['status'] = 'FEW'
                return None, stats

            rows = np.array([point[0] for point in points], dtype=float)
            cols = np.array([point[1] for point in points], dtype=float)
            if rows[-1] - rows[0] < self.min_row_span:
                stats['status'] = 'SPAN'
                return None, stats

            try:
                raw = np.polyfit(rows, cols, self.poly_degree)
            except Exception:
                stats['status'] = 'FIT'
                return None, stats

            if self.poly_degree == 1:
                raw = np.array([0.0, raw[0], raw[1]])
            stats['raw_coef'] = raw.copy()

            residuals = np.abs(cols - np.array([poly_eval(raw, row) for row in rows]))
            inlier_mask = residuals <= self.fit_residual_px
            inlier_count = int(np.count_nonzero(inlier_mask))
            if (inlier_count < self.min_poly_pts
                    or inlier_count / len(points) < self.min_inlier_ratio):
                stats['status'] = 'OUTLIER'
                stats['rejected'] = [
                    point for point, valid in zip(points, inlier_mask) if not valid
                ]
                return None, stats

            inlier_rows = rows[inlier_mask]
            inlier_cols = cols[inlier_mask]
            try:
                coef = np.polyfit(inlier_rows, inlier_cols, self.poly_degree)
            except Exception:
                stats['status'] = 'REFIT'
                return None, stats
            if self.poly_degree == 1:
                coef = np.array([0.0, coef[0], coef[1]])

            row_min = float(np.min(inlier_rows))
            row_max = float(np.max(inlier_rows))
            check_rows = np.linspace(row_min, row_max, 3)
            slopes = 2.0 * coef[0] * check_rows + coef[1]
            if np.max(np.abs(slopes)) > self.max_poly_slope:
                stats['status'] = 'SLOPE'
                return None, stats

            final_residuals = np.abs(
                inlier_cols
                - np.array([poly_eval(coef, row) for row in inlier_rows])
            )
            stats['residual'] = float(np.mean(final_residuals))
            stats['status'] = 'RAW'
            stats['row_min'] = row_min
            stats['row_max'] = row_max
            stats['inliers'] = [
                point for point, valid in zip(points, inlier_mask) if valid
            ]
            stats['rejected'] = [
                point for point, valid in zip(points, inlier_mask) if not valid
            ]
            return coef, stats

        lc_coef, lc_stats = fit(left_points, 'left')
        rc_coef, rc_stats = fit(right_points, 'right')

        # 1순위: 피팅 전후 진단 데이터 갱신
        self._vis_left_total_pts  = len(left_points)
        self._vis_right_total_pts = len(right_points)
        self._vis_left_mean_run_w  = (float(np.mean(left_run_widths))
                                      if left_run_widths else 0.0)
        self._vis_right_mean_run_w = (float(np.mean(right_run_widths))
                                      if right_run_widths else 0.0)
        self._vis_left_row_span  = (float(lc_stats['row_max'] - lc_stats['row_min'])
                                    if lc_stats['row_min'] is not None else 0.0)
        self._vis_right_row_span = (float(rc_stats['row_max'] - rc_stats['row_min'])
                                    if rc_stats['row_min'] is not None else 0.0)
        # 차선 폭/점프/reject 필드를 기본값으로 초기화 (이후 코드 경로에서 덮어씀)
        self._vis_lane_width_top = None
        self._vis_lane_width_mid = None
        self._vis_lane_width_bot = None
        self._vis_center_jump_px = 0.0
        self._vis_reject_reason  = ''

        # ── A 필터: 좌/우 차선 다항식 독립 EMA ──────────────────────────
        now = time.time()

        def update_smooth(raw_coef, smooth_coef, last_seen, stats, label):
            accepted = False
            if raw_coef is not None:
                if smooth_coef is None:
                    smooth_coef = raw_coef.copy()
                    accepted = True
                else:
                    compare_rows = np.linspace(
                        stats['row_min'], stats['row_max'], 3
                    )
                    new_x = np.array([poly_eval(raw_coef, row) for row in compare_rows])
                    old_x = np.array([poly_eval(smooth_coef, row) for row in compare_rows])
                    max_delta = float(np.max(np.abs(new_x - old_x)))
                    if max_delta <= self.poly_reject_delta:
                        smooth_coef = (
                            self.poly_alpha * raw_coef
                            + (1.0 - self.poly_alpha) * smooth_coef
                        )
                        accepted = True
                    else:
                        stats['status'] = 'JUMP'
                        self.get_logger().warn(
                            f'{label} rejected: max_delta={max_delta:.1f}px'
                            f' rows={stats["row_min"]:.0f}-{stats["row_max"]:.0f}'
                            f' (limit={self.poly_reject_delta})',
                            throttle_duration_sec=0.5,
                        )

            if accepted:
                last_seen = now
                stats['status'] = 'ACCEPT'
            elif smooth_coef is not None:
                if (now - last_seen) > self.poly_hold_sec:
                    smooth_coef = None
                    stats['status'] = 'MISS'
                else:
                    stats['status'] = f'HOLD/{stats["status"]}'
            return smooth_coef, last_seen

        self._smooth_lc_coef, self._lc_last_seen = update_smooth(
            lc_coef, self._smooth_lc_coef, self._lc_last_seen, lc_stats, 'left'
        )
        self._smooth_rc_coef, self._rc_last_seen = update_smooth(
            rc_coef, self._smooth_rc_coef, self._rc_last_seen, rc_stats, 'right'
        )

        # 이후 연산은 smooth 계수만 사용
        lc_use = self._smooth_lc_coef
        rc_use = self._smooth_rc_coef
        lc_current = lc_stats['status'] == 'ACCEPT'
        rc_current = rc_stats['status'] == 'ACCEPT'
        center_mode = 'NONE'
        path_row_start = row_start

        def single_lane_center(side):
            if side == 'LEFT':
                candidate = lc_use + self._smooth_half_width_coef
            else:
                candidate = rc_use - self._smooth_half_width_coef

            check_start = max(
                row_start,
                int(H * np.clip(self.single_lane_path_start_ratio, 0.0, 0.95)),
            )
            check_rows = np.linspace(check_start, H - 1, 3)
            check_x = np.array([poly_eval(candidate, row) for row in check_rows])
            margin = self.single_lane_center_margin
            if np.any(check_x < -margin) or np.any(check_x > (W - 1 + margin)):
                self.get_logger().warn(
                    f'{side.lower()} single-lane center rejected:'
                    f' rows={check_rows.round(1).tolist()}'
                    f' x={check_x.round(1).tolist()}',
                    throttle_duration_sec=0.5,
                )
                return None
            return candidate

        def lane_quality(stats):
            row_span = (
                stats['row_max'] - stats['row_min']
                if stats['row_min'] is not None else 0.0
            )
            residual = stats['residual'] if stats['residual'] is not None else 1e6
            return row_span + len(stats['inliers']), -residual

        def lane_width_sane(top_w, mid_w, bot_w, center_coef):
            mid_row = float((row_start + H) / 2.0)
            slope = 2.0 * center_coef[0] * mid_row + center_coef[1]
            epsi_deg = abs(math.degrees(math.atan(slope)))
            return (
                self.lane_width_top_min_px <= top_w <= self.lane_width_top_max_px
                and self.lane_width_mid_min_px <= mid_w <= self.lane_width_mid_max_px
                and self.lane_width_bot_min_px <= bot_w <= self.lane_width_bot_max_px
                and epsi_deg <= self.half_width_update_max_epsi_deg
            )

        # ── 중심 경로 다항식(center_coef) 생성 ──────────────────────────
        if lc_current and rc_current:
            common_start = max(lc_stats['row_min'], rc_stats['row_min'])
            common_end = min(lc_stats['row_max'], rc_stats['row_max'])
            common_span = common_end - common_start

            if common_span >= self.min_row_span:
                width_rows = np.linspace(common_start, common_end, 3)
                left_x = np.array([poly_eval(lc_use, row) for row in width_rows])
                right_x = np.array([poly_eval(rc_use, row) for row in width_rows])
                widths = right_x - left_x
                width_valid = (
                    np.all(left_x >= 0.0)
                    and np.all(right_x <= W - 1.0)
                    and np.all(widths >= self.min_lane_width)
                    and np.all(widths <= self.max_lane_width)
                )
            else:
                width_rows = np.array([])
                widths = np.array([])
                width_valid = False

            if width_valid:
                center_coef = (lc_use + rc_use) / 2.0
                center_mode = 'BOTH'
                # 3순위: 상/중/하단 차선 폭 측정
                _mid_r = float((row_start + H) // 2)
                self._vis_lane_width_top = float(poly_eval(rc_use, float(row_start))
                                                 - poly_eval(lc_use, float(row_start)))
                self._vis_lane_width_mid = float(poly_eval(rc_use, _mid_r)
                                                 - poly_eval(lc_use, _mid_r))
                self._vis_lane_width_bot = float(poly_eval(rc_use, float(H - 1))
                                                 - poly_eval(lc_use, float(H - 1)))
                if lane_width_sane(
                    self._vis_lane_width_top,
                    self._vis_lane_width_mid,
                    self._vis_lane_width_bot,
                    center_coef,
                ):
                    measured_half_width = (rc_use - lc_use) / 2.0
                    alpha = float(np.clip(self.half_width_alpha, 0.0, 1.0))
                    self._smooth_half_width_coef = (
                        alpha * measured_half_width
                        + (1.0 - alpha) * self._smooth_half_width_coef
                    )
                    self._half_width_learned = True
                    self._half_width_last_seen = now
                else:
                    self.get_logger().warn(
                        'half_width update skipped:'
                        f' width=[{self._vis_lane_width_top:.1f},'
                        f' {self._vis_lane_width_mid:.1f},'
                        f' {self._vis_lane_width_bot:.1f}]',
                        throttle_duration_sec=0.5,
                    )
            else:
                # 폭 관계가 모순되면 포인트 분포와 residual이 더 좋은 한쪽만
                # 사용한다. 기본/학습 반폭 모델이 중심 경로를 복원한다.
                if lane_quality(lc_stats) >= lane_quality(rc_stats):
                    center_coef = single_lane_center('LEFT')
                    center_mode = 'LEFT/WIDTH' if center_coef is not None else 'NONE'
                    lc_stats['status'] = 'SINGLE'
                    rc_stats['status'] = 'WIDTH'
                else:
                    center_coef = single_lane_center('RIGHT')
                    center_mode = 'RIGHT/WIDTH' if center_coef is not None else 'NONE'
                    lc_stats['status'] = 'WIDTH'
                    rc_stats['status'] = 'SINGLE'
                self.get_logger().warn(
                    f'lane width rejected: rows={width_rows.round(1).tolist()}'
                    f' widths={widths.round(1).tolist()}'
                    f' common_span={common_span:.1f}'
                    f' fallback={center_mode}',
                    throttle_duration_sec=0.5,
                )
        elif lc_current:                                      # Case B: 왼쪽만
            center_coef = single_lane_center('LEFT')
            center_mode = 'LEFT' if center_coef is not None else 'NONE'
            if center_coef is not None:
                lc_stats['status'] = 'SINGLE'
        elif rc_current:                                      # Case C: 오른쪽만
            center_coef = single_lane_center('RIGHT')
            center_mode = 'RIGHT' if center_coef is not None else 'NONE'
            if center_coef is not None:
                rc_stats['status'] = 'SINGLE'
        else:                                                  # Case D: 없음
            center_coef = None

        if center_coef is not None and center_mode != 'BOTH':
            path_row_start = max(
                row_start,
                int(H * np.clip(self.single_lane_path_start_ratio, 0.0, 0.95)),
            )

        if center_coef is None:
            self._vis_left_pts    = lc_stats['inliers']
            self._vis_right_pts   = rc_stats['inliers']
            self._vis_lc_coef     = lc_use
            self._vis_rc_coef     = rc_use
            self._vis_center_coef = None
            self._vis_raw_lc_coef = lc_stats['raw_coef']
            self._vis_raw_rc_coef = rc_stats['raw_coef']
            self._vis_left_candidates = left_candidates
            self._vis_right_candidates = right_candidates
            self._vis_left_rejected = left_rejected + lc_stats['rejected']
            self._vis_right_rejected = right_rejected + rc_stats['rejected']
            self._vis_left_status = lc_stats['status']
            self._vis_right_status = rc_stats['status']
            self._vis_left_residual = lc_stats['residual']
            self._vis_right_residual = rc_stats['residual']
            self._vis_center_mode = 'NONE'
            self._vis_reject_reason = (f'lc:{lc_stats["status"]}'
                                       f'/rc:{rc_stats["status"]}')
            self._vis_row_start = path_row_start
            return None, None

        # ── 제어용 center_x: 먼 상단 대신 가까운 target row 기준 ────────
        control_row = float(np.clip(
            H * self.control_row_ratio,
            path_row_start,
            H - 1,
        ))
        center_x = float(poly_eval(center_coef, control_row))

        # center_jump 크기 항상 기록 (reject 여부와 무관)
        if self._prev_center_x is not None:
            self._vis_center_jump_px = float(abs(center_x - self._prev_center_x))

        if (center_mode != 'BOTH'
                and self._prev_center_x is not None
                and self._vis_center_jump_px > self.single_lane_max_center_jump):
            self.get_logger().warn(
                f'single-lane center jump rejected: mode={center_mode}'
                f' prev={self._prev_center_x:.1f} new={center_x:.1f}'
                f' limit={self.single_lane_max_center_jump:.1f}',
                throttle_duration_sec=0.5,
            )
            self._vis_left_pts = lc_stats['inliers']
            self._vis_right_pts = rc_stats['inliers']
            self._vis_lc_coef = lc_use
            self._vis_rc_coef = rc_use
            self._vis_center_coef = None
            self._vis_raw_lc_coef = lc_stats['raw_coef']
            self._vis_raw_rc_coef = rc_stats['raw_coef']
            self._vis_left_status = lc_stats['status']
            self._vis_right_status = rc_stats['status']
            self._vis_center_mode = 'REJECT'
            self._vis_reject_reason = f'jump:{self._vis_center_jump_px:.0f}px'
            self._vis_row_start = path_row_start
            return None, None

        # 다음 프레임의 동적 분리 기준점 갱신
        self._prev_center_x = center_x
        # reject_reason: 성공 경로에서도 단일 차선 또는 폭 충돌 여부를 기록
        if 'WIDTH' in center_mode:
            self._vis_reject_reason = 'width_conflict'
        elif center_mode in ('LEFT', 'RIGHT'):
            self._vis_reject_reason = f'single_{center_mode.lower()}'
        else:
            self._vis_reject_reason = ''

        # ── 헤딩 오차(epsi): 데이터 구간 중앙에서 기울기 계산 ───────────
        mid_row = float((path_row_start + H) / 2.0)
        slope   = 2.0 * center_coef[0] * mid_row + center_coef[1]
        epsi    = math.atan(slope)

        # 디스플레이용 저장
        self._vis_left_pts    = lc_stats['inliers']
        self._vis_right_pts   = rc_stats['inliers']
        self._vis_lc_coef     = lc_use   # smooth 계수 표시
        self._vis_rc_coef     = rc_use   # smooth 계수 표시
        self._vis_center_coef = center_coef
        self._vis_raw_lc_coef = lc_stats['raw_coef']
        self._vis_raw_rc_coef = rc_stats['raw_coef']
        self._vis_left_candidates = left_candidates
        self._vis_right_candidates = right_candidates
        self._vis_left_rejected = left_rejected + lc_stats['rejected']
        self._vis_right_rejected = right_rejected + rc_stats['rejected']
        self._vis_left_status = lc_stats['status']
        self._vis_right_status = rc_stats['status']
        self._vis_left_residual = lc_stats['residual']
        self._vis_right_residual = rc_stats['residual']
        width_source = 'LEARNED' if self._half_width_learned else 'DEFAULT'
        self._vis_center_mode = f'{center_mode}/{width_source}'
        self._vis_row_start   = path_row_start   # 그리기/검증 범위 제한용

        return center_x, epsi

    # ------------------------------------------------------------------
    # 정규화 (ey + epsi 합산)
    # ------------------------------------------------------------------

    def _compute_offset(self, center_x: float | None, epsi: float | None) -> float | None:
        """center_x(px) + epsi(rad) → [-1.0, +1.0] lane_offset.

        ey_norm   : 횡 편차 (오른쪽 = +)
        epsi_norm : 헤딩 오차 (차선이 우측으로 기울 = +, 우회전 필요)
        출력      : ey_norm + epsi_weight × epsi_norm  (클리핑)
        """
        if center_x is None:
            return None
        ey_norm = float(np.clip(
            (center_x - self.IMG_W / 2.0) / (self.IMG_W / 2.0), -1.0, 1.0
        ))
        if epsi is None or self.epsi_weight == 0.0:
            return ey_norm
        epsi_norm = float(np.clip(epsi / (math.pi / 2.0), -1.0, 1.0))
        return float(np.clip(ey_norm + self.epsi_weight * epsi_norm, -1.0, 1.0))

    def _compute_path_x_span(self) -> float | None:
        """중심 경로가 scan 상단~하단에서 좌우로 얼마나 움직이는지 계산."""
        if self._vis_center_coef is None:
            self._vis_path_x_span = None
            return None
        coef = self._vis_center_coef
        row_start = float(getattr(self, '_vis_row_start', 0))
        row_end = float(self.IMG_H - 1)
        x_start = coef[0] * row_start ** 2 + coef[1] * row_start + coef[2]
        x_end = coef[0] * row_end ** 2 + coef[1] * row_end + coef[2]
        self._vis_path_x_span = float(abs(x_end - x_start))
        return self._vis_path_x_span

    def _evaluate_lane_valid(
        self,
        center_x: float | None,
        epsi: float | None,
        path_x_span: float | None,
    ) -> bool:
        """NN 검출과 별개로 제어에 사용할 수 있는 차선 경로인지 판단."""
        self._vis_lane_valid = False
        self._vis_lane_state = 'MISS'

        if center_x is None or self._vis_center_coef is None:
            if not self._vis_reject_reason:
                self._vis_reject_reason = 'no_center'
            return False

        mode = self._vis_center_mode or ''
        if mode.startswith('NONE') or mode.startswith('REJECT'):
            self._vis_lane_state = 'REJECT'
            if not self._vis_reject_reason:
                self._vis_reject_reason = 'mode'
            return False

        reject_reasons = []
        if epsi is not None:
            epsi_deg = abs(math.degrees(epsi))
            if epsi_deg > self.max_valid_epsi_deg:
                reject_reasons.append(f'epsi:{epsi_deg:.1f}')

        if path_x_span is not None and path_x_span > self.max_valid_path_span_px:
            reject_reasons.append(f'path_span:{path_x_span:.0f}')

        max_mean_run = max(self._vis_left_mean_run_w, self._vis_right_mean_run_w)
        if max_mean_run > self.max_run_width:
            reject_reasons.append(f'run_w:{max_mean_run:.0f}')

        is_single = mode.startswith('LEFT') or mode.startswith('RIGHT')
        if is_single:
            if path_x_span is None:
                reject_reasons.append('single_no_span')
            if (self._vis_left_row_span < self.min_row_span
                    and self._vis_right_row_span < self.min_row_span):
                reject_reasons.append('single_row_span')

        # IMU 교차 검증: epsi 기반 REJECT인데 IMU가 직진이라면 억제
        # _imu_last_t is None 이면 IMU 미수신 상태 → 초기값(0.0)으로 오판 방지
        self._imu_epsi_suppressed = False
        if reject_reasons and self.use_imu_heading and self._imu_last_t is not None:
            imu_abs_deg = abs(math.degrees(self._imu_delta_yaw))
            if imu_abs_deg < self.imu_epsi_suppress_deg:
                before = len(reject_reasons)
                reject_reasons = [r for r in reject_reasons
                                  if not r.startswith('epsi:')]
                if len(reject_reasons) < before:
                    self._imu_epsi_suppressed = True

        if reject_reasons:
            self._vis_lane_state = 'REJECT'
            reason = ';'.join(reject_reasons)
            self._vis_reject_reason = (
                f'{self._vis_reject_reason};{reason}'
                if self._vis_reject_reason else reason
            )
            return False

        self._vis_lane_valid = True
        self._vis_lane_state = 'VALID'
        return True

    # ------------------------------------------------------------------
    # 곡률 계산
    # ------------------------------------------------------------------

    def _compute_curvature_m(self, center_coef, row: float) -> float:
        """BEV 다항식에서 물리적 곡률반경(m)을 반환한다.

        poly_degree=1이면 무한대(1e6)를 반환(직선).
        xm_per_pix=0이면 픽셀 단위 반경을 반환.
        """
        if center_coef is None:
            return 1e6
        a = float(center_coef[0])
        b = float(center_coef[1])
        dxdr   = 2.0 * a * row + b     # 1차 미분
        d2xdr2 = 2.0 * a               # 2차 미분
        kappa_px = abs(d2xdr2) / (1.0 + dxdr ** 2) ** 1.5
        if kappa_px < 1e-9:
            return 1e6
        R_px = 1.0 / kappa_px
        scale = self._xm_per_pix if self._xm_per_pix > 0 else 1.0
        return R_px * scale

    # ------------------------------------------------------------------
    # 경로 좌표 CSV 저장
    # ------------------------------------------------------------------

    def _write_path_csv(self, ts_sec: float, center_x, curv_r: float,
                        epsi, lane_valid: bool) -> None:
        if (self._path_writer is None
                or self._vis_center_coef is None
                or not lane_valid):
            self._path_frame_no += 1
            return

        coef = self._vis_center_coef
        row_start = getattr(self, '_vis_row_start', 0)
        wp_rows = np.arange(row_start, self.IMG_H, 15, dtype=float)
        heading_deg = math.degrees(epsi) if epsi is not None else 0.0

        for idx, r in enumerate(wp_rows):
            x_px = coef[0] * r ** 2 + coef[1] * r + coef[2]
            dx_m = ((x_px - self.IMG_W / 2.0) * self._xm_per_pix
                    if self._xm_per_pix > 0 else None)
            self._path_writer.writerow([
                round(ts_sec, 4),
                self._path_frame_no,
                idx,
                round(float(r), 1),
                round(float(x_px), 1),
                round(float(dx_m), 5) if dx_m is not None else '',
                round(float(min(curv_r, 1e6)), 2),
                round(heading_deg, 2),
            ])

        self._path_frame_no += 1
        if self._path_frame_no % 30 == 0:
            self._path_f.flush()

    # ------------------------------------------------------------------
    # 경로 오버레이 헬퍼 (debug 이미지와 show_window 공용)
    # ------------------------------------------------------------------

    def _draw_path_overlay(self, vis: np.ndarray) -> None:
        """center_coef 기반 민트 경로선 + 노란 웨이포인트 + 헤딩 화살표를 vis에 그린다."""
        if self._vis_center_coef is None:
            return
        H, W = vis.shape[:2]
        coef = self._vis_center_coef
        row_start = getattr(self, '_vis_row_start', 0)
        rows_draw = np.arange(row_start, H, dtype=float)

        # ── 민트 경로선 ────────────────────────────────────────────────
        xs = coef[0] * rows_draw ** 2 + coef[1] * rows_draw + coef[2]
        pts = [(int(round(float(x))), int(r))
               for r, x in zip(rows_draw, xs)
               if 0 <= int(round(float(x))) < W]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], (0, 220, 160), 2)

        # ── 노란 웨이포인트 (매 15행) ──────────────────────────────────
        wp_rows = np.arange(row_start, H, 15, dtype=float)
        wp_xs = coef[0] * wp_rows ** 2 + coef[1] * wp_rows + coef[2]
        for r, x in zip(wp_rows, wp_xs):
            px = int(round(float(x)))
            if 0 <= px < W:
                cv2.circle(vis, (px, int(r)), 3, (0, 255, 255), -1)

        # ── 헤딩 화살표 ────────────────────────────────────────────────
        bot_row = float(H - 1)
        bot_x   = coef[0] * bot_row ** 2 + coef[1] * bot_row + coef[2]
        slope   = 2.0 * coef[0] * bot_row + coef[1]
        mag     = math.sqrt(slope ** 2 + 1.0)
        arrow   = 40
        tip_x   = int(bot_x + arrow * (-slope / mag))   # 전방(위)으로
        tip_y   = int(bot_row + arrow * (-1.0 / mag))
        bx, by  = int(bot_x), int(bot_row)
        if 0 <= bx < W and 0 <= tip_x < W and 0 <= tip_y < H:
            cv2.arrowedLine(vis, (bx, by), (tip_x, tip_y),
                            (0, 200, 255), 2, tipLength=0.35)

        # ── 곡률반경 HUD ───────────────────────────────────────────────
        cr = self._last_curvature_r
        if cr is not None:
            cr_str = f'R={cr:.1f}m' if cr < 999 else 'R=∞'
            cv2.putText(vis, cr_str, (W - 90, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 160), 1)

    # ------------------------------------------------------------------
    # /lane_debug 이미지 발행 (rqt_image_view로 확인)
    # ------------------------------------------------------------------

    def _publish_debug(self, binary: np.ndarray | None, center_x: float | None) -> None:
        """바이너리 마스크 + 다항식 곡선 + 중심선을 이미지로 발행."""
        H, W = self.IMG_H, self.IMG_W

        # 배경: 바이너리 마스크를 흑백으로
        if binary is not None:
            vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        else:
            vis = np.zeros((H, W, 3), dtype=np.uint8)

        # 피팅 범위만 그림 — 외삽 구간(상단)에 잘못된 곡선이 표시되지 않도록
        row_draw_start = getattr(self, '_vis_row_start', 0)
        rows = np.arange(row_draw_start, H)

        def draw_poly(coef, color, radius=2):
            if coef is None:
                return
            xs = coef[0] * rows**2 + coef[1] * rows + coef[2]
            for r, x in zip(rows, xs):
                px = int(round(x))
                if 0 <= px < W:
                    cv2.circle(vis, (px, int(r)), radius, color, -1)

        # raw 피팅은 얇게, EMA 피팅은 굵게 표시
        draw_poly(self._vis_raw_lc_coef, (140, 70, 0), 1)
        draw_poly(self._vis_raw_rc_coef, (0, 80, 140), 1)
        draw_poly(self._vis_lc_coef, (255, 100, 0))       # 파란색: 왼쪽
        draw_poly(self._vis_rc_coef, (0, 140, 255))       # 주황색: 오른쪽
        draw_poly(self._vis_center_coef, (0, 255, 255))   # 노란색: 차량 추종 경로

        # 모든 run 후보는 회색, 제거된 후보는 자홍색, inlier는 좌/우 색으로 표시
        for r, x in self._vis_left_candidates + self._vis_right_candidates:
            cv2.circle(vis, (int(round(x)), int(r)), 1, (100, 100, 100), -1)
        for r, x in self._vis_left_rejected + self._vis_right_rejected:
            cv2.circle(vis, (int(round(x)), int(r)), 2, (255, 0, 255), -1)
        for r, x in self._vis_left_pts:
            cv2.circle(vis, (int(round(x)), int(r)), 2, (255, 200, 100), -1)
        for r, x in self._vis_right_pts:
            cv2.circle(vis, (int(round(x)), int(r)), 2, (100, 200, 255), -1)

        # 실제 피팅에 사용하는 scan 시작 행
        cv2.line(vis, (0, row_draw_start), (W - 1, row_draw_start), (0, 255, 0), 1)

        # 이미지 중앙선 (빨간)
        cv2.line(vis, (W // 2, 0), (W // 2, H - 1), (0, 0, 255), 1)

        # raw 중심 (회색)
        if center_x is not None:
            cv2.line(vis, (int(center_x), 0), (int(center_x), H - 1), (160, 160, 160), 1)

        # smooth 중심 (초록)
        if self._smooth_offset is not None:
            scx = int((self._smooth_offset + 1.0) * (W / 2.0))
            cv2.line(vis, (scx, 0), (scx, H - 1), (0, 255, 0), 2)

        # HUD
        epsi_val = getattr(self, '_last_epsi', None)
        epsi_str = f'{math.degrees(epsi_val):.1f}deg' if epsi_val is not None else '-'
        off_str  = f'{self._smooth_offset:.3f}' if self._smooth_offset is not None else '-'
        cv2.putText(vis, f'off:{off_str}  epsi:{epsi_str}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        l_res = '-' if self._vis_left_residual is None else f'{self._vis_left_residual:.1f}'
        r_res = '-' if self._vis_right_residual is None else f'{self._vis_right_residual:.1f}'
        cv2.putText(vis, f'L:{len(self._vis_left_pts)} {self._vis_left_status} e:{l_res}',
                    (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f'R:{len(self._vis_right_pts)} {self._vis_right_status} e:{r_res}',
                    (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f'CENTER:{self._vis_center_mode}', (5, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f'LANE:{self._vis_lane_state} {self._vis_reject_reason}',
                    (5, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0) if self._vis_lane_valid else (0, 165, 255), 1)

        # 민트 경로 + 웨이포인트 + 헤딩 화살표 오버레이
        self._draw_path_overlay(vis)

        msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_pub.publish(msg)

    # ------------------------------------------------------------------
    # 디버그 표시 (show_window=True 전용)
    # ------------------------------------------------------------------

    def _update_display(self, binary: np.ndarray | None, cx: float | None) -> None:
        # _process_cb에서 이미 소비·보관한 최신 프레임 사용
        rgb_frame = self._last_rgb_frame
        source_preview = None

        if rgb_frame is not None:
            if self.use_bev:
                source_preview = rgb_frame.copy()
                polygon = np.round(self._bev_src_points).astype(np.int32)
                cv2.polylines(
                    source_preview, [polygon], True, (0, 255, 255), 2
                )
                cv2.putText(
                    source_preview, 'SOURCE', (570, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
                )
            vis = self._to_bev(rgb_frame, cv2.INTER_LINEAR)
            # 차선 마스크를 초록 반투명으로 오버레이
            if binary is not None:
                overlay = np.zeros_like(vis)
                overlay[binary > 0] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        elif binary is not None:
            # 컬러 프레임 없으면 흑백 마스크로 폴백
            vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        else:
            return

        # 이미지 중앙선 (빨간)
        cv2.line(vis, (self.IMG_W // 2, 0), (self.IMG_W // 2, self.IMG_H - 1),
                 (0, 0, 255), 1)

        # 민트 경로 + 웨이포인트 + 헤딩 화살표
        self._draw_path_overlay(vis)

        # raw center_x 위치 (회색 가로 마커) — 가중 평균 결과를 하단에 점으로 표시
        if cx is not None:
            cv2.circle(vis, (int(cx), self.IMG_H - 8), 5, (160, 160, 160), -1)

        # NN bbox는 원본 좌표이므로 BEV 화면에는 그리지 않는다.
        if not self.use_bev:
            for cx_a, cy_a, w_a, h_a, conf_a in getattr(self, '_vis_anchors', []):
                x1 = max(0, int(cx_a - w_a / 2))
                y1 = max(0, int(cy_a - h_a / 2))
                x2 = min(self.IMG_W - 1, int(cx_a + w_a / 2))
                y2 = min(self.IMG_H - 1, int(cy_a + h_a / 2))
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(vis, f'{conf_a:.2f}', (x1, max(y1 - 2, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # ── HUD: 최대 신뢰도 / 통과 앵커 수 / 평활 오프셋 ──────────────
        max_conf = getattr(self, '_vis_max_conf', 0.0)
        n_anchors = len(getattr(self, '_vis_anchors', []))
        cv2.putText(vis, f'max:{max_conf:.2f}  n:{n_anchors}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(vis, 'BEV' if self.use_bev else 'CAM', (570, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if self._smooth_offset is not None:
            state_color = (0, 255, 0) if self._vis_lane_valid else (0, 165, 255)
            state_label = self._vis_lane_state
            cv2.putText(vis, f'off:{self._smooth_offset:.3f}  {state_label}', (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1)
        else:
            cv2.putText(vis, f'LANE:{self._vis_lane_state}', (5, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        # epsi + 곡률
        epsi_val = getattr(self, '_last_epsi', None)
        if epsi_val is not None:
            cv2.putText(vis, f'epsi:{math.degrees(epsi_val):.1f}deg', (5, 49),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cr = self._last_curvature_r
        if cr is not None:
            cr_str = f'R={cr:.1f}m' if cr < 999 else 'R=∞(직선)'
            cv2.putText(vis, cr_str, (5, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 160), 1)

        display = (
            np.hstack((source_preview, vis))
            if source_preview is not None else vis
        )
        cv2.imshow('lane_detector', display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC → 노드 종료
            self.get_logger().info('ESC 입력 — 노드 종료')
            rclpy.shutdown()
        elif key == ord('s'):   # 's' → raw 이미지 저장
            if rgb_frame is not None:
                ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                path = os.path.join(
                    self.capture_dir, f'lane_{ts}_{self._capture_count:04d}.png'
                )
                cv2.imwrite(path, rgb_frame)
                self._capture_count += 1
                self.get_logger().info(f'[캡처] {path}')

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        if hasattr(self, '_diag'):
            self._diag.close()
        if self._path_f is not None:
            self._path_f.flush()
            self._path_f.close()
            self._path_f = None
        if hasattr(self, 'pipeline'):
            self.pipeline = None
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
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
