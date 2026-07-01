"""차선 인식 진단 로거.

사용법:
    from camera_stack.diag_logger import DiagLogger
    self._diag = DiagLogger(enabled=self.log_diag, save_dir=self.capture_dir,
                            xm_per_pix=self.xm_per_pix)

    # 매 프레임 write_row() 호출 (None 전달 가능, 빈 칸으로 기록됨)
    self._diag.write_row(...)

파일은 capture_dir/lane_diag_YYYYMMDD_HHMMSS.csv 로 저장됨.
"""

import csv
import datetime
import math
import os
import threading
from collections import deque
from typing import Optional

import numpy as np

# CSV 컬럼 순서 (분석 시 읽기 편한 순서로 배치)
_HEADER = [
    # ── 기본 ────────────────────────────────────────────────────────────
    'ts_sec',           # 프레임 ROS 타임스탬프(float)
    'frame_no',         # 프레임 번호(0부터 단조증가)
    'fps',              # 최근 10프레임 평균 처리속도(Hz)
    # ── 검출 결과 ────────────────────────────────────────────────────────
    'center_mode',      # BOTH / LEFT / RIGHT / LEFT/WIDTH / RIGHT/WIDTH / NONE / REJECT
    'lc_status',        # ACCEPT / MISS / SINGLE / WIDTH / REJECT
    'rc_status',
    'lc_pts',           # 왼쪽 차선 inlier 점 수
    'rc_pts',           # 오른쪽 차선 inlier 점 수
    # ── 오프셋 ──────────────────────────────────────────────────────────
    'center_x_px',      # BEV 내 중심 x 좌표(px)
    'lane_offset_norm', # 기존 /lane_offset 정규화값 [-1,1]
    'lane_offset_m',    # 신규 /lane_offset_m (meter, + = 차선중심이 우측)
    'lane_valid',       # 신규 /lane_valid (1=신뢰, 0=무효)
    # ── 헤딩 (offset에 섞지 않지만 진단용) ─────────────────────────────
    'epsi_deg',         # 헤딩 오차(도). 크면 차선이 비스듬히 보임
    # ── 다항식 계수 (poly: a·row²+b·row+c = x) ──────────────────────────
    'lc_a', 'lc_b', 'lc_c',      # 왼쪽 차선 (smooth)
    'rc_a', 'rc_b', 'rc_c',      # 오른쪽 차선 (smooth)
    'center_a', 'center_b', 'center_c',   # 중심 경로
    # ── 차선 폭 ─────────────────────────────────────────────────────────
    'half_width_px',    # 학습된 반폭 (px, ROI 중간 행 기준)
    'half_width_m',     # 반폭 meter 환산
    # ── 피팅 품질 ────────────────────────────────────────────────────────
    'lc_residual_px',   # 왼쪽 피팅 잔차(px)
    'rc_residual_px',   # 오른쪽 피팅 잔차(px)
    # ── 이상 여부 ────────────────────────────────────────────────────────
    'center_jumped',    # 1 = 프레임 간 jump rejection 발생
    # ── 보조 ────────────────────────────────────────────────────────────
    'row_start',        # ROI 시작 행(px)
    'bev_enabled',      # BEV 사용 여부(1/0)
    # ── 1순위 신규 ──────────────────────────────────────────────────────
    'reject_reason',        # NONE/REJECT 원인 문자열
    'left_total_pts',       # 피팅 전 추적 포인트 총 수 (왼쪽)
    'right_total_pts',      # 피팅 전 추적 포인트 총 수 (오른쪽)
    'left_row_span_px',     # inlier 행 범위 (row_max - row_min, 왼쪽)
    'right_row_span_px',    # inlier 행 범위 (오른쪽)
    'left_mean_run_w',      # 선택된 run 평균 픽셀 폭 (왼쪽)
    'right_mean_run_w',     # 선택된 run 평균 픽셀 폭 (오른쪽)
    'path_x_span_px',       # center poly 상단~하단 x 이동량(px)
    'curvature_r_m',        # 곡률반경(m). poly_degree=1이면 1e6
    # ── 2순위 신규 ──────────────────────────────────────────────────────
    'raw_mask_px',          # blob 필터 전 마스크 픽셀 수 (ROI cut 후)
    'bev_mask_px',          # BEV 변환 후 마스크 픽셀 수
    'filtered_mask_px',     # blob 필터 후 마스크 픽셀 수
    'blob_removed_px',      # blob 필터로 제거된 픽셀 수
    # ── 3순위 신규 ──────────────────────────────────────────────────────
    'lane_width_top_px',    # BOTH 모드 시 scan 상단 차선 폭(px)
    'lane_width_mid_px',    # BOTH 모드 시 중단 차선 폭(px)
    'lane_width_bot_px',    # BOTH 모드 시 하단 차선 폭(px)
    'center_jump_px',       # 이전 프레임 대비 center_x 변화량(px)
    # ── IMU ─────────────────────────────────────────────────────────────
    'imu_fresh',            # IMU 수신 여부 (1=수신중, 0=미수신)
    'imu_angular_z_rads',   # 최신 yaw rate (rad/s, CCW+)
    'imu_delta_yaw_deg',    # 마지막 유효 감지 이후 누적 회전량 (deg)
    'imu_epsi_suppressed',  # IMU가 epsi reject 억제 여부 (1=억제됨, 0=억제안됨)
    # ── NN 신뢰도 (no_mask 원인 분석) ────────────────────────────────────
    'nn_max_conf',    # 프레임 최대 NN 신뢰도 (0=NN결과없음, <conf_thresh=점수미달)
    'nn_pass_count',  # conf_thresh 통과 앵커 수 (0이면 마스크 생성 실패)
    # ── 차선 인식 상태 요약 ───────────────────────────────────────────────
    'lane_count',     # 2=양쪽 검출, 1=한쪽 검출, 0=미검출 (center_mode에서 자동 계산)
]


def _coef3(c) -> tuple:
    """다항식 계수 배열 → (a, b, c). 없으면 ('', '', '')."""
    if c is None:
        return ('', '', '')
    arr = np.asarray(c, dtype=float).flatten()
    if len(arr) == 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    if len(arr) == 2:       # 1차 피팅 결과 → a=0
        return (0.0, float(arr[0]), float(arr[1]))
    return ('', '', '')


def _eval_poly(coef, row: float) -> Optional[float]:
    """poly 계수 배열로 x = a*row^2 + b*row + c 계산. None이면 None."""
    if coef is None:
        return None
    a, b, c = _coef3(coef)
    if a == '':
        return None
    return float(a) * row ** 2 + float(b) * row + float(c)


class DiagLogger:
    """스레드 안전 CSV 진단 로거.

    Parameters
    ----------
    enabled    : True면 기록, False면 write_row가 no-op.
    save_dir   : CSV 저장 폴더 경로.
    xm_per_pix : BEV 픽셀 1개당 meter(좌우방향). 캘리브레이션 후 설정.
                 0이면 lane_offset_m / half_width_m 비워둠.
    """

    def __init__(self, enabled: bool, save_dir: str, xm_per_pix: float = 0.0):
        self.enabled = enabled
        self.xm_per_pix = float(xm_per_pix)
        self._lock = threading.Lock()
        self._frame_no = 0
        self._last_ts = 0.0
        self._fps_buf: deque = deque(maxlen=10)
        self._path = ''

        self._f = None
        self._writer = None

        if not enabled:
            return

        os.makedirs(save_dir, exist_ok=True)
        ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(save_dir, f'lane_diag_{ts_str}.csv')
        self._f = open(path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._f, fieldnames=_HEADER,
                                      extrasaction='ignore')
        self._writer.writeheader()
        self._f.flush()
        self._path = path
        print(f'[DiagLogger] 저장 시작: {path}')

    # ------------------------------------------------------------------

    def write_row(
        self,
        ts_sec: float,
        center_mode: str,
        lc_status: str,
        rc_status: str,
        lc_pts: int,
        rc_pts: int,
        center_x_px: Optional[float],
        lane_offset_norm: Optional[float],
        lane_offset_m: Optional[float],
        lane_valid: bool,
        epsi_rad: Optional[float],
        lc_coef,                     # np.ndarray or None
        rc_coef,
        center_coef,
        half_width_coef,             # 반폭 다항식 or None
        lc_residual: Optional[float],
        rc_residual: Optional[float],
        center_jumped: bool,
        row_start: int,
        bev_enabled: bool,
        # ── 1순위 신규 파라미터 ─────────────────────────────────────────
        reject_reason: str = '',
        left_total_pts: int = 0,
        right_total_pts: int = 0,
        left_row_span_px: float = 0.0,
        right_row_span_px: float = 0.0,
        left_mean_run_w: float = 0.0,
        right_mean_run_w: float = 0.0,
        path_x_span_px: Optional[float] = None,
        curvature_r_m: Optional[float] = None,
        # ── 2순위 신규 파라미터 ─────────────────────────────────────────
        raw_mask_px: int = 0,
        bev_mask_px: int = 0,
        filtered_mask_px: int = 0,
        # ── 3순위 신규 파라미터 ─────────────────────────────────────────
        lane_width_top_px: Optional[float] = None,
        lane_width_mid_px: Optional[float] = None,
        lane_width_bot_px: Optional[float] = None,
        center_jump_px: float = 0.0,
        # ── IMU 파라미터 ────────────────────────────────────────────────
        imu_fresh: bool = False,
        imu_angular_z_rads: Optional[float] = None,
        imu_delta_yaw_deg: Optional[float] = None,
        imu_epsi_suppressed: bool = False,
        # ── NN 신뢰도 파라미터 ───────────────────────────────────────────
        nn_max_conf: float = 0.0,
        nn_pass_count: int = 0,
    ) -> None:
        if not self.enabled or self._writer is None:
            return

        # FPS 계산
        if self._last_ts > 0:
            dt = ts_sec - self._last_ts
            if 0 < dt < 5.0:
                self._fps_buf.append(1.0 / dt)
        self._last_ts = ts_sec
        fps = float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

        # 반폭 픽셀 (ROI 중간 행에서 평가)
        mid_row = (row_start + 224) / 2.0
        hw_px_val = _eval_poly(half_width_coef, mid_row)
        hw_px = round(hw_px_val, 1) if hw_px_val is not None else ''
        hw_m  = (round(hw_px_val * self.xm_per_pix, 4)
                 if (hw_px_val is not None and self.xm_per_pix > 0) else '')

        lca, lcb, lcc = _coef3(lc_coef)
        rca, rcb, rcc = _coef3(rc_coef)
        ca,  cb,  cc  = _coef3(center_coef)

        def r1(v, n=1):
            return round(float(v), n) if v != '' else ''

        def _opt(v, decimals=2):
            return round(float(v), decimals) if v is not None else ''

        # lane_count: center_mode 문자열에서 자동 파생
        # BOTH/... → 2, LEFT/... or RIGHT/... → 1, NONE/REJECT → 0
        if 'BOTH' in center_mode:
            _lane_count = 2
        elif 'LEFT' in center_mode or 'RIGHT' in center_mode:
            _lane_count = 1
        else:
            _lane_count = 0

        row = {
            'ts_sec':           round(ts_sec, 4),
            'frame_no':         self._frame_no,
            'fps':              round(fps, 1),
            'center_mode':      center_mode,
            'lc_status':        lc_status,
            'rc_status':        rc_status,
            'lc_pts':           lc_pts,
            'rc_pts':           rc_pts,
            'center_x_px':      round(center_x_px, 1) if center_x_px is not None else '',
            'lane_offset_norm': round(lane_offset_norm, 4) if lane_offset_norm is not None else '',
            'lane_offset_m':    round(lane_offset_m, 4) if lane_offset_m is not None else '',
            'lane_valid':       int(lane_valid),
            'epsi_deg':         round(math.degrees(epsi_rad), 2) if epsi_rad is not None else '',
            'lc_a': r1(lca, 6), 'lc_b': r1(lcb, 4), 'lc_c': r1(lcc, 2),
            'rc_a': r1(rca, 6), 'rc_b': r1(rcb, 4), 'rc_c': r1(rcc, 2),
            'center_a': r1(ca, 6), 'center_b': r1(cb, 4), 'center_c': r1(cc, 2),
            'half_width_px':    hw_px,
            'half_width_m':     hw_m,
            'lc_residual_px':   round(lc_residual, 2) if lc_residual is not None else '',
            'rc_residual_px':   round(rc_residual, 2) if rc_residual is not None else '',
            'center_jumped':    int(center_jumped),
            'row_start':        row_start,
            'bev_enabled':      int(bev_enabled),
            # 1순위
            'reject_reason':     reject_reason,
            'left_total_pts':    left_total_pts,
            'right_total_pts':   right_total_pts,
            'left_row_span_px':  _opt(left_row_span_px, 1),
            'right_row_span_px': _opt(right_row_span_px, 1),
            'left_mean_run_w':   _opt(left_mean_run_w, 2),
            'right_mean_run_w':  _opt(right_mean_run_w, 2),
            'path_x_span_px':    _opt(path_x_span_px, 1),
            'curvature_r_m':     _opt(curvature_r_m, 2),
            # 2순위
            'raw_mask_px':       raw_mask_px,
            'bev_mask_px':       bev_mask_px,
            'filtered_mask_px':  filtered_mask_px,
            'blob_removed_px':   max(0, raw_mask_px - filtered_mask_px),
            # 3순위
            'lane_width_top_px': _opt(lane_width_top_px, 1),
            'lane_width_mid_px': _opt(lane_width_mid_px, 1),
            'lane_width_bot_px': _opt(lane_width_bot_px, 1),
            'center_jump_px':    round(center_jump_px, 1),
            # IMU
            'imu_fresh':           int(imu_fresh),
            'imu_angular_z_rads':  _opt(imu_angular_z_rads, 4),
            'imu_delta_yaw_deg':   _opt(imu_delta_yaw_deg, 3),
            'imu_epsi_suppressed': int(imu_epsi_suppressed),
            # NN 신뢰도
            'nn_max_conf':   round(float(nn_max_conf), 4),
            'nn_pass_count': int(nn_pass_count),
            # 차선 인식 상태
            'lane_count':    _lane_count,
        }

        with self._lock:
            self._writer.writerow(row)
            self._frame_no += 1
            if self._frame_no % 50 == 0:   # 50프레임마다 flush
                self._f.flush()

    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        if self._f is not None:
            with self._lock:
                self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            with self._lock:
                self._f.flush()
                self._f.close()
                self._f = None
            print(f'[DiagLogger] 저장 완료: {self._path}')

    def __del__(self):
        self.close()
