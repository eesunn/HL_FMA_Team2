# 카메라 스택 종합 학습 가이드

`camera_stack` 패키지의 전체 구조, 각 노드의 동작 원리, 파라미터 의미,
데이터 흐름을 학습 목적으로 정리한 문서입니다.

> **최종 갱신**: 2026-06-20 (신모델 2022.1 적용, 파라미터 전체 재조정)

---

## 목차

1. [전체 구조 개요](#1-전체-구조-개요)
2. [하드웨어 — OAK-D Pro와 depthai](#2-하드웨어--oak-d-pro와-depthai)
3. [YOLOv8-seg 모델 이해](#3-yolov8-seg-모델-이해)
4. [lane_detector_node 상세](#4-lane_detector_node-상세)
5. [lane_recovery_node 상세](#5-lane_recovery_node-상세)
6. [traffic_light_detector_node 상세](#6-traffic_light_detector_node-상세)
7. [DiagLogger — 진단 CSV 로거](#7-diaglogger--진단-csv-로거)
8. [Blob 변환 방법 (pt → blob)](#8-blob-변환-방법-pt--blob)
9. [ROS2 토픽 전체 정리](#9-ros2-토픽-전체-정리)
10. [파라미터 튜닝 가이드](#10-파라미터-튜닝-가이드)
11. [알려진 한계와 미래 과제](#11-알려진-한계와-미래-과제)

---

## 1. 전체 구조 개요

### 패키지 파일 구조

```
camera_stack/
├── camera_stack/                  ← Python 패키지 (노드 소스)
│   ├── lane_detector_node.py      ← 차선 인식 (핵심)
│   ├── lane_recovery_node.py      ← 차선 추종 제어
│   ├── traffic_light_detector_node.py  ← 신호등 인식
│   ├── diag_logger.py             ← CSV 진단 로거
│   └── lane_detector_bev_node.py  ← BEV 전용 실험 노드 (미사용)
├── models/
│   ├── best/
│   │   └── lane_seg_640x224/
│   │       └── best_openvino_2022.1_6shave1.blob  ← 현재 기본 모델
│   ├── lane_seg_640x224/          ← 구모델 (백업)
│   └── traffic_light_640x224*/    ← 신호등 모델
├── tools/
│   └── lane_seg_blob_check.py     ← OAK-D 직접 연결 모델 확인 도구
└── CAMERA_SYSTEM_GUIDE.md         ← 이 파일
```

### 전체 데이터 흐름

```
OAK-D (IMX219, 640×224, ~27 fps)
     │
     ├─ [MyriadX NPU] best_openvino_2022.1_6shave1.blob 추론
     │        │
     │    lane_detector_node
     │        ├── /lane_offset   (Float32, [-1,+1], TRANSIENT_LOCAL QoS)
     │        ├── /lane_valid    (Bool, TRANSIENT_LOCAL QoS)
     │        ├── /lane_debug    (Image)
     │        └── lane_diag_*.csv  → ~/capstone_ws/logs/
     │
     ├─ [MyriadX NPU] traffic_light blob 또는 HSV
     │        │
     │    traffic_light_detector_node
     │        ├── /traffic_light       (String: RED/GREEN/NONE)
     │        └── /cmd_vel_traffic     (Twist: 현재 미연결)
     │
lane_recovery_node
     ├── 구독: /lane_offset, /lane_valid, /lane_recovery_enable
     ├── 발행: /cmd_vel_recovery (또는 output_topic 파라미터)
     └── lane_recovery_*.csv  → ~/capstone_ws/logs/
          │
     mecanum_bridge_node  (control_stack)
          │  /wheel_targets [A,B,C,D] km/h
     can_bridge_node      (control_stack)
          │  CAN 0x300 SpeedCommand, 0x301 ControlMode=3
     TC275 → 4채널 PI 속도 제어
```

---

## 2. 하드웨어 — OAK-D Pro와 depthai

### OAK-D Pro 구성

| 구성 요소 | 사양 |
|---|---|
| RGB 카메라 | IMX219-120 (CSI), 광각 120° |
| 사용 해상도 | 640×224 @ ~27fps |
| NPU | MyriadX (Intel) — 최대 4 TOPS |
| 연결 | USB 3.0 |
| 추론 형식 | `.blob` (OpenVINO IR, FP16) |

### depthai 3.x API 핵심 패턴

```python
pipeline = dai.Pipeline(dai.Device())
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cam_out = cam.requestOutput((640, 224), type=dai.ImgFrame.Type.BGR888p, fps=25)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath('/path/to/model.blob')
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
cam_out.link(nn.input)

rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
nn_queue  = nn.out.createOutputQueue(maxSize=4, blocking=False)
pipeline.start()
```

> depthai 3.x는 큐를 `pipeline.start()` **전에** 생성해야 합니다 (2.x와 차이).

---

## 3. YOLOv8-seg 모델 이해

### 현재 사용 모델

| 항목 | 값 |
|---|---|
| 파일명 | `best_openvino_2022.1_6shave1.blob` |
| 아키텍처 | YOLOv8n-seg (nano) |
| 입력 크기 | 640×224 (W×H) |
| 클래스 | class 0 = 차선 영역 |
| 출력 텐서 1 | shape `(1, 37, 2940)` — 앵커 예측값 |
| 출력 텐서 2 | shape `(1, 32, 56, 160)` — 마스크 프로토타입 |
| conf_thresh 적정값 | **0.45** (2022.1 모델은 구모델보다 score 분포가 낮음) |

### 출력 텐서 해석

```
텐서 0: (37, 2940) — 앵커별 예측
  ├── [0:4]  : bounding box (cx, cy, w, h)
  ├── [4]    : objectness score
  └── [5:37] : mask coefficients (32개)

텐서 1: (32, 56, 160) — 프로토타입 마스크
```

### 디코딩 과정

```python
scores = preds[:, 4]
keep   = scores > conf_thresh     # 0.45 이상만 통과
mask_coeffs = preds[keep, 5:]
masks_raw   = mask_coeffs @ protos.reshape(32, -1)
masks       = sigmoid(masks_raw).reshape(-1, 56, 160)
combined    = masks.max(axis=0)
binary      = (cv2.resize(combined, (640, 224)) > mask_thresh).astype(np.uint8)
```

### 모델별 conf 특성 (로그 기반)

| 모델 | nn_max_conf 평균 (no_mask 시) | 적정 conf_thresh |
|---|---|---|
| 구모델 (lane_seg_640x224) | ~0.60 이상 | 0.60 |
| **현재 (2022.1)** | **~0.34~0.46** | **0.45** |

> **no_mask 진단**: `nn_max_conf < conf_thresh`이면 모델 점수 미달, `nn_pass_count > 0`이면 마스크 생성 단계 문제

---

## 4. lane_detector_node 상세

### 처리 파이프라인 (한 프레임 기준)

```
OAK-D 카메라 프레임 (640×224 BGR)
       │
       ▼
[1] YOLOv8-seg 추론 (MyriadX)
  conf_thresh=0.45 로 앵커 필터링
  nn_max_conf / nn_pass_count 기록
  → binary mask (224×640, 0/1)
       │
       ▼
[2] ROI 크롭 (roi_start_ratio=0.55)
  상단 55% 제거
       │
       ▼
[3] Blob 형태 필터
  크기 / 면적 / 가로세로비 필터
       │
       ▼
[4] BEV 변환 (use_bev=True)
  사다리꼴 → 직사각형 원근 변환
       │
       ▼
[5] 차선 탐색 및 피팅 (_lane_center_poly)
  행별 run 검출 → polyfit(deg=1)
  half_width EMA 학습 (|epsi| ≤ 30° 조건)
       │
       ▼
[6] 중심 복원 (single-lane 시)
  BOTH: (lc_x + rc_x) / 2
  LEFT/RIGHT: 한쪽 + learned_half_width
  center_jump > 150px → REJECT
       │
       ▼
[7] 유효성 검사
  epsi > max_valid_epsi_deg(100°) → REJECT
  path_x_span > max_valid_path_span_px(320px) → REJECT
       │
       ▼
[8] 오프셋 계산 + epsi 혼합
  raw_offset = (center_x - BEV_center) / (IMG_W/2)
  offset = raw_offset + epsi_weight × epsi_norm  (epsi_weight=0.3)
  → /lane_offset (TRANSIENT_LOCAL QoS)
  → /lane_valid  (TRANSIENT_LOCAL QoS)
```

### BEV 변환 원리

```
원본 (사다리꼴)              BEV (직사각형)
  top_left ── top_right      left ──────── right
     /                \       │              │
    /                  \      │  (평행 보임) │
 bot_left ──────── bot_right  left ──────── right
```

파라미터 (기본값 기준, 640×224):
```
top_left  = (0.320×640, 0.620×224) = (204.8, 138.9)
top_right = (0.680×640, 0.620×224) = (435.2, 138.9)
bot_left  = (0.050×640, 224      ) = ( 32.0, 224  )
bot_right = (0.950×640, 224      ) = (608.0, 224  )
BEV 유효 구간: x=[160, 480]px
```

### 전체 파라미터 정리

**모델/추론**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `blob_path` | (자동) | share/camera_stack/models/best/lane_seg_640x224/best_openvino_2022.1_6shave1.blob |
| `conf_thresh` | **0.45** | 앵커 신뢰도 임계값 (2022.1 모델에 맞게 조정) |
| `mask_thresh` | 0.45 | 마스크 sigmoid 이진화 임계값 |

**ROI / BEV**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `roi_start_ratio` | 0.55 | 이미지 상단 무시 비율 |
| `use_bev` | True | BEV 변환 사용 여부 |
| `bev_src_top_y_ratio` | 0.620 | 소스 사다리꼴 상단 y |
| `bev_src_top_left_ratio` | 0.320 | 소스 상단 왼쪽 x |
| `bev_src_top_right_ratio` | 0.680 | 소스 상단 오른쪽 x |
| `bev_src_bottom_left_ratio` | 0.050 | 소스 하단 왼쪽 x |
| `bev_src_bottom_right_ratio` | 0.950 | 소스 하단 오른쪽 x |
| `bev_dst_left_ratio` | 0.250 | 목표 왼쪽 경계 |
| `bev_dst_right_ratio` | 0.750 | 목표 오른쪽 경계 |
| `bev_edge_margin_px` | 15 | BEV 좌우 가장자리 제거 (아티팩트) |

**차선 탐색 / 피팅**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `min_run_width` | 2 | 최소 연속 픽셀 폭 |
| `max_run_width` | 45 | 최대 run 폭 (넘으면 벽/콘) |
| `lane_search_margin` | 45.0 | 행간 탐색 허용 거리 (px) |
| `min_row_span` | 18 | 피팅 포인트 최소 세로 범위 |
| `min_poly_pts` | 8 | 다항식 피팅 최소 점 수 |
| `poly_degree` | 1 | 1=직선, 2=2차 곡선 |
| `fit_residual_px` | 9.0 | inlier 허용 잔차 (px) |
| `max_poly_slope` | 3.0 | 다항식 기울기 최대 절댓값 |

**반폭 학습 (half_width)**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `half_width_update_max_epsi_deg` | **30.0** | 반폭 학습 허용 최대 epsi (이전 15°→30°로 완화) |
| `half_width_alpha` | 0.15 | 반폭 EMA 학습률 |
| `lane_width_bot_min_px` | 360 | 반폭 학습 유효 최소 하단 폭 |

**단일 차선 복원**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `single_lane_max_center_jump` | **150.0** | 단일차선 최대 센터 점프 (이전 80→150px) |
| `single_lane_half_width_top` | 160.0 | 반폭 학습 전 상단 기본값 |
| `single_lane_half_width_bottom` | 160.0 | 반폭 학습 전 하단 기본값 |

**유효성 검사**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `max_valid_epsi_deg` | 100.0 | 최대 허용 헤딩각 |
| `max_valid_path_span_px` | **320.0** | 경로 상하단 x 변화 허용치 (이전 230→320px) |
| `poly_reject_delta` | 50.0 | 이전 EMA 대비 최대 위치 변화량 |

**오프셋 계산**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `control_row_ratio` | 0.72 | 제어 행 위치 (ROI 기준 비율) |
| `epsi_weight` | **0.3** | 헤딩 오차 가중치 (이전 0.0→0.3, 큰 이탈 보정) |

**로깅**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `log_diag` | **True** | 진단 CSV 자동 저장 (기본 활성화) |
| `capture_dir` | `~/capstone_ws/logs` | CSV / 캡처 이미지 저장 위치 |
| `show_window` | False | OpenCV 디버그 창 |

### 발행 토픽

| 토픽 | 타입 | QoS | 설명 |
|---|---|---|---|
| `/lane_offset` | Float32 | TRANSIENT_LOCAL | 정규화 오프셋 [-1,+1]. + = 차선 중심 우측 |
| `/lane_valid` | Bool | TRANSIENT_LOCAL | 차선 유효성 (False → lane_recovery 정지) |
| `/lane_debug` | Image | default | 디버그 시각화 |

> **TRANSIENT_LOCAL QoS**: 구독자가 늦게 시작해도 마지막 값을 즉시 수신합니다 (latching). lane_recovery_node 재시작 시 lane_valid를 즉시 받을 수 있습니다.

---

## 5. lane_recovery_node 상세

### 제어 원리 (P제어 + 속도 스케일링)

```
입력: /lane_offset (Float32, [-1,+1])
출력: /cmd_vel_recovery (Twist)

오프셋 부호:
  offset > 0 (우측 이탈) → wz < 0 (우회전으로 복귀)
  offset < 0 (좌측 이탈) → wz > 0 (좌회전으로 복귀)

제어 수식:
  raw_wz      = -k_steer × offset
  wz_target   = clamp(raw_wz, -max_wz, +max_wz)
  wz          = rate_limiter(wz_target, prev_wz, wz_rate_limit)  ← 급변 억제
  speed_factor = max(0.4, 1.0 - speed_offset_factor × |offset|) ← 이탈 시 감속
  vx          = min(base_speed × speed_factor, max_vx)
```

### 상태 전이

```
DISABLED
  → 아무것도 발행하지 않음

ENABLED:
  ├── offset 없거나 오래됨 (> offset_timeout_sec=0.35s)
  │     → zero Twist 발행 (stale_or_missing_offset)
  ├── lane_valid=False가 invalid_hold_frames=4 연속 이상
  │     → zero Twist 발행 (lane_invalid)  ← 히스테리시스
  └── 정상
        → P제어 Twist 발행 (drive)
```

### lane_valid QoS (TRANSIENT_LOCAL)

```python
_latching_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
self.sub_valid = self.create_subscription(
    Bool, '/lane_valid', self._on_lane_valid, _latching_qos
)
```

lane_detector_node가 먼저 시작되어도 lane_recovery_node가 나중에 실행되면 마지막 lane_valid 값을 즉시 수신합니다.

### 종료 처리

```python
finally:
    if rclpy.ok():
        node._publish_zero('shutdown')  # 정지 명령 전송 후 종료
```

Ctrl+C / 정상 종료 시 반드시 zero Twist를 발행하고 종료합니다.

### 파라미터 정리

**속도**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `base_speed` | **0.20** | 기본 전진 속도 (m/s) |
| `max_vx` | **0.20** | 전진 속도 상한 (m/s) |
| `speed_offset_factor` | 0.5 | offset 비례 감속 계수 (이탈 시 최소 40%까지 감속) |

**조향**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `k_steer` | 0.5 | P 이득 |
| `dead_zone` | 0.05 | 불감대 (이 범위 내 offset 무시) |
| `max_wz` | **0.8** | 최대 각속도 (rad/s) |
| `wz_rate_limit` | 0.5 | wz 최대 변화율 (rad/s²) — 급변 억제 |

**타이밍 / 유효성**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `cmd_rate_hz` | 20.0 | 제어 명령 주기 (Hz) |
| `offset_timeout_sec` | 0.35 | offset stale 판정 시간 (초) |
| `require_lane_valid` | True | False이면 lane_valid 무시 |
| `invalid_hold_frames` | 4 | 연속 invalid 프레임 수 초과 시 정지 |

**기타**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `auto_enable` | False | True이면 시작 즉시 활성화 |
| `output_topic` | `/cmd_vel_recovery` | 발행 토픽 (twist_mux 없으면 `/cmd_vel`로 변경) |
| `keyboard_stop` | True | SPACE 키 비상 정지 |
| `log_csv` | **True** | CSV 로그 자동 저장 |
| `capture_dir` | `~/capstone_ws/logs` | CSV 저장 위치 |

### CSV 로그 컬럼 (14컬럼, 20 Hz)

`ts_sec`, `enabled`, `lane_valid`, `offset`, `offset_age_sec`, `offset_fresh`,
`reason`, `cmd_vx`, `cmd_vy`, `cmd_wz`, `base_speed`, `k_steer`, `dead_zone`, `max_wz`

`reason` 가능한 값:
- `drive` — 정상 주행 명령
- `stale_or_missing_offset` — offset 타임아웃 (lane_detector 미발행)
- `lane_invalid` — lane_valid=False 연속 초과
- `disabled` — lane_recovery_enable=False
- `keyboard_space_stop` — SPACE 키 정지
- `shutdown` — 노드 종료 시 zero 발행

---

## 6. traffic_light_detector_node 상세

### 두 가지 동작 모드

**HSV 모드** (`blob_path=""`, 기본값)
- OpenCV HSV 색상 분석
- 조명 변화에 취약, 수동 조정 필요

**ML 모드** (`blob_path=<경로>`)
- YOLOv8n-detect blob 사용 (green=0, red=1)
- 조명 변화에 강건

### Voting 시스템

```
vote_buffer_size = 10  (최근 10개)
min_vote_samples = 3   (최소 3개 수집 후 판정)
red_vote_ratio   = 0.5 (50% 이상 → RED)
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `blob_path` | `""` | 비어있으면 HSV 모드 |
| `conf_thresh` | 0.50 | ML 모드 탐지 임계값 |
| `roi_bottom_ratio` | 0.55 | 신호등 ROI 하단 (하단은 차선) |
| `vote_buffer_size` | 10 | 판정 버퍼 크기 |
| `show_window` | False | 디버그 창 |

### 발행 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/traffic_light` | String | `"RED"` / `"GREEN"` / `"NONE"` |
| `/cmd_vel_traffic` | Twist | RED 시 zero Twist (**현재 미연결**) |
| `/traffic_light_debug` | Image | 디버그 시각화 |

---

## 7. DiagLogger — 진단 CSV 로거

`log_diag:=true`(기본값)로 실행하면 `~/capstone_ws/logs/lane_diag_YYYYMMDD_HHMMSS.csv`에 매 프레임을 기록합니다.

### 전체 컬럼 (52컬럼)

**기본 정보**

| 컬럼 | 설명 |
|---|---|
| `ts_sec` | 프레임 타임스탬프 |
| `frame_no` | 프레임 번호 |
| `fps` | 최근 10프레임 평균 처리 속도 |

**인식 결과**

| 컬럼 | 설명 | 분석 포인트 |
|---|---|---|
| `center_mode` | BOTH/LEFT/RIGHT/NONE/REJECT | BOTH/LEARNED 비율이 높을수록 안정 |
| `lc_status` / `rc_status` | 좌/우 차선 상태 | ACCEPT가 정상 |
| `lane_offset_norm` | 정규화 오프셋 [-1,+1] | 0=중앙 |
| `lane_valid` | 제어 유효 여부 (0/1) | 0 비율 높으면 파라미터 확인 |
| `epsi_deg` | 차선 헤딩 오차 (도) | 평균 20° 초과 시 BEV 조정 필요 |
| `reject_reason` | 거부 원인 문자열 | 가장 많은 원인을 파라미터로 수정 |

**경로 품질**

| 컬럼 | 설명 |
|---|---|
| `path_x_span_px` | 경로 상하단 x 이동량 (기준: 320px) |
| `center_jump_px` | 이전 프레임 대비 센터 x 변화 (기준: 150px) |
| `half_width_px` | 현재 학습된 반폭 (px) |
| `lane_width_top/mid/bot_px` | 상/중/하단 차선 폭 |

**마스크 통계**

| 컬럼 | 설명 |
|---|---|
| `raw_mask_px` | blob 필터 전 마스크 픽셀 수 |
| `filtered_mask_px` | blob 필터 후 마스크 픽셀 수 |
| `blob_removed_px` | 필터로 제거된 픽셀 수 |

**NN 진단** ← 신규 추가

| 컬럼 | 설명 | 해석 |
|---|---|---|
| `nn_max_conf` | 프레임 최대 NN 신뢰도 | `< conf_thresh`이면 검출 미달 |
| `nn_pass_count` | conf_thresh 통과 앵커 수 | `== 0`이면 모델 점수 미달, `> 0`이면 마스크 생성 단계 문제 |

**IMU 교차 검증**

| 컬럼 | 설명 |
|---|---|
| `imu_fresh` | IMU 수신 여부 (1/0) |
| `imu_angular_z_rads` | 최신 yaw rate (rad/s, CCW+) |
| `imu_delta_yaw_deg` | 마지막 유효 감지 이후 누적 회전량 |
| `imu_epsi_suppressed` | IMU가 epsi reject 억제 여부 |

### no_mask 원인 진단 방법

```python
import pandas as pd
df = pd.read_csv('~/capstone_ws/logs/lane_diag_YYYYMMDD_HHMMSS.csv')

no_mask = df[df['reject_reason'] == 'no_mask']
print(f"no_mask 비율: {len(no_mask)/len(df)*100:.1f}%")
print(f"nn_max_conf 평균: {no_mask['nn_max_conf'].mean():.4f}")
print(f"conf 미달(nn_pass_count=0): {(no_mask['nn_pass_count']==0).mean()*100:.1f}%")

# conf_thresh 조정 시뮬레이션
for t in [0.40, 0.45, 0.50, 0.55]:
    salvage = (no_mask['nn_max_conf'] >= t).sum()
    print(f"conf_thresh={t}: {salvage}건 복구 가능")
```

### reject_reason 분포 빠른 확인

```python
df['reject_reason'].value_counts(normalize=True).head(10).mul(100).round(1)
```

---

## 8. Blob 변환 방법 (pt → blob)

### 변환 스크립트

```python
import shutil, blobconverter
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', imgsz=[224, 640], opset=12, simplify=True)

blob_path = blobconverter.from_onnx(
    model='best.onnx',
    data_type='FP16',
    shaves=6,
    use_cache=False,
    output_dir='/tmp/blob_out',
)
shutil.copy(blob_path, 'src/camera_stack/models/best/lane_seg_640x224/best_openvino_2022.1_6shave1.blob')
```

> **주의**: `imgsz=[224, 640]` — **[H, W] 순서**. 반대로 하면 추론 오류.

### blob 빠른 확인 도구

```bash
QT_QPA_PLATFORM=xcb python3 tools/lane_seg_blob_check.py \
  --blob models/best/lane_seg_640x224/best_openvino_2022.1_6shave1.blob \
  --conf 0.45 --mask-thresh 0.45

# 키: +/- conf 조정, [/] mask 조정, s 저장, ESC 종료
```

---

## 9. ROS2 토픽 전체 정리

### camera_stack 발행 토픽

| 토픽 | 타입 | QoS | 발행 노드 |
|---|---|---|---|
| `/lane_offset` | Float32 | TRANSIENT_LOCAL | lane_detector_node |
| `/lane_valid` | Bool | TRANSIENT_LOCAL | lane_detector_node |
| `/lane_debug` | Image | default | lane_detector_node |
| `/traffic_light` | String | default | traffic_light_detector_node |
| `/cmd_vel_traffic` | Twist | default | traffic_light_detector_node (**미연결**) |
| `/cmd_vel_recovery` | Twist | default | lane_recovery_node |

### camera_stack 구독 토픽

| 토픽 | 타입 | QoS | 구독 노드 |
|---|---|---|---|
| `/lane_recovery_enable` | Bool | default | lane_recovery_node |
| `/lane_valid` | Bool | TRANSIENT_LOCAL | lane_recovery_node |
| `/imu/data` | Imu | default | lane_detector_node |

---

## 10. 파라미터 튜닝 가이드

### no_mask가 많을 때

```bash
# conf_thresh 확인 (lane_diag CSV의 nn_max_conf 평균 확인 후 결정)
# 2022.1 모델은 0.45가 적정
-p conf_thresh:=0.45   # 기본값 (이미 적용됨)
-p conf_thresh:=0.40   # 더 낮출 경우 (오탐 위험 증가)
```

### path_span reject가 많을 때

```bash
-p max_valid_path_span_px:=320.0  # 기본값 (이미 적용됨)
-p max_valid_path_span_px:=400.0  # 더 완화 시
```

### single_lane jump reject가 많을 때

```bash
-p single_lane_max_center_jump:=150.0  # 기본값 (이미 적용됨)
-p single_lane_max_center_jump:=200.0  # 더 완화 시
```

### half_width가 DEFAULT에 머물 때 (LEARNED 안 됨)

```bash
# epsi가 크면 학습이 차단됨 → 허용 범위 확인
-p half_width_update_max_epsi_deg:=30.0  # 기본값 (이미 적용됨)
```

### 조향 진동이 심할 때

```bash
-p k_steer:=0.3       # 이득 낮춤 (기본 0.5)
-p dead_zone:=0.08    # 사각지대 넓힘
-p wz_rate_limit:=0.3 # 변화율 제한 강화 (기본 0.5 rad/s²)
```

### 조향 반응이 느릴 때

```bash
-p k_steer:=0.7       # 이득 높임
-p max_wz:=1.0        # 최대 각속도 높임 (기본 0.8)
```

### BEV 사다리꼴 조정 방법

1. `show_window:=true`로 실행 → 디버그 창 확인
2. BEV 창에서 차선이 수직에 가깝게 보이도록 조정
   - V자로 모이면: `bev_src_top_*` 좌우 간격을 늘림
   - 역V자면: 줄임

---

## 11. 알려진 한계와 미래 과제

### 현재 한계

| 항목 | 내용 |
|---|---|
| **twist_mux 미설치** | `/cmd_vel_traffic` 미연결, 신호등 정지 불가 |
| **위치 추정 없음** | EKF/AMCL 미사용 → 누적 오차 |
| **장애물 회피 없음** | LiDAR costmap 미연동 |
| **epsi 큰 구간 추종 한계** | epsi 평균 20°+ 시 single_left 증가 → 속도/조향 파라미터 조정 필요 |
| **SIGKILL 시 터미널 raw mode** | `kill -9` 후 `reset` 명령으로 복구 |

### 통합을 위한 미래 과제

1. **twist_mux 설치** — `apt install ros-humble-twist-mux`
2. **mission_stack** — RED 감지 시 lane_recovery 일시 정지
3. **카메라 + LiDAR 통합 launch** 파일 구성
4. **BEV 캘리브레이션** — `xm_per_pix` 실측으로 lane_offset_m 정확도 향상
5. **stop_line_detector** — 정지선 감지 → mission_fsm 연동
