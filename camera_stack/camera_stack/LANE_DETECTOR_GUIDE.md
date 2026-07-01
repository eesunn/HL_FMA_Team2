# camera_stack — 차선 인식 구현 정리

OAK-D Pro + YOLOv8-seg + ROS2 Humble 기반 차선 인식 및 경로 추종 시스템의 전체 구현 내용을 정리한 문서.

---

## 목차

1. [전체 구조](#1-전체-구조)
2. [하드웨어 / 소프트웨어 환경](#2-하드웨어--소프트웨어-환경)
3. [lane_detector_node 상세](#3-lane_detector_node-상세)
4. [파라미터 레퍼런스](#4-파라미터-레퍼런스)
5. [디버그 화면 읽는 법](#5-디버그-화면-읽는-법)
6. [모델 파일](#6-모델-파일)
7. [데이터 수집](#7-데이터-수집)
8. [실행 명령어](#8-실행-명령어)
9. [파라미터 튜닝 가이드](#9-파라미터-튜닝-가이드)
10. [알려진 이슈 및 해결책](#10-알려진-이슈-및-해결책)

---

## 1. 전체 구조

```
OAK-D Pro
  └─ Camera (CAM_A, 640×224 BGR, 25fps)
       ├─ → NeuralNetwork (YOLOv8-seg blob)
       │         └─ nn_queue → _parse_yolov8seg() → binary mask
       │                            └─ _filter_blobs()        (벽·노이즈·반사광 제거)
       │                                 └─ perspective transform → BEV mask
       │                                      └─ _apply_bev_mask_filters()   ← NEW
       │                                           └─ row run 추적 + robust polyfit
       │                                           └─ _lane_center_poly() → center_x, epsi
       │                                               └─ _compute_path_x_span()    ← NEW
       │                                               └─ _evaluate_lane_valid()    ← NEW
       │                                                    └─ (VALID 시) _compute_offset()
       │                                                         └─ EMA 평활화 → 토픽 발행
       └─ → rgb_queue (show_window=true 일 때만) → _update_display()
                                                         └─ 's' 키 → raw 이미지 캡처 저장
```

**발행 토픽:**

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/lane_offset` | `std_msgs/Float32` | 차선 중심 오프셋. 오른쪽 = +, 왼쪽 = -, 범위 [-1, +1] |
| `/lane_offset_m` | `std_msgs/Float32` | 오프셋 미터 환산 (xm_per_pix > 0 일 때만 발행) |
| `/lane_valid` | `std_msgs/Bool` | 제어에 사용 가능한 차선 여부. False면 offset 무시 권장 |
| `/lane_curvature` | `std_msgs/Float32` | 곡률반경(m). poly_degree=1이면 1e6(직선) |
| `/lane_debug` | `sensor_msgs/Image` | 마스크 + 다항식 곡선 + 경로 시각화 이미지 |

---

## 2. 하드웨어 / 소프트웨어 환경

| 항목 | 사양 |
|---|---|
| 카메라 | OAK-D Pro (IMX219-120, 120° FOV) |
| VPU | MyriadX (OAK-D 내장) |
| 개발 PC | Ubuntu 22.04, ROS2 Humble |
| depthai | 3.x (2.x와 API 호환 불가) |
| 모델 | YOLOv8s-seg, 입력 640×224 |

---

## 3. lane_detector_node 상세

### 3-1. depthai 3.x 초기화 방식

depthai 3.x는 2.x와 API가 완전히 다릅니다.

```python
# 3.x 방식 (현재 코드)
pipeline = dai.Pipeline(dai.Device())          # Device를 Pipeline에 주입
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cam_out = cam.requestOutput((640, 224), type=dai.ImgFrame.Type.BGR888p, fps=25)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blob_path)
nn.input.setBlocking(False)
cam_out.link(nn.input)
nn_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)  # start() 이전에 호출
pipeline.start()
```

`createOutputQueue()`는 반드시 `pipeline.start()` 이전에 호출해야 합니다.

---

### 3-2. YOLOv8-seg 출력 디코딩

YOLOv8-seg blob은 두 개의 출력 텐서를 반환합니다.

| 출력 | 형태 | 내용 |
|---|---|---|
| output0 | (37 × 2940) | 앵커별 예측값: [cx, cy, w, h, conf, mask_coeff×32] |
| output1 | (32 × 56 × 160) | 마스크 프로토타입 |

**2940 앵커 계산:**

```
stride 8  : (640/8) × (224/8)  = 80 × 28 = 2240
stride 16 : (640/16) × (224/16) = 40 × 14 =  560
stride 32 : (640/32) × (224/32) = 20 ×  7 =  140
총합 : 2940
```

**마스크 합성 과정:**

```python
preds = output0.reshape(37, 2940).T        # (2940, 37)
protos = output1.reshape(32, 56, 160)      # (32, 56, 160)

scores = preds[:, 4]
keep = scores > conf_thresh                # 신뢰도 필터

mask_coeffs = preds[keep, 5:]              # (N, 32)
proto_flat  = protos.reshape(32, 56*160)   # (32, 8960)
masks_raw   = mask_coeffs @ proto_flat     # (N, 8960)
masks       = sigmoid(masks_raw)           # (N, 56, 160)
combined    = max(masks, axis=0)           # (56, 160) — 최대 투표
binary      = combined > mask_thresh       # 이진화
binary      = resize(binary, (640, 224))   # 원본 해상도로 업스케일
binary[:roi_cut, :] = 0                    # 상단 ROI 제거
bev_binary  = warpPerspective(binary)      # 도로 평면을 버드아이뷰로 변환
```

YOLO 추론과 마스크 생성은 학습 영상과 동일한 원본 카메라 좌표에서 수행합니다.
BEV는 blob 필터가 끝난 이진 마스크에만 적용하므로 모델 입력 자체는 바뀌지 않습니다.

---

### 3-3. blob 형태 필터 (`_filter_blobs`)

마스크에 포함된 벽·반사광·점 노이즈를 blob 단위로 제거합니다.
차선은 가늘고 길기 때문에 아래 세 조건을 모두 통과합니다.

| 제거 조건 | 파라미터 | 제거 대상 |
|---|---|---|
| blob 면적 > 이미지 면적 × `max_blob_ratio` | `max_blob_ratio=0.15` | 벽·대형 반사 |
| blob 면적 < `min_blob_area` | `min_blob_area=100` | 점 노이즈 |
| `max(w,h) / min(w,h)` < `min_aspect_ratio` | `min_aspect_ratio=1.5` | 둥근 반사광 |

---

### 3-4. BEV 가장자리 마스크 필터 (`_apply_bev_mask_filters`) — NEW

BEV 변환 직후 좌우 가장자리 `bev_edge_margin_px`(기본 40px)를 강제 0으로 만듭니다.
BEV 워프 시 좌우 끝부분에 생기는 왜곡 아티팩트와 도로 외벽 픽셀을 제거합니다.

```python
# BEV 변환 → 가장자리 40px 제거 → 차선 추적
filtered[:, :40] = 0
filtered[:, 600:] = 0   # (640-40=600)
```

`use_bev=false`이거나 `bev_edge_margin_px=0`이면 이 단계를 건너뜁니다.

---

### 3-5. 경로 생성 및 중심 계산 (`_lane_center_poly`)

현재 기본 설정은 `use_bev=true`입니다. 원본 영상에서 사다리꼴로 보이는 도로
영역을 직사각형으로 펼친 뒤 차선 피팅과 중심 계산을 수행합니다. 따라서
원근 때문에 영상 위·아래에서 달라지던 픽셀 차선 폭이 거의 일정해지고,
한쪽 차선을 일정 폭만큼 이동해 중심 경로를 복원하기 쉬워집니다.

현재 확정된 BEV 사다리꼴 (2026-06-17 실측):

```text
상단 y=0.62H: x=0.32W ~ 0.68W
하단 y=1.00H: x=0.05W ~ 0.95W
           ↓ perspective transform
BEV 전체 높이: x=0.25W ~ 0.75W
```

행별로 모든 흰색 픽셀을 평균하지 않고, 연속된 흰색 픽셀 구간(run)을
차선 후보로 분리합니다. 차량 가까운 하단부터 이전 차선 위치와 가까운 run을
추적한 뒤 robust fitting을 적용하여 좌우 차선을 모델링합니다.

#### 행별 스캔

```
실제 사용 행 범위:
  row_start = IMG_H × roi_start_ratio
  → row_start ~ IMG_H 구간만 사용
```

각 행에서 **이전 프레임의 `center_x`** 를 기준으로 좌/우 후보를 분리합니다.
후보 픽셀은 연속된 구간별로 나누며 `min_run_width` 이상 `max_run_width` 이하인 구간만 선택합니다.

```
한 행의 흰색 픽셀
  → 연속 run 분리
  → min_run_width ≤ run 폭 ≤ max_run_width 필터   ← max_run_width 상한 추가
  → 좌/우 영역 분리
  → 이전 smooth 다항식의 예상 X와 가까운 run 선택
  → 이전 다항식이 없으면 하단의 바깥쪽 run에서 추적 시작
```

`max_run_width=45`보다 넓은 run은 벽·콘·대형 반사광으로 판단하여 upstream에서 제외합니다.

#### Robust 다항식 피팅

```python
# 1. 최초 피팅
raw_coef = np.polyfit(rows, cols, poly_degree)

# 2. 피팅선에서 fit_residual_px 이내인 포인트만 inlier로 선택
inliers = abs(cols - polyval(raw_coef, rows)) <= fit_residual_px

# 3. inlier 수와 비율을 검사한 뒤 재피팅
coef = np.polyfit(rows[inliers], cols[inliers], poly_degree)
```

다음 조건을 만족하지 않으면 해당 차선 피팅을 폐기합니다.

- 포인트 수 `min_poly_pts` 이상
- 포인트 세로 범위 `min_row_span` 이상
- inlier 비율 `min_inlier_ratio` 이상
- 실제 inlier 행 범위의 기울기가 `max_poly_slope` 이하
- 이전 EMA 다항식과 실제 inlier 행 범위의 위치 차이가 `poly_reject_delta` 이하

`poly_degree=1`은 직선(기본), `poly_degree=2`는 곡선 피팅입니다.

#### 중심 경로 다항식 생성 (4가지 케이스)

| 케이스 | 조건 | center_coef 계산 |
|---|---|---|
| A | 좌/우 smooth 계수 및 폭 검사 유효 | `(smooth_lc + smooth_rc) / 2` |
| B | 현재 프레임에서 좌측만 유효 | `smooth_lc + smooth_half_width` |
| C | 현재 프레임에서 우측만 유효 | `smooth_rc - smooth_half_width` |
| D | smooth 계수 모두 None | `None` 반환 |

"유효"란 robust fitting과 프레임 간 변화 검사를 통과한 현재 프레임의
차선을 말합니다. 양쪽 차선이 현재 프레임에서 모두 `ACCEPT`되면 좌우 inlier가
동시에 존재하는 공통 행 범위를 구하고, 그 범위 안의 상단·중앙·하단에서만
차선 폭을 검사합니다.

양쪽 차선의 폭 검사가 실패하면 inlier 세로 범위, 포인트 수, residual을
비교해 품질이 더 높은 한쪽 차선만 사용합니다.

#### 다점 가중 평균으로 center_x 계산

```python
eval_rows = np.arange(row_start, H, 5)           # 데이터 구간 전체, 5행마다
centers   = A·eval_rows² + B·eval_rows + C        # 각 행에서 중심 위치
weights   = eval_rows / H                         # 선형 가중치: 차량에 가까운 행 = 높은 가중치
center_x  = np.average(centers, weights=weights)  # 가중 평균
```

#### 헤딩 오차(epsi) 계산

```python
mid_row = (row_start + H) / 2.0                   # 데이터 구간 중앙
slope   = 2·A·mid_row + B                         # 중심 경로의 접선 기울기
epsi    = atan(slope)                              # 헤딩 오차 (rad)
```

`epsi`는 제어 출력에 직접 합산하지 않고(`epsi_weight=0.0` 기본), 진단 및 유효성 검사에만 사용합니다.

#### A 필터: 좌/우 차선 다항식 독립 EMA

raw `lc_coef` / `rc_coef`를 그대로 사용하지 않고, 각 차선에 대해 독립적으로 EMA(지수 이동 평균)를 적용합니다.

```
감지 성공 시:
  smooth_lc = poly_alpha × lc_coef + (1 - poly_alpha) × smooth_lc  (EMA 갱신)
  _lc_last_seen = now

피팅 거부 또는 감지 실패 시:
  (now - _lc_last_seen) ≤ poly_hold_sec  → smooth_lc 유지 (HOLD)
  (now - _lc_last_seen) >  poly_hold_sec → smooth_lc = None (리셋)
```

center_coef 계산에는 **smooth_lc_coef / smooth_rc_coef** 만 사용합니다.

---

### 3-6. 경로 x 이동량 계산 (`_compute_path_x_span`) — NEW

```python
# center poly의 scan 시작 행 ~ 영상 최하단 사이에서 x가 얼마나 변하는지 계산
x_start = coef[0]*row_start² + coef[1]*row_start + coef[2]
x_end   = coef[0]*(H-1)²   + coef[1]*(H-1)   + coef[2]
path_x_span = abs(x_end - x_start)
```

`path_x_span`이 크면 경로가 급격하게 구부러졌다는 의미입니다.
`max_valid_path_span_px` 초과 시 `_evaluate_lane_valid`에서 발행을 차단합니다.

---

### 3-7. 차선 유효성 게이트 (`_evaluate_lane_valid`) — NEW

NN 검출과 별개로, 계산된 경로가 실제 제어에 사용할 수 있는지 다중 조건으로 검증합니다.
아래 조건 중 하나라도 실패하면 `/lane_offset` 발행을 이 프레임에서 건너뜁니다.

```
1. center_x is None           → 경로 없음 (MISS)
2. mode = NONE / REJECT       → 검출 실패
3. |epsi| > 20°               → 헤딩 너무 큼: 차량이 비스듬히 보임
4. path_x_span > 140px        → 경로가 급격히 꺾임 (노이즈 의심)
5. max_mean_run_w > 45px      → run 폭이 너무 두꺼움 (벽·콘 의심)
6. 단일 차선 모드 시 추가:
     - path_x_span is None    → 단일 차선인데 경로 범위 없음
     - 좌/우 row_span < min_row_span → 차선 검출 범위가 너무 좁음
```

| 결과 | `_vis_lane_state` | `/lane_valid` 발행 |
|---|---|---|
| 모든 조건 통과 | VALID (초록) | True |
| 조건 실패 | REJECT (주황) | False |
| 경로 없음 | MISS | False |

`reject_reason`은 디버그 이미지 HUD와 진단 CSV에 기록됩니다.

---

### 3-8. 오프셋 정규화 (`_compute_offset`)

```
ey_norm   = (center_x - 320) / 320       # 횡 편차, 오른쪽 = +
epsi_norm = epsi / (π/2)                 # 헤딩 오차 정규화
offset    = clip(ey_norm + epsi_weight × epsi_norm, -1, +1)
```

- `offset > 0`: 경로 중심이 이미지 중앙 기준 오른쪽 → 차량이 차선 왼쪽에 있음 → 우회전 필요
- `offset < 0`: 경로 중심이 이미지 중앙 기준 왼쪽 → 차량이 차선 오른쪽에 있음 → 좌회전 필요
- `epsi_weight=0.0`(기본): 헤딩 오차는 오프셋에 합산하지 않음. 직선로에서 권장.

**허용 오차(dead zone) 설정:**

`lane_recovery_node`의 `dead_zone` 파라미터로 설정합니다.

```
픽셀 단위 허용 범위 → 정규화 값 변환:
  ±10 px → dead_zone = 10 / 320 ≈ 0.031
  ±20 px → dead_zone = 20 / 320 ≈ 0.063
```

---

### 3-9. C 필터: scalar EMA + Hold 기능

A 필터로 평활화된 다항식 → center_x → 정규화된 raw offset을 스칼라 EMA로 한 번 더 평활화합니다.
`_evaluate_lane_valid` 통과 시에만 EMA를 갱신하고 발행합니다.

```python
# 감지 성공(lane_valid=True) 시: EMA 평활화
smooth = ema_alpha × raw + (1 - ema_alpha) × prev_smooth

# 발행 조건:
#   - 감지 성공: smooth 값 발행
#   - 감지 실패이지만 hold_sec 이내: 마지막 smooth 값 발행 (HOLD 상태)
#   - hold_sec 초과 미감지: 상태 초기화, 발행 중단
```

**디버그 로그:**

```
[offset] raw=0.123  smooth=0.105  epsi=3.2°  VALID     ← 정상 감지
[offset] raw=MISS   smooth=0.105  epsi=-      HOLD      ← hold 중
```

---

### 3-10. 진단 CSV 로깅 (DiagLogger)

`log_diag:=true` 파라미터로 활성화합니다. 매 프레임 47개 컬럼을 기록합니다.

파일 경로: `capture_dir/lane_diag_YYYYMMDD_HHMMSS.csv`

주요 컬럼:

| 그룹 | 컬럼 | 설명 |
|---|---|---|
| 기본 | `ts_sec`, `frame_no`, `fps` | 타임스탬프, 프레임 번호, FPS |
| 검출 | `center_mode`, `lc_status`, `rc_status` | BOTH/LEFT/RIGHT/NONE/REJECT |
| 오프셋 | `lane_offset_norm`, `lane_offset_m`, `lane_valid` | 오프셋 및 유효성 |
| 헤딩 | `epsi_deg` | 차선 헤딩 오차(도) |
| 다항식 | `lc_a,b,c` `rc_a,b,c` `center_a,b,c` | EMA 평활 계수 |
| 진단 1순위 | `reject_reason`, `left/right_total_pts`, `left/right_row_span_px` | 검출 품질 |
| 진단 1순위 | `left/right_mean_run_w`, `path_x_span_px`, `curvature_r_m` | 경로 형상 |
| 진단 2순위 | `raw_mask_px`, `bev_mask_px`, `filtered_mask_px`, `blob_removed_px` | 마스크 픽셀 추적 |
| 진단 3순위 | `lane_width_top/mid/bot_px`, `center_jump_px` | 차선 폭·점프 |

---

## 4. 파라미터 레퍼런스

### lane_detector_node

| 파라미터 | 기본값 | 역할 | 조정 방향 |
|---|---|---|---|
| `blob_path` | (필수) | `.blob` 파일 경로 | — |
| `conf_thresh` | 0.65 | YOLO 앵커 신뢰도 커트라인 | 노이즈 많으면 올림, 인식 안 되면 낮춤 |
| `mask_thresh` | 0.45 | 마스크 sigmoid 이진화 임계값 | 차선 끊기면 낮춤 (0.3) |
| `roi_start_ratio` | 0.55 | 원본 마스크 상단 N% 차단 | BEV 사다리꼴 상단과 함께 조정 |
| `use_bev` | true | 마스크를 버드아이뷰로 변환 | 원본 좌표 비교 시 false |
| `bev_src_top_y_ratio` | **0.620** | 원본 사다리꼴 상단 y | 카메라 지평선 아래 차선 시작점 |
| `bev_src_top_left_ratio` | **0.320** | 원본 사다리꼴 좌상단 x | 실제 왼쪽 차선에 맞춤 |
| `bev_src_top_right_ratio` | **0.680** | 원본 사다리꼴 우상단 x | 실제 오른쪽 차선에 맞춤 |
| `bev_src_bottom_left_ratio` | **0.050** | 원본 사다리꼴 좌하단 x | 실제 왼쪽 차선에 맞춤 |
| `bev_src_bottom_right_ratio` | **0.950** | 원본 사다리꼴 우하단 x | 실제 오른쪽 차선에 맞춤 |
| `bev_dst_left_ratio` | 0.250 | BEV 왼쪽 차선 목표 x | 보통 기본값 유지 |
| `bev_dst_right_ratio` | 0.750 | BEV 오른쪽 차선 목표 x | 보통 기본값 유지 |
| `bev_scan_start_ratio` | 0.05 | BEV 피팅 시작 y | 상단 변환 노이즈가 많으면 올림 |
| `bev_edge_margin_px` | **15** | BEV 좌우 가장자리 mask 제거 폭(px) | 외곽 노이즈가 많으면 올림 (최대 IMG_W/3) |
| `road_half_w` | 80 | 이전 실행 인자 호환용 초기값 | 현재 중심 복원에는 y별 반폭 모델 사용 |
| `gap_thresh` | 1 | 행당 최소 흰색 픽셀 수 | 가는 차선이 무시되면 낮춤, 노이즈 많으면 올림 |
| `max_blob_ratio` | 0.15 | blob 최대 허용 면적 비율 (벽 제거) | 벽이 차선으로 잡히면 낮춤 |
| `min_blob_area` | 100 | blob 최소 픽셀 수 (노이즈 제거) | 작은 노이즈 잡히면 올림 |
| `min_aspect_ratio` | 1.5 | blob 최소 종횡비 (반사광 제거) | 둥근 반사광 잡히면 올림, 차선 잘 잘리면 낮춤 |
| `min_poly_pts` | 8 | robust fitting 최소 포인트 수 | 너무 낮으면 소수 노이즈로 피팅됨 |
| `poly_degree` | **1** | 피팅 차수 (1=직선, 2=곡선) | 직선로는 1 권장. 2는 노이즈로 곡선이 출렁임 |
| `epsi_weight` | 0.0 | 헤딩 오차 가중치 (0=사용 안 함) | 직선로는 0 권장. 곡선로에서 반응이 느리면 0.1~0.2 |
| `poly_alpha` | 0.25 | 다항식 EMA 알파 (A 필터) | 경로 곡선이 흔들리면 낮춤 (0.1~0.2) |
| `poly_hold_sec` | 0.8 | 차선 미감지/거부 시 다항식 유지 시간(초) | 길면 잘못된 과거 경로 고착 |
| `min_run_width` | 2 | 행 내 연속 차선 후보의 최소 폭(px) | 점 노이즈가 많으면 올림 |
| `max_run_width` | **45** | 행 내 연속 run의 최대 폭(px). 초과 시 벽·콘으로 판단 | 정상 차선이 걸리면 올림 |
| `lane_search_margin` | 45.0 | 이전 차선 위치 기준 run 탐색 거리(px) | 작으면 급곡선을 놓치고 크면 다른 blob으로 이동 |
| `min_row_span` | 18 | 피팅 포인트의 최소 세로 분포(px) | 짧으면 기울기 오차가 커짐 |
| `fit_residual_px` | 9.0 | 최초 피팅선 기준 inlier 잔차(px) | 노이즈가 통과하면 낮춤 |
| `min_inlier_ratio` | 0.55 | 최종 피팅 최소 inlier 비율 | 노이즈가 많으면 올림 |
| `max_poly_slope` | 3.0 | ROI 내 최대 차선 기울기 | 정상 급곡선이 거부되면 올림 |
| `max_valid_epsi_deg` | **35.0** | 제어에 사용할 최대 헤딩각(도). 초과 시 REJECT | 정상 구간이 걸리면 올림 |
| `max_valid_path_span_px` | **180.0** | 경로 상하단 x 변화 허용치(px). 초과 시 REJECT | 급곡선 구간 주행 시 올림 |
| `poly_reject_delta` | 50.0 | 이전 EMA와 ROI 내 최대 위치 차이(px) | 경로 점프가 있으면 낮춤, 정상 급곡선이 거부되면 올림 |
| `min_lane_width` | 40.0 | 최소 좌우 차선 폭(px) | 카메라 환경에 맞춰 조정 |
| `max_lane_width` | 680.0 | ROI 중·하단 최대 좌우 차선 폭(px) | 카메라 환경에 맞춰 조정 |
| `lane_width_top_min_px` | 220.0 | 반폭 학습 허용 상단 차선폭 최소값 | 잘못된 반폭 학습 방지 |
| `lane_width_top_max_px` | 430.0 | 반폭 학습 허용 상단 차선폭 최대값 | 잘못된 반폭 학습 방지 |
| `lane_width_mid_min_px` | 320.0 | 반폭 학습 허용 중단 차선폭 최소값 | 최신 CSV 기준 p50 근처 보호 |
| `lane_width_mid_max_px` | 460.0 | 반폭 학습 허용 중단 차선폭 최대값 | 너무 넓은 폭 학습 차단 |
| `lane_width_bot_min_px` | 360.0 | 반폭 학습 허용 하단 차선폭 최소값 | 최신 CSV 기준 p50 근처 보호 |
| `lane_width_bot_max_px` | 540.0 | 반폭 학습 허용 하단 차선폭 최대값 | 너무 넓은 폭 학습 차단 |
| `half_width_update_max_epsi_deg` | 15.0 | 반폭 모델 업데이트 허용 최대 헤딩각 | 급곡선/불안정 프레임의 반폭 오염 방지 |
| `single_lane_half_width_top` | 160.0 | BEV 상단 차선→중심 반폭(px) | 노란 경로가 차선 쪽이면 늘림 |
| `single_lane_half_width_bottom` | 160.0 | BEV 하단 차선→중심 반폭(px) | 하단 노란 경로 위치로 조정 |
| `half_width_alpha` | 0.15 | 양쪽 검출 시 반폭 모델 학습 비율 | 폭 변화가 느리면 올림 |
| `single_lane_max_center_jump` | 80.0 | 편측 중심의 프레임 간 최대 변화(px) | 정상 급변이 거부되면 올림 |
| `single_lane_center_margin` | 80.0 | 편측 중심 경로의 화면 밖 허용 범위(px) | 보통 기본값 유지 |
| `single_lane_path_start_ratio` | 0.55 | 편측 경로 표시/검증 시작 y 비율 | 상단 왜곡이 심하면 올림 |
| `control_row_ratio` | 0.72 | 제어용 center_x를 계산할 y 비율 | 반응 늦으면 낮춤, 흔들리면 올림 |
| `ema_alpha` | 0.4 | 오프셋 scalar EMA 알파 (C 필터) | 흔들리면 낮춤, 반응 느리면 올림 |
| `hold_sec` | 1.5 | 미감지 시 마지막 오프셋 유지 시간(초) | 너무 길면 오방향 보정 위험 |
| `hold_decay` | 0.97 | HOLD 중 오프셋 프레임별 감쇠율 | 차선 잃을 때 서서히 직진 복귀 |
| `xm_per_pix` | 0.002222 | BEV 픽셀 1개당 meter (캘리브레이션 값) | 실측 후 고정 |
| `log_diag` | false | 진단 CSV 저장 여부 | 주행 분석 시 true |
| `log_path` | false | 경로 좌표 CSV 저장 여부 | 경로 분석 시 true |
| `show_window` | false | 디버그 OpenCV 창 표시 여부 | 개발 시 true |
| `capture_dir` | `~/lane_captures` | 's' 키 캡처 이미지 + CSV 저장 경로 | 원하는 경로로 변경 |

### lane_recovery_node

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `base_speed` | 0.15 | 전진 속도 (m/s) |
| `k_steer` | 0.5 | 조향 비례 게인 |
| `dead_zone` | 0.05 | 조향 무시 구간 (≈ ±16 px) |
| `max_wz` | 0.5 | 최대 각속도 (rad/s) |

### y별 차선 반폭 모델

노드 시작 시 BEV 스캔 시작 행의 반폭과 영상 최하단의 반폭을 직선으로 연결해
기본 y별 반폭 모델을 만듭니다. 양쪽 차선이 정상 검출되면
`smooth_half_width = (right - left) / 2` 측정값으로 모델을 보정합니다.
한쪽 차선만 보이면 이 모델을 더하거나 빼서 원근에 따라 달라지는 중심을
복원합니다.

초기값은 카메라를 실제 장착 위치에 고정한 상태에서 맞춰야 합니다.

- `single_lane_half_width_top`: 초록 BEV 스캔 시작선에서 차선과 도로 중심의 X 거리
- `single_lane_half_width_bottom`: 영상 최하단에서 차선과 도로 중심의 X 거리

---

## 5. 디버그 화면 읽는 법

### /lane_debug 이미지 (rqt_image_view)

```bash
ros2 run rqt_image_view rqt_image_view /lane_debug
```

`use_bev=true`이면 BEV 좌표계로 표시됩니다.

| 요소 | 색상 | 내용 |
|---|---|---|
| 곡선 | 파란색 | 왼쪽 차선 EMA 다항식 |
| 곡선 | 주황색 | 오른쪽 차선 EMA 다항식 |
| 곡선 | **노란색** | **차량이 실제 추종하는 중심 경로** |
| 경로 | **민트색** | center_coef 기반 추종 경로선 + 웨이포인트 |
| 얇은 곡선 | 어두운 파랑·주황 | EMA 전 raw 피팅 |
| 점 | 회색 | 행별로 발견된 모든 연속 run 후보 |
| 점 | 자홍색 | 추적 또는 residual 검사에서 제거된 후보 |
| 점 | 연한 파랑·주황 | 최종 피팅에 사용된 inlier |
| 화살표 | 하늘색 | 최하단에서의 헤딩 방향 |
| 가로선 | 초록 | 실제 scan 시작 행 |
| 세로선 | 빨간색 | 이미지 중앙 기준선 |
| 세로선 | 회색 | raw center_x |
| 세로선 | 초록색 | EMA 평활화된 중심 |

**HUD (좌상단):**

```
off:0.123  epsi:3.2deg          ← 평활 오프셋 / 헤딩 오차
L:15 ACCEPT e:2.1               ← 왼쪽 inlier 수 / 상태 / residual
R:12 ACCEPT e:3.4               ← 오른쪽
CENTER:BOTH                     ← 중심 경로 모드
LANE:VALID                      ← 초록: 제어 가능 / 주황: REJECT {원인}
```

`LANE:VALID` → `/lane_valid = True`, 제어 가능  
`LANE:REJECT epsi:22.3` → 헤딩각 초과로 이 프레임 offset 발행 차단

### show_window=true (OpenCV 창)

왼쪽: 원본 카메라 + 노란 BEV 사다리꼴, 오른쪽: BEV 결과

| HUD | 색상 | 내용 |
|---|---|---|
| `max:X.XX  n:Y` | 하늘색 | 최대 앵커 신뢰도 / 통과 앵커 수 |
| `off:X.XXX  VALID/REJECT` | 초록/주황 | 평활 오프셋과 유효성 상태 |
| `epsi:X.Xdeg` | 황색 | 헤딩 오차 |
| `R=X.Xm` 또는 `R=∞(직선)` | 민트 | 곡률반경 |

**초록 경로선이 빨간 기준선보다:**
- 오른쪽: offset > 0 → 차량이 차선 기준 왼쪽 → 우회전 보정
- 왼쪽: offset < 0 → 차량이 차선 기준 오른쪽 → 좌회전 보정

좌우 차선 상태는 `ACCEPT`, `HOLD/...`, `FEW`, `SPAN`, `OUTLIER`,
`SLOPE`, `JUMP`, `WIDTH`, `SINGLE`, `MISS`로 표시됩니다.

---

## 6. 모델 파일

경로: `src/camera_stack/models/`

| 파일 | 모델 | 설명 |
|---|---|---|
| `best_openvino_2022.1_6shave_1.blob` | YOLOv8s-seg | 차선 인식 주력 blob — **현재 사용 중 (기본값)** |
| `traffic_light_640x224_6shave.blob` | YOLOv8n-detect | 신호등 인식 blob |
| `lane_seg_640x224_6shave.blob` | YOLOv8n-seg | 추가학습/경량 모델 후보 (보류) |
| `best_openvino_2022.1_6shave.blob` | YOLOv8s-seg | 구버전 blob (백업) |

추가학습 후보 모델 (`lane_seg_640x224_6shave.blob`) 변환 정보:

| 항목 | 내용 |
|---|---|
| 원본 | `lane_seg_640x224/weights/best.pt` |
| 아키텍처 | YOLOv8n-seg (nano), 3.26M params, 11.5 GFLOPs |
| 입력 | 640 × 224 (W × H) |
| 출력 | `(37, 2940)` 앵커 + `(32, 56, 160)` 마스크 프로토타입 |
| 변환 | pt → ONNX (opset 12) → MyriadX blob (FP16, 6 shaves) |

현재는 추가학습 전까지 `best_openvino_2022.1_6shave_1.blob`을 기본값으로 사용합니다.

### 새 모델 배포 절차

```bash
# 1. best.pt를 src/camera_stack/models/lane_seg_640x224/weights/ 에 복사

# 2. ONNX 변환
cd ~/capstone_ws/src/camera_stack/models/lane_seg_640x224/weights
python3 -c "
from ultralytics import YOLO
YOLO('best.pt').export(format='onnx', imgsz=[224, 640], opset=12, simplify=True)
"

# 3. blob 변환
python3 -c "
import blobconverter, shutil
blob = blobconverter.from_onnx(
    model='best.onnx',
    data_type='FP16',
    shaves=6,
    use_cache=False,
    output_dir='/tmp/blob_out',
)
shutil.copy(blob, '../../lane_seg_640x224_6shave.blob')
print('완료:', blob)
"

# 4. 이 후보 모델을 테스트하려면 blob_path로 명시 지정한다.
```

---

## 7. 데이터 수집

차선 인식이 실패하는 구간의 이미지를 수집하여 추가 학습에 사용합니다.

### 방법 1 — 주행 중 's' 키 캡처 (lane_detector_node 내장)

`show_window:=true` 로 실행 중 OpenCV 창에 포커스를 두고 **`s` 키**를 누르면
그 시점의 raw 카메라 원본 프레임이 PNG로 저장됩니다.

| 키 | 동작 |
|---|---|
| `s` | 현재 프레임 raw 이미지 저장 |
| `ESC` | 노드 종료 |

**저장 파일 정보:**

| 항목 | 내용 |
|---|---|
| 형식 | PNG (무손실) |
| 내용 | 처리 전 raw 카메라 원본 (오버레이 없음) |
| 해상도 | 640 × 224 |
| 파일명 | `lane_YYYYMMDD_HHMMSS_밀리초_순번.png` |
| 저장 위치 | `capture_dir` 파라미터 (기본: `~/lane_captures`) |

### 방법 2 — capture.py 독립 실행 (자동 연속 수집)

경로: `src/camera_stack/tools/capture.py`

```bash
# 기본 (1초 간격 자동 저장)
QT_QPA_PLATFORM=xcb python3 src/camera_stack/tools/capture.py

# 저장 경로·간격 지정
QT_QPA_PLATFORM=xcb python3 src/camera_stack/tools/capture.py \
  --out ~/lane_images \
  --interval 1.5
```

| 키 | 동작 |
|---|---|
| `s` | 자동 저장 시작 / 정지 |
| `Space` | 수동 1장 저장 |
| `ESC` / `q` | 종료 |

### 추가 학습 흐름

```
수집한 PNG 이미지
  └→ 라벨링 (Roboflow, LabelImg 등)
       └→ YOLOv8-seg 추가 학습 (fine-tuning)
            └→ ONNX 변환 → blob 변환 → blob_path 교체
```

---

## 8. 실행 명령어

### 8-1. 빌드

```bash
cd ~/capstone_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select camera_stack --symlink-install
source install/setup.bash
```

`--symlink-install` 옵션을 사용하면 `.py` 파일 수정 시 재빌드 없이 반영됩니다.

### 8-2. 기본 실행 (디버그 창 없음)

`blob_path`를 생략하면 `best_openvino_2022.1_6shave_1.blob`이 자동으로 사용됩니다.

```bash
ros2 run camera_stack lane_detector_node
```

다른 blob을 명시적으로 지정할 때:

```bash
ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p blob_path:=/home/hyeonjun/capstone_ws/src/camera_stack/models/best_openvino_2022.1_6shave_1.blob
```

### 8-3. 디버그 창 포함 실행 (현재 확정 BEV 파라미터)

```bash
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p show_window:=true \
  -p conf_thresh:=0.65 \
  -p use_bev:=true \
  -p bev_src_top_y_ratio:=0.620 \
  -p bev_src_top_left_ratio:=0.320 \
  -p bev_src_top_right_ratio:=0.680 \
  -p bev_src_bottom_left_ratio:=0.050 \
  -p bev_src_bottom_right_ratio:=0.950 \
  -p bev_edge_margin_px:=15 \
  -p max_run_width:=45 \
  -p max_valid_epsi_deg:=35.0 \
  -p max_valid_path_span_px:=180.0 \
  -p poly_reject_delta:=50.0 \
  -p single_lane_path_start_ratio:=0.55 \
  -p control_row_ratio:=0.72 \
  -p hold_decay:=0.97 \
  -p lane_width_top_min_px:=220.0 \
  -p lane_width_top_max_px:=430.0 \
  -p lane_width_mid_min_px:=320.0 \
  -p lane_width_mid_max_px:=460.0 \
  -p lane_width_bot_min_px:=360.0 \
  -p lane_width_bot_max_px:=540.0 \
  -p half_width_update_max_epsi_deg:=15.0
```

### 8-4. 진단 CSV 로깅 포함 실행

```bash
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p show_window:=true \
  -p log_diag:=true \
  -p log_path:=true \
  -p xm_per_pix:=0.002222 \
  -p capture_dir:=/home/hyeonjun/lane_captures \
  -p conf_thresh:=0.65 \
  -p bev_edge_margin_px:=15 \
  -p max_valid_epsi_deg:=35.0 \
  -p max_valid_path_span_px:=180.0 \
  -p poly_reject_delta:=50.0 \
  -p single_lane_path_start_ratio:=0.55 \
  -p control_row_ratio:=0.72 \
  -p hold_decay:=0.97 \
  -p lane_width_top_min_px:=220.0 \
  -p lane_width_top_max_px:=430.0 \
  -p lane_width_mid_min_px:=320.0 \
  -p lane_width_mid_max_px:=460.0 \
  -p lane_width_bot_min_px:=360.0 \
  -p lane_width_bot_max_px:=540.0 \
  -p half_width_update_max_epsi_deg:=15.0 \
  -p bev_src_top_y_ratio:=0.620 \
  -p bev_src_top_left_ratio:=0.320 \
  -p bev_src_top_right_ratio:=0.680 \
  -p bev_src_bottom_left_ratio:=0.050 \
  -p bev_src_bottom_right_ratio:=0.950
```

CSV는 `/home/hyeonjun/lane_captures/` 아래에 자동 생성됩니다:
- `lane_diag_YYYYMMDD_HHMMSS.csv` — 47컬럼 프레임별 진단
- `lane_path_YYYYMMDD_HHMMSS.csv` — 경로 웨이포인트 좌표

### 8-5. lane_recovery_node 실행

```bash
ros2 run camera_stack lane_recovery_node \
  --ros-args \
  -p base_speed:=0.15 \
  -p k_steer:=0.5 \
  -p dead_zone:=0.05 \
  -p max_wz:=0.5
```

### 8-6. 토픽 확인

```bash
# 오프셋 값 확인
ros2 topic echo /lane_offset

# 유효성 확인
ros2 topic echo /lane_valid

# 곡률 확인
ros2 topic echo /lane_curvature

# 오프셋 발행 주기 확인
ros2 topic hz /lane_offset

# 디버그 이미지 확인 (별도 터미널)
ros2 run rqt_image_view rqt_image_view /lane_debug

# 현재 적용된 파라미터 확인
ros2 param dump /lane_detector_node
```

---

## 9. 파라미터 튜닝 가이드

### 노이즈(벽, 반사)가 많이 잡힐 때

```
conf_thresh 올리기:       0.65 → 0.75 → 0.85
roi_start_ratio 올리기:   0.55 → 0.60 → 0.65
max_blob_ratio 낮추기:    0.15 → 0.10
min_aspect_ratio 올리기:  1.5  → 3.0
max_run_width 낮추기:     45   → 30      ← NEW: 두꺼운 run 더 엄격하게 차단
bev_edge_margin_px 올리기: 15  → 25 → 40  ← BEV 외곽 더 많이 제거
```

### 차선이 자꾸 놓칠 때

```
conf_thresh 낮추기:    0.65 → 0.60 → 0.55
mask_thresh 낮추기:    0.45 → 0.35 → 0.30
poly_hold_sec 늘리기:  0.8  → 1.0           (짧은 미감지만 완충)
hold_sec 늘리기:       1.5  → 2.0  → 3.0   (오프셋 발행 유지 연장)
min_poly_pts 낮추기:   8    → 6   (노이즈가 늘어나는지 반드시 확인)
max_run_width 올리기:  45   → 60      ← NEW: 차선이 두꺼워 걸리면
```

### LANE:REJECT가 자주 발생할 때

```
max_valid_epsi_deg 올리기:       35 → 45   ← 헤딩각 거부 완화
max_valid_path_span_px 올리기:  180 → 220  ← 경로 곡률 거부 완화
max_run_width 올리기:            45 → 60   ← run 폭 거부 완화
```

### 경로(노란선)가 이상하게 꺾이거나 C자 모양으로 그려질 때

```
poly_degree=2로 돼 있으면 1로 변경   (2차 항이 외삽 구간에서 폭발적으로 발산)
epsi_weight=0.0으로 설정             (헤딩 오차가 과도하면 offset을 크게 왜곡)
```

### 경로(경로선)가 흔들릴 때

```
poly_alpha 낮추기:    0.25 → 0.15 → 0.1  (다항식 EMA를 더 느리게 → 경로가 부드러워짐)
ema_alpha 낮추기:     0.4  → 0.2  → 0.1  (오프셋 scalar를 추가 평활화)
epsi_weight 낮추기:   0.1  → 0.05 → 0.0  (헤딩 오차 기여를 줄임)
```

### 조향이 차선 방향에 늦게 반응할 때 (커브에서 느릴 때)

```
epsi_weight 올리기:   0.3 → 0.5
k_steer 올리기:       0.5 → 0.7
```

### 한쪽 차선만 있는 구간에서 경로가 나오지 않을 때

양쪽 차선을 먼저 검출할 필요는 없습니다. HUD에서 검출된 쪽이 `SINGLE`,
중심 상태가 `CENTER:LEFT/DEFAULT` 또는 `CENTER:RIGHT/DEFAULT`인지 확인합니다.

노란 경로가 실제 도로 중심보다 차선에 가까우면 해당 위치의 반폭을 늘리고,
도로 중심을 지나 반대편으로 치우치면 줄입니다.

```
BEV 스캔 시작선 위치 조정: single_lane_half_width_top
BEV 영상 최하단 위치 조정: single_lane_half_width_bottom
```

### BEV에서 차선이 수직이 아니거나 폭이 크게 달라질 때

원본 디버그 영상을 기준으로 사다리꼴 네 점을 차선 안쪽이 아니라 실제 좌우
차선 중심에 맞춥니다.

```text
위쪽 차선 위치: bev_src_top_left_ratio / bev_src_top_right_ratio
아래쪽 차선 위치: bev_src_bottom_left_ratio / bev_src_bottom_right_ratio
사다리꼴 시작 높이: bev_src_top_y_ratio
```

BEV에서 좌우 차선이 위로 갈수록 벌어지면 원본 상단 좌우 점 간격을 줄이고,
위로 갈수록 모이면 상단 좌우 점 간격을 늘립니다.

### 조향이 진동(좌우 흔들림)할 때

```
dead_zone 올리기:     0.05 → 0.08 → 0.10  (≈ ±16px → ±26px → ±32px)
k_steer 낮추기:       0.5  → 0.3
max_wz 낮추기:        0.5  → 0.3
```

---

## 10. 알려진 이슈 및 해결책

### OpenCV 창이 열리지 않음 (Wayland/GNOME)

```bash
# 실행 전 환경변수 설정
export QT_QPA_PLATFORM=xcb
# 또는 명령어 앞에 추가
QT_QPA_PLATFORM=xcb ros2 run ...
```

### depthai 디바이스가 이미 사용 중

```bash
# 다른 프로세스 확인 및 종료
sudo fuser -k /dev/bus/usb/*/*
# 또는
ps aux | grep dai
kill -9 <PID>
```

### 카메라 연결 후 화면이 검정

depthai 카메라는 시작 후 2~3초 워밍업이 필요합니다.
연결 후 바로 프레임을 읽으면 검정 화면이 나옵니다.

### 's' 키를 눌러도 저장 안 됨

- `show_window:=true` 옵션이 있는지 확인 (창이 없으면 키 입력 불가)
- OpenCV 창에 마우스 포커스가 있는지 확인
- 터미널에 `[캡처]` 로그가 찍히는지 확인

### blob 출력 크기 불일치 오류

```
출력 크기 불일치: output0=... output1=...
```

ONNX 변환 시 `imgsz`가 맞지 않으면 발생합니다.

```bash
# 올바른 변환 명령어
yolo export model=best.pt format=onnx imgsz=224,640 opset=12 simplify=True
#                                              ↑높이, 너비 순서 주의
```

### LANE:REJECT가 계속 발생하고 offset이 발행되지 않음

진단 CSV의 `reject_reason` 컬럼을 확인합니다:

- `epsi:25.3` → `max_valid_epsi_deg` 를 올리거나 카메라 장착 각도 재확인
- `path_span:180` → `max_valid_path_span_px` 를 올리거나 BEV 파라미터 재조정
- `run_w:52` → `max_run_width` 를 올리거나 벽·반사 차단 파라미터 강화

### declare_parameter 이름 오타 시 노드 즉시 종료

파라미터 선언 이름과 get 이름이 다르면 ROS2가 예외를 던지고 노드가 죽습니다.

```python
# 잘못된 예
self.declare_parameter('_start_ratio', 0.60)
self.roi_start_ratio = self.get_parameter('roi_start_ratio')  # → 오류!

# 올바른 예
self.declare_parameter('roi_start_ratio', 0.55)
self.roi_start_ratio = self.get_parameter('roi_start_ratio')
```
