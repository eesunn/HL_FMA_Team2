# lane_detector_node 실차 테스트 가이드

OAK-D + YOLOv8-seg 기반 차선 인식 노드의 실차 테스트 절차 및 명령어 정리.

---

## 목차

1. [빌드](#1-빌드)
2. [STEP 1 — 차선 인식 노드 단독 실행](#2-step-1--차선-인식-노드-단독-실행-조향-없음)
3. [STEP 2 — 디버그 토픽 모니터링](#3-step-2--디버그-토픽-모니터링)
4. [STEP 3 — 조향 부호 확인](#4-step-3--조향-부호-확인-차량-이동-전-필수)
5. [STEP 4 — lane_recovery 포함 전체 실행](#5-step-4--lane_recovery-포함-전체-실행)
6. [STEP 5 — 시나리오별 테스트](#6-step-5--시나리오별-테스트)
7. [STEP 6 — 런타임 파라미터 조정](#7-step-6--런타임-파라미터-조정)
8. [STEP 7 — 문제 상황별 즉시 확인 명령어](#8-step-7--문제-상황별-즉시-확인-명령어)
9. [테스트 전 핵심 체크리스트](#9-테스트-전-핵심-체크리스트)

---

## 1. 빌드

코드 수정 후 또는 최초 1회 실행한다.

```bash
cd ~/capstone_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select camera_stack --symlink-install
source install/setup.bash
```

> `--symlink-install` 옵션 사용 시 `.py` 파일 수정은 재빌드 없이 바로 반영된다.
> `setup.py`를 수정했을 때만 재빌드가 필요하다.

---

## 2. STEP 1 — 차선 인식 노드 단독 실행 (조향 없음)

**가장 먼저 실행한다. lane_recovery 없이 인식만 확인.**

```bash
cd ~/capstone_ws
source install/setup.bash

QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p blob_path:=/home/hyeonjun/capstone_ws/src/camera_stack/models/best_openvino_2022.1_6shave_1.blob \
  -p show_window:=true \
  -p conf_thresh:=0.65 \
  -p use_bev:=true \
  -p poly_degree:=1 \
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
  -p half_width_update_max_epsi_deg:=15.0
```

**사용 가능한 blob 파일:**

| 파일 | 설명 |
|---|---|
| `lane_seg_640x224_6shave.blob` | 추가학습 후보 모델 |
| `best_openvino_2022.1_6shave_1.blob` | 현재 기본 사용 모델 |
| `best_openvino_2022.1_6shave.blob` | 구버전 백업 |
| `best_s_640x224_6shave.blob` | 구버전 백업 |

경로: `~/capstone_ws/src/camera_stack/models/`

**이 단계에서 확인할 것:**

- OpenCV 창이 뜨는지
- 차선 위에 초록 반투명 마스크가 덮이는지
- 상단 HUD `max:` 값이 `conf_thresh`(기본 0.65) 이상인지
- `off:` 값이 차량 위치에 따라 변하는지

**OpenCV 창 키 조작:**

| 키 | 동작 |
|---|---|
| `s` | 현재 프레임 raw 이미지 저장 (`~/lane_captures/`) |
| `ESC` | 노드 종료 |

---

## 3. STEP 2 — 디버그 토픽 모니터링

STEP 1과 병렬로 다른 터미널에서 실행한다.

**터미널 2 — offset 값 숫자로 보기:**
```bash
source ~/capstone_ws/install/setup.bash
ros2 topic echo /lane_offset
```

**터미널 3 — 디버그 이미지 보기:**
```bash
source ~/capstone_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /lane_debug
```

**터미널 4 — 토픽 발행 주파수 확인:**
```bash
source ~/capstone_ws/install/setup.bash
ros2 topic hz /lane_offset
# 목표: ~30Hz
```

**`/lane_debug` 이미지 요소 설명:**

| 색상 | 요소 | 의미 |
|---|---|---|
| 파란 곡선 | 왼쪽 차선 다항식 | smooth EMA 결과 |
| 주황 곡선 | 오른쪽 차선 다항식 | smooth EMA 결과 |
| 노란 곡선 | 중심 경로 다항식 | 차량이 추종하는 경로 |
| 빨간 세로선 | 이미지 중앙 | 기준선 |
| 회색 세로선 | raw center_x | EMA 이전 가중 평균 결과 |
| 초록 세로선 | smooth center | EMA 적용 후 중심 |

---

## 4. STEP 3 — 조향 부호 확인 (차량 이동 전 필수)

**차량을 손으로 움직여서 부호를 먼저 확인한다. 이 단계를 건너뛰면 안 된다.**

```bash
source ~/capstone_ws/install/setup.bash
ros2 topic echo /lane_offset
```

| 상황 | 기대값 | 의미 |
|---|---|---|
| 차선 중앙에 놓음 | `-0.05 ~ +0.05` | 정상 |
| 차량을 오른쪽으로 옮김 | `+` 증가 | 오른쪽 이탈 |
| 차량을 왼쪽으로 옮김 | `-` 증가 | 왼쪽 이탈 |

> **부호가 반대이면 차량이 이탈 방향으로 가속한다. 절대 주행하면 안 된다.**

---

## 5. STEP 4 — lane_recovery 포함 전체 실행

**터미널 1 — lane_detector_node:**
```bash
source ~/capstone_ws/install/setup.bash

QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p blob_path:=/home/hyeonjun/capstone_ws/src/camera_stack/models/best_openvino_2022.1_6shave_1.blob \
  -p show_window:=true \
  -p conf_thresh:=0.65 \
  -p use_bev:=true \
  -p poly_degree:=1 \
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
  -p half_width_update_max_epsi_deg:=15.0
```

**터미널 2 — lane_recovery_node:**
```bash
source ~/capstone_ws/install/setup.bash

ros2 run camera_stack lane_recovery_node \
  --ros-args \
  -p base_speed:=0.10 \
  -p k_steer:=0.4 \
  -p dead_zone:=0.05 \
  -p max_wz:=0.4 \
  -p cmd_rate_hz:=20.0 \
  -p offset_timeout_sec:=0.35 \
  -p require_lane_valid:=true \
  -p log_csv:=true \
  -p capture_dir:=/home/hyeonjun/capstone_ws/src/camera_stack
```

> 처음엔 `base_speed:=0.10` 저속으로 시작한다. 안정 확인 후 0.15로 올린다.

**실차 CAN 연결 테스트에서는 lane_recovery 출력 토픽을 `/cmd_vel`로 바꾼다:**
```bash
ros2 run camera_stack lane_recovery_node \
  --ros-args \
  -p output_topic:=/cmd_vel \
  -p base_speed:=0.08 \
  -p k_steer:=0.35 \
  -p dead_zone:=0.05 \
  -p max_wz:=0.30 \
  -p cmd_rate_hz:=20.0 \
  -p offset_timeout_sec:=0.35 \
  -p require_lane_valid:=true \
  -p log_csv:=true \
  -p capture_dir:=/home/hyeonjun/capstone_ws/src/camera_stack
```

이때 하위 제어 경로는 다음과 같다:
`lane_recovery_node → /cmd_vel → mecanum_bridge_node → /wheel_targets → can_bridge_node → CAN 0x300 → TC275`

`log_csv:=true`이면 같은 폴더에 `lane_recovery_YYYYMMDD_HHMMSS.csv`가 생성된다.

**터미널 2-1 — mecanum_bridge_node:**
```bash
source ~/capstone_ws/install/setup.bash

ros2 run control_stack mecanum_bridge_node \
  --ros-args \
  -p use_strafe:=false
```

**터미널 2-2 — can_bridge_node:**
```bash
source ~/capstone_ws/install/setup.bash

ros2 run control_stack can_bridge_node \
  --ros-args \
  -p can_interface:=can0 \
  -p initial_ctrl_mode:=3 \
  -p ctrl_mode_resend_hz:=1.0 \
  -p use_strafe:=false
```

**터미널 3 — lane_recovery 수동 활성화/비활성화:**
```bash
source ~/capstone_ws/install/setup.bash

# 차선 추종 시작
ros2 topic pub /lane_recovery_enable std_msgs/msg/Bool "{data: true}" -1

# 차선 추종 정지
ros2 topic pub /lane_recovery_enable std_msgs/msg/Bool "{data: false}" -1
```

**터미널 4 — cmd_vel 모니터링:**
```bash
source ~/capstone_ws/install/setup.bash
ros2 topic echo /cmd_vel_recovery

# 실차 CAN 연결 테스트에서는 이것도 확인
ros2 topic echo /cmd_vel
ros2 topic echo /wheel_targets
```

---

## 6. STEP 5 — 시나리오별 테스트

### [1단계] 직진 테스트 — 가장 먼저

```bash
ros2 topic echo /lane_offset
```

확인 항목:
- 차선 중앙 → offset이 ±0.05 이내인지
- 좌우로 흔들리지 않는지 (흔들리면 `ema_alpha` 낮추기)
- NN 로그 `[NN] 최대신뢰도` 가 `conf_thresh`(기본 0.65) 이상인지

---

### [2단계] 한쪽 차선만 있는 구간

`/lane_debug` 에서 확인:
- 파란선(왼쪽)만 또는 주황선(오른쪽)만 표시되는지
- 노란선(center)이 차선으로부터 `road_half_w` 만큼 이동했는지

`road_half_w` 보정 방법:
```
1. show_window=true 상태에서 양쪽 차선이 모두 보이는 순간 확인
2. road_half_w = (오른쪽 차선 X픽셀 - 왼쪽 차선 X픽셀) / 2
   예: 왼쪽=200px, 오른쪽=440px → road_half_w = 120
```

```bash
# 런타임 변경
ros2 param set /lane_detector_node road_half_w 120
```

---

### [3단계] 차선 순간 미검출 시뮬레이션

손으로 카메라 앞을 잠깐 가린 뒤 터미널에서 확인:

```bash
ros2 topic echo /lane_offset
```

확인 항목:
- 가리는 동안 로그에 `HOLD` 가 찍히는지
- `hold_sec` (기본 1.5초) 이후 발행이 멈추는지
- 손 치운 후 `DETECT` 로 다시 전환되는지

---

### [4단계] 반사광/노이즈 구간

`show_window=true` 에서 확인:
- 시안색 사각형(앵커 박스)이 차선이 아닌 곳에 많이 잡히는지
- blob 필터 후 `/lane_debug` 에서 제거됐는지

잡힌다면:
```bash
ros2 param set /lane_detector_node conf_thresh 0.80
ros2 param set /lane_detector_node min_aspect_ratio 2.5
```

---

### [5단계] 조향 부호 실주행 확인

- 오른쪽 이탈 시 `cmd_vel_recovery.angular.z < 0` (우회전)인지
- 왼쪽 이탈 시 `cmd_vel_recovery.angular.z > 0` (좌회전)인지

```bash
ros2 topic echo /cmd_vel_recovery
```

---

## 7. STEP 6 — 런타임 파라미터 조정

노드 재시작 없이 파라미터를 변경할 수 있다.

```bash
# 단일 파라미터 변경
ros2 param set /lane_detector_node conf_thresh 0.65
ros2 param set /lane_detector_node road_half_w 100
ros2 param set /lane_detector_node hold_sec 0.5
ros2 param set /lane_detector_node poly_alpha 0.15
ros2 param set /lane_detector_node ema_alpha 0.3

# 현재 파라미터 전체 확인
ros2 param list /lane_detector_node

# 특정 파라미터 값 확인
ros2 param get /lane_detector_node conf_thresh
```

> `blob_path`, `show_window`는 런타임 변경이 적용되지 않는다. 재시작 필요.

---

### 증상별 파라미터 조정 방향

**차선 인식이 안 될 때:**
```bash
ros2 param set /lane_detector_node conf_thresh 0.55
ros2 param set /lane_detector_node mask_thresh 0.35
ros2 param set /lane_detector_node hold_sec 2.0
ros2 param set /lane_detector_node poly_hold_sec 3.0
```

**노이즈/벽이 차선으로 잡힐 때:**
```bash
ros2 param set /lane_detector_node conf_thresh 0.85
ros2 param set /lane_detector_node min_aspect_ratio 2.5
ros2 param set /lane_detector_node max_blob_ratio 0.10
```

**경로(노란선)가 흔들릴 때:**
```bash
ros2 param set /lane_detector_node poly_alpha 0.15
ros2 param set /lane_detector_node ema_alpha 0.2
```

**조향이 코너에서 느릴 때:**
```bash
ros2 param set /lane_recovery_node k_steer 0.7
ros2 param set /lane_detector_node ema_alpha 0.6
```

**조향이 좌우로 진동할 때:**
```bash
ros2 param set /lane_recovery_node dead_zone 0.08
ros2 param set /lane_recovery_node k_steer 0.3
ros2 param set /lane_recovery_node max_wz 0.3
```

---

## 8. STEP 7 — 문제 상황별 즉시 확인 명령어

**노드 및 토픽 상태 확인:**
```bash
# 노드가 살아있는지
ros2 node list

# 토픽 목록 확인
ros2 topic list

# 토픽 발행 주파수
ros2 topic hz /lane_offset
ros2 topic hz /cmd_vel_recovery
```

**카메라 연결 확인:**
```bash
# OAK-D USB 인식 여부
lsusb | grep -i luxonis

# depthai 프로세스 충돌 시 강제 해제
sudo fuser -k /dev/bus/usb/*/*
```

**depthai 프로세스 충돌 시:**
```bash
ps aux | grep dai
kill -9 <PID>
```

**로그 레벨 올려서 상세 확인:**
```bash
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p blob_path:=/home/hyeonjun/capstone_ws/src/camera_stack/models/best_openvino_2022.1_6shave_1.blob \
  --log-level DEBUG
```

**OpenCV 창이 안 열릴 때 (Wayland 환경):**
```bash
export QT_QPA_PLATFORM=xcb
# 또는 명령어 앞에 붙이기
QT_QPA_PLATFORM=xcb ros2 run ...
```

**blob 출력 크기 불일치 오류 발생 시:**
```
출력 크기 불일치: output0=... output1=...
```
→ 사용 중인 blob 파일이 640×224 모델인지 확인. 현재 기본은 `best_openvino_2022.1_6shave_1.blob`.

---

## 9. 테스트 전 핵심 체크리스트

```
[ ] OAK-D USB 연결 확인 (lsusb | grep -i luxonis)
[ ] source ~/capstone_ws/install/setup.bash 실행 여부 확인
[ ] blob 파일 경로 확인 (ls ~/capstone_ws/src/camera_stack/models/)
[ ] show_window:=true 로 OpenCV 창 뜨는지 확인
[ ] offset 부호 확인 (오른쪽 이탈 → +, 왼쪽 이탈 → -)
[ ] lane_recovery_enable=false 상태에서 시작
[ ] 처음엔 base_speed:=0.10 저속으로 시작
[ ] 안정 확인 후 base_speed:=0.15로 증가
[ ] 긴급 정지 수단 준비 (CAN 0x301 Mode=0 또는 전원 차단)
```

---

## 참고: 관련 파일 경로

| 항목 | 경로 |
|---|---|
| 차선 인식 노드 | `src/camera_stack/camera_stack/lane_detector_node.py` |
| 차선 추종 노드 | `src/camera_stack/camera_stack/lane_recovery_node.py` |
| blob 모델 | `src/camera_stack/models/` |
| 이미지 캡처 저장 | `~/lane_captures/` |
| 노드 상세 가이드 | `src/camera_stack/camera_stack/LANE_DETECTOR_GUIDE.md` |
| lane_recovery 가이드 | `src/camera_stack/LANE_RECOVERY_GUIDE.md` |
