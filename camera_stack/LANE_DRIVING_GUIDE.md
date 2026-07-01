# 차선 인식 기반 주행 실행 가이드

카메라로 차선을 인식하고 TC275 하위 제어기까지 명령을 보내 실제 주행하는 전체 절차를 정리한 문서입니다.

> **최종 갱신**: 2026-06-21 (신호등 박스 크기 기반 정지 방식으로 전환)

---

## 목차

1. [전체 제어 흐름](#1-전체-제어-흐름)
2. [알려진 문제점](#2-알려진-문제점)
3. [사전 준비](#3-사전-준비)
4. [실행 순서 (터미널별)](#4-실행-순서-터미널별)
5. [정지 방법](#5-정지-방법)
6. [상태 모니터링](#6-상태-모니터링)
7. [주행 전 체크리스트](#7-주행-전-체크리스트)
8. [CSV 로그 파일 분석](#8-csv-로그-파일-분석)
9. [파라미터 빠른 조정](#9-파라미터-빠른-조정)
10. [검증 시나리오](#10-검증-시나리오)
11. [코드 수정 이력](#11-코드-수정-이력)

---

## 1. 전체 제어 흐름

```
OAK-D (IMX219-120, 640×224, ~27 fps)
  └─ lane_detector_node
        ├─ /lane_offset           (Float32, [-1,+1], TRANSIENT_LOCAL)
        ├─ /lane_valid            (Bool, TRANSIENT_LOCAL)
        ├─ /lane_debug            (Image, 디버그용)
        └─ /camera/image_raw      (Image, 신호등 노드에 전달)
                                        │
                               traffic_light_detector_node (PC 추론)
                                        ├─ /traffic_light          (String: RED/GREEN/NONE)
                                        └─ /traffic_light_box_area (Float32: EMA 박스 면적 px²)
              │                                │                │
              ▼                                ▼                ▼
        lane_recovery_node  ◄─── /traffic_light + /traffic_light_box_area
              │  RED AND box_area >= threshold → 정지
              │  RED AND box_area < threshold  → 주행 (아직 신호등에서 멀리 있음)
              │  RED 해제 → 차선 추종 재개
              │  output_topic := /cmd_vel
              ▼
        /cmd_vel  (geometry_msgs/Twist)
              │
              ▼
        mecanum_bridge_node
              │  /wheel_targets  [A,B,C,D] km/h
              ▼
        can_bridge_node
              │  0x300 SpeedCommand (CAN 500 kbps)
              │  0x301 ControlMode = 3 (ROS2 Autonomous)
              ▼
        TC275 하위 제어기 → 4채널 PI 속도 제어 → 모터
```

> **현재 상태**: `twist_mux`가 설치되어 있지 않습니다. 단독 차선 주행 테스트 시
> `lane_recovery_node`의 `output_topic:=/cmd_vel`로 직접 연결합니다.

---

## 2. 알려진 문제점

### 문제 1 — twist_mux 미설치

**증상**: `lane_recovery_node`가 `/cmd_vel_recovery`를 발행해도 차량이 움직이지 않음.  
**해결**: 실행 시 `output_topic:=/cmd_vel` 파라미터로 우회.

### 문제 2 — auto_enable 기본값 False

**증상**: 노드 실행 후 차량이 움직이지 않음.  
**해결**: `auto_enable:=true`로 실행.

### 문제 3 — CAN 인터페이스 ✅ udev 자동 설정 완료

PCAN-USB를 꽂으면 `can0`가 500 kbps로 자동 UP됩니다. 수동 명령 불필요.

### 문제 4 — box_area_threshold 보정 필요

**내용**: 기본값 2000 px²는 환경에 따라 다름. 처음 테스트 시 `/traffic_light_box_area` 토픽을 모니터링해 정지하고 싶은 거리에서의 EMA 값을 확인한 후 파라미터를 조정하세요.  
**보정 방법**: [§10-C 박스 크기 임계값 보정](#보정-방법) 참조.

---

## 3. 사전 준비

### 3-1. CAN 인터페이스 설정

> **2026-06-20 이후**: udev 자동 설정 완료. PCAN-USB 연결 시 `can0`가 자동 UP.

```bash
# 확인 (PCAN-USB 연결 후 약 1초 뒤)
ip link show can0
# 정상: state UP

# 자동 실패 시 수동
sudo ip link set can0 type can bitrate 500000 && sudo ip link set can0 up
```

### 3-2. 워크스페이스 빌드

```bash
cd ~/capstone_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select camera_stack control_stack --symlink-install --base-paths src
source install/setup.bash
```

> `--symlink-install` 사용 시 `.py` 파일 수정은 재빌드 없이 반영됩니다.  
> `--base-paths src` 는 중복 패키지 감지 오류 방지용으로 필수입니다.

### 3-3. 신호등 모델 파일 확인

```bash
# traffic_light_best.pt 파일 존재 여부 확인
ls -lh ~/capstone_ws/src/camera_stack/models/traffic_light_best.pt

# 없으면 spare/ 에서 복사
cp ~/capstone_ws/src/camera_stack/models/spare/traffic_light_best.pt \
   ~/capstone_ws/src/camera_stack/models/traffic_light_best.pt
```

---

## 4. 실행 순서 (터미널별)

총 **5개 터미널**. 순서를 지켜야 합니다.

| 목적 | 필요 터미널 |
|---|---|
| 차선 인식만 확인 | 터미널 3만 |
| 차선 추종 주행 (신호등 없음) | 터미널 1→2→3→4 |
| 차선 추종 + 신호등 연동 전체 검증 | 터미널 1→2→3→4→5 |

> 모든 터미널에서:
> ```bash
> cd ~/capstone_ws && source install/setup.bash
> ```

---

### 터미널 1 — CAN 브리지 (TC275 연결)

```bash
ros2 run control_stack can_bridge_node
```

**역할**: TC275에 `ControlMode=3` 송신 + `/wheel_targets` → CAN 0x300 발행

**정상 로그**:
```
CAN interface opened: can0
TX 0x301 ControlMode=3
```

---

### 터미널 2 — 메카넘 브리지 (역운동학)

```bash
ros2 run control_stack mecanum_bridge_node
```

**역할**: `/cmd_vel` (Twist) → `/wheel_targets` (A,B,C,D km/h) 변환

---

### 터미널 3 — 차선 인식 노드

```bash
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args \
  -p show_window:=true
```

**역할**: OAK-D → YOLOv8-seg 추론 → BEV → 차선 피팅 → `/lane_offset`, `/lane_valid`, `/camera/image_raw` 발행  
**CSV**: `~/capstone_ws/logs/lane_diag_YYYYMMDD_HHMMSS.csv` 자동 생성 (~27 Hz)

**정상 로그**:
```
lane_detector_node ready  blob=best_openvino_2022.1_6shave1.blob  conf=0.45  ...
[DiagLogger] 저장 시작: ~/capstone_ws/logs/lane_diag_YYYYMMDD_HHMMSS.csv
```

---

### 터미널 4 — 차선 추종 제어 노드

```bash
ros2 run camera_stack lane_recovery_node \
  --ros-args \
  -p output_topic:=/cmd_vel \
  -p auto_enable:=true \
  -p base_speed:=0.10 \
  -p box_area_threshold:=2000.0
```

**역할**: `/lane_offset` → P제어 → `/cmd_vel` 발행.  
`/traffic_light` + `/traffic_light_box_area` 수신 시 이중 조건(RED AND 박스 크기 ≥ 임계값)으로 정지.  
**CSV**: `~/capstone_ws/logs/lane_recovery_YYYYMMDD_HHMMSS.csv` 자동 생성 (20 Hz)

**정상 로그**:
```
lane_recovery_node ready  base_speed=0.1 ... use_box_area=True  box_area_threshold=2000px²
[RecoveryLog] ~/capstone_ws/logs/lane_recovery_YYYYMMDD_HHMMSS.csv
keyboard stop ready: press SPACE to stop
```

> **이 터미널에서 SPACE를 누르면 즉시 정지합니다.**

옵션:
```bash
# 신호등 연동 비활성화 (차선 주행만)
-p traffic_light_stop:=false

# 박스 크기 조건 없이 RED만으로 정지 (구형 동작)
-p use_box_area:=false
```

---

### 터미널 5 — 신호등 인식 노드 (신호등 연동 시에만)

```bash
ros2 run camera_stack traffic_light_detector_node \
  --ros-args \
  -p model_path:=$(ros2 pkg prefix camera_stack)/share/camera_stack/models/traffic_light_best.pt
```

**역할**: `/camera/image_raw` → YOLO `.pt` 추론 → voting → `/traffic_light` + `/traffic_light_box_area` 발행  
**CSV**: `~/capstone_ws/logs/traffic_light_YYYYMMDD_HHMMSS.csv` 자동 생성

**정상 로그**:
```
YOLO 모델 로드 완료: traffic_light_best.pt  classes={0: 'green light', 1: 'red light'}
traffic_light_detector_node ready  input=/camera/image_raw  conf>=0.50 ...
[TLLog] ~/capstone_ws/logs/traffic_light_YYYYMMDD_HHMMSS.csv
```

> **주의**: 터미널 3 (lane_detector_node)이 먼저 실행되어야 `/camera/image_raw`가 발행됩니다.

---

## 5. 정지 방법

### 즉시 정지 (우선순위 순)

| 방법 | 동작 |
|---|---|
| 터미널 4 **SPACE 키** | lane_recovery_node가 즉시 zero Twist 발행 (가장 빠름) |
| 터미널 4 **Ctrl+C** | zero Twist 발행 후 종료 |
| 터미널 1 **Ctrl+C** | can_bridge가 0x300=0 전송 → TC275 정지 (최후 수단) |

> 신호등 RED + 박스 크기 ≥ 임계값 감지 시: 별도 조작 없이 차량이 자동 정지됩니다.  
> 신호등이 GREEN 또는 NONE으로 바뀌면 차선 추종이 자동 재개됩니다.

### 정상 종료 순서

```
터미널 4 → Ctrl+C   (lane_recovery: shutdown zero 발행 후 CSV 저장)
터미널 5 → Ctrl+C   (traffic_light_detector: CSV 저장)
터미널 3 → ESC or Ctrl+C  (lane_detector: CSV 저장)
터미널 2 → Ctrl+C   (mecanum_bridge)
터미널 1 → Ctrl+C   (can_bridge: TC275에 0x300=0 전송)
```

> **주의**: 터미널 1을 먼저 종료하면 TC275가 마지막 속도를 최대 500ms 유지하다 자체 정지.
> 반드시 터미널 1을 **마지막**에 종료하세요.

---

## 6. 상태 모니터링

```bash
# 차선 오프셋 확인 (±1 범위)
ros2 topic echo /lane_offset

# 차선 유효성 확인
ros2 topic echo /lane_valid

# 신호등 상태 확인
ros2 topic echo /traffic_light

# EMA 박스 면적 확인 (임계값 보정에 사용)
ros2 topic echo /traffic_light_box_area

# 바퀴 속도 목표 [A,B,C,D km/h]
ros2 topic echo /wheel_targets

# 발행 주기 확인
ros2 topic hz /lane_offset
ros2 topic hz /traffic_light
ros2 topic hz /cmd_vel

# 디버그 이미지 (차선)
ros2 run rqt_image_view rqt_image_view /lane_debug

# 디버그 이미지 (신호등 — 박스 + EMA 면적 + 투표 현황 오버레이)
ros2 run rqt_image_view rqt_image_view /traffic_light_debug
```

### 신호등 수동 시뮬레이션 (박스 크기 조건 없이 동작 검증)

`use_box_area:=false`로 실행 후, `/traffic_light` 직접 발행:

```bash
# RED 발행 → 차량 즉시 정지 확인
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'RED'}"

# GREEN 발행 → 차선 추종 재개 확인
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'GREEN'}"
```

### 박스 크기 조건 수동 시뮬레이션

`use_box_area:=true` (기본값) 상태에서 두 토픽을 함께 발행:

```bash
# RED + 작은 박스 → 계속 주행
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'RED'}"
ros2 topic pub --once /traffic_light_box_area std_msgs/Float32 "{data: 500.0}"

# RED + 큰 박스 → 정지
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'RED'}"
ros2 topic pub --once /traffic_light_box_area std_msgs/Float32 "{data: 3000.0}"
```

---

## 7. 주행 전 체크리스트

```
[ ] PCAN-USB 연결 확인
[ ] TC275 전원 인가
[ ] ip link show can0 → state UP 확인
[ ] OAK-D USB 연결 확인
[ ] 워크스페이스 빌드 완료 (--symlink-install --base-paths src)
[ ] traffic_light_best.pt 파일 존재 확인 (§3-3 참조)
[ ] lane_detector_node 창에서 초록 마스크가 차선 위에 보임
[ ] ros2 topic echo /lane_valid → true 확인
[ ] ros2 topic hz /lane_offset → ~27 Hz 확인
[ ] ros2 topic echo /traffic_light_box_area → 신호등 앞에서 값 증가 확인
[ ] 주변 장애물 제거, 안전 거리 확보
[ ] 비상 정지: 터미널 4 SPACE 키 위치 확인
```

---

## 8. CSV 로그 파일 분석

주행할 때마다 `~/capstone_ws/logs/` 폴더에 CSV 파일이 자동 생성됩니다.

### 8-1. lane_diag_*.csv (lane_detector_node, ~27 Hz)

#### 핵심 컬럼

| 컬럼 | 설명 | 기준값 |
|---|---|---|
| `lane_valid` | 제어 유효 여부 (1/0) | 1 비율 ≥ 60% 목표 |
| `center_mode` | BOTH/LEFT/RIGHT/NONE/REJECT | BOTH/LEARNED 비율 높을수록 좋음 |
| `reject_reason` | 거부 원인 | 최다 원인 파악 후 파라미터 조정 |
| `epsi_deg` | 헤딩 오차 (도) | 평균 20° 이하 목표 |
| `half_width_px` | 학습된 반폭 | LEARNED 값으로 수렴 목표 |

```bash
# 최근 CSV 분석
python3 -c "
import pandas as pd
df = pd.read_csv(sorted(__import__('glob').glob(__import__('os').path.expanduser('~/capstone_ws/logs/lane_diag_*.csv')))[-1])
print('VALID 비율:',       df['lane_valid'].mean())
print('REJECT 원인 TOP5:', df['reject_reason'].value_counts().head())
print('center_mode 분포:', df['center_mode'].value_counts())
"
```

---

### 8-2. traffic_light_*.csv (traffic_light_detector_node, ~27 Hz)

#### 핵심 컬럼

| 컬럼 | 설명 | 활용 |
|---|---|---|
| `state` | voting 후 최종 상태 (RED/GREEN/NONE) | 정지 판단 기준 |
| `box_area_ema_px2` | EMA 평활화된 RED 박스 면적 (px²) | 임계값 보정에 활용 |
| `n_red_boxes` | 프레임 내 RED 박스 수 | 오탐 확인 |

```bash
# box_area EMA 통계 (RED 상태일 때)
python3 -c "
import pandas as pd
df = pd.read_csv(sorted(__import__('glob').glob(__import__('os').path.expanduser('~/capstone_ws/logs/traffic_light_*.csv')))[-1])
red = df[df['state'] == 'RED']
print('RED 프레임 수:', len(red))
print('box_area_ema 분포:')
print(red['box_area_ema_px2'].describe())
"
```

---

### 8-3. lane_recovery_*.csv (lane_recovery_node, 20 Hz)

| 컬럼 | 설명 |
|---|---|
| `reason` | `drive` / `traffic_light_red_stop` / `stale_or_missing_offset` 등 |
| `box_area_px2` | 정지 판단 시점의 EMA 박스 면적 |
| `traffic_light` | 해당 시점 신호등 상태 |

```bash
# reason 분포 및 정지 시점 box_area 확인
python3 -c "
import pandas as pd
df = pd.read_csv(sorted(__import__('glob').glob(__import__('os').path.expanduser('~/capstone_ws/logs/lane_recovery_*.csv')))[-1])
print(df['reason'].value_counts())
stops = df[df['reason'] == 'traffic_light_red_stop']
print('정지 시 box_area 통계:')
print(stops['box_area_px2'].describe())
"
```

---

## 9. 파라미터 빠른 조정

### lane_detector_node 주요 파라미터

| 파라미터 | 현재 기본값 | 조정 가이드 |
|---|---|---|
| `conf_thresh` | **0.45** | no_mask 많으면 낮추기 (단, 오탐 증가) |
| `epsi_weight` | **0.3** | 직선만 달리면 0.0도 가능 |
| `max_valid_path_span_px` | **320** | path_span reject 많으면 더 높이기 |
| `single_lane_max_center_jump` | **150** | jump reject 많으면 더 높이기 |

### lane_recovery_node 주요 파라미터

| 파라미터 | 현재 기본값 | 조정 가이드 |
|---|---|---|
| `base_speed` | **0.20** | 처음 테스트 시 0.10으로 시작 |
| `max_vx` | **0.20** | 속도 상한 (base_speed보다 작으면 의미 없음) |
| `k_steer` | 0.5 | 진동 시 낮추기, 반응 느리면 높이기 |
| `max_wz` | **0.8** | 큰 epsi 교정 필요 시 1.0 이상으로 |
| `box_area_threshold` | **2000.0** | 환경에 맞게 보정 필요 — §10-C 참조 |
| `use_box_area` | **True** | False로 설정 시 RED만으로 즉시 정지 |
| `traffic_light_stop` | **True** | False로 설정 시 신호등 무시 |

### traffic_light_detector_node 주요 파라미터

| 파라미터 | 현재 기본값 | 조정 가이드 |
|---|---|---|
| `conf_thresh` | **0.50** | 오탐 많으면 0.60으로 높이기 |
| `box_area_ema_alpha` | **0.3** | 클수록 최신 값에 빠르게 반응 (0~1) |
| `box_area_decay` | **0.9** | RED 박스 없을 때 프레임당 감쇠율 |
| `vote_buffer_size` | **10** | 클수록 voting 안정, 반응 느림 |
| `red_vote_ratio` | **0.5** | 낮추면 RED 판정 민감해짐 |

---

## 10. 검증 시나리오

### 시나리오 A — 차선 인식 단독 검증 (주행 없음)

**목적**: 카메라가 차선을 올바르게 인식하는지 확인.

```bash
# 터미널 3만 실행
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args -p show_window:=true

# 별도 터미널에서 확인
ros2 topic echo /lane_valid    # true가 나오면 차선 인식 성공
ros2 topic echo /lane_offset   # ±1 범위 값 확인
```

**판단 기준**: `lane_valid=true` 비율 ≥ 60%, `lane_offset`이 -0.3 ~ +0.3 사이.

---

### 시나리오 B — 차선 추종 주행 검증 (신호등 없음)

**목적**: 차선을 따라 주행하는지, 이탈 시 복귀하는지 확인.

```bash
ros2 run control_stack can_bridge_node
ros2 run control_stack mecanum_bridge_node
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node --ros-args -p show_window:=true
ros2 run camera_stack lane_recovery_node \
  --ros-args -p output_topic:=/cmd_vel -p auto_enable:=true \
  -p base_speed:=0.10 -p traffic_light_stop:=false
```

---

### 시나리오 C — 신호등 박스 크기 기반 정지 검증 ★

**목적**: 차량이 신호등에 충분히 가까워졌을 때만 RED로 정지하는지 확인.

#### C-1. box_area_threshold 보정 방법

1. 터미널 3 (lane_detector_node) + 터미널 5 (traffic_light_detector_node)만 실행
2. 신호등 앞 **정지하고 싶은 위치**에 차량을 수동으로 놓는다
3. `/traffic_light_box_area` 값을 확인:
   ```bash
   ros2 topic echo /traffic_light_box_area
   ```
4. 해당 값의 **80~90%** 를 `box_area_threshold`로 설정

예시: 정지 위치에서 EMA 값이 3500 px²이면 → `box_area_threshold:=3000.0`

#### C-2. 전체 검증 실행

```bash
# 터미널 1
ros2 run control_stack can_bridge_node

# 터미널 2
ros2 run control_stack mecanum_bridge_node

# 터미널 3
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args -p show_window:=true

# 터미널 4 (보정한 임계값 적용)
ros2 run camera_stack lane_recovery_node \
  --ros-args -p output_topic:=/cmd_vel -p auto_enable:=true \
  -p base_speed:=0.10 -p box_area_threshold:=3000.0

# 터미널 5
ros2 run camera_stack traffic_light_detector_node \
  --ros-args \
  -p model_path:=$(ros2 pkg prefix camera_stack)/share/camera_stack/models/traffic_light_best.pt
```

#### C-3. 검증 포인트

| 단계 | 확인 사항 | 예상 동작 |
|---|---|---|
| 1 | 멀리서 RED 신호등 바라봄 | `box_area` 작음 → `RED 접근 중` 로그, 계속 주행 |
| 2 | 신호등에 가까워짐 | `box_area` 증가 → 임계값 도달 시 `RED 정지` 로그, 정지 |
| 3 | 신호등 GREEN으로 변경 | `traffic_light=GREEN` → 차선 추종 자동 재개 |
| 4 | SPACE 키 | 어느 단계에서든 즉시 정지 |

**터미널 4 로그 예시**:
```
[INFO] RED 접근 중 — box_area=850px²  thr=3000 (정지 미도달)
[WARN] RED 정지 — box_area=3250px²  thr=3000
[INFO] [TL] RED → GREEN
```

---

## 11. 코드 수정 이력

### 2026-06-21 (3차) — 신호등 박스 크기 기반 정지 방식으로 전환

#### 변경 이유

기존 방식(RED 감지 즉시 정지)은 신호등이 멀리 있어도 RED면 정지했기 때문에
정확한 정지 위치 제어가 불가능했습니다.
YOLO 바운딩박스 면적은 거리에 반비례하므로 박스 크기로 접근 거리를 추정하고,
일정 크기 이상일 때만 정지하는 방식으로 변경했습니다.

#### traffic_light_detector_node.py

| 항목 | 내용 |
|---|---|
| `/traffic_light_box_area` 토픽 추가 | EMA 평활화된 RED 박스 면적(px²) 발행 |
| `box_area_ema_alpha` 파라미터 추가 | EMA 계수 (기본값 0.3) |
| `box_area_decay` 파라미터 추가 | RED 박스 없을 때 감쇠율 (기본값 0.9) |
| `_detect()` 반환값 확장 | `max_red_area` 추가 (가장 큰 RED 박스 면적) |
| CSV `box_area_ema_px2` 컬럼 추가 | 매 프레임 EMA 값 기록 |
| 디버그 HUD `boxEMA` 표시 추가 | 실시간 면적 모니터링 |

#### lane_recovery_node.py

| 항목 | 내용 |
|---|---|
| `/traffic_light_box_area` 구독 추가 | EMA 박스 면적 수신 |
| `box_area_threshold` 파라미터 추가 | 정지 기준 면적 (기본값 2000 px², 보정 필요) |
| `use_box_area` 파라미터 추가 | False = 구형 동작 (RED만으로 즉시 정지) |
| 정지 조건 변경 | `RED` → `RED AND box_area >= threshold` |
| 안전 동작 | box_area 데이터 미수신 시 RED만으로 정지 |
| CSV `box_area_px2` 컬럼 추가 | 정지 시점 박스 면적 기록 |
| `reason` 값 변경 | `traffic_light_red` → `traffic_light_red_stop` |

---

### 2026-06-21 (2차) — 신호등 즉시 정지 연동

| 파일 | 항목 |
|---|---|
| `lane_recovery_node.py` | `/traffic_light` 구독 추가, `traffic_light_stop` 파라미터, RED 정지 우선 처리 |
| `lane_recovery_node.py` | CSV `traffic_light` 컬럼 추가 |

---

### 2026-06-21 (1차) — 카메라 이미지 공유

| 파일 | 항목 |
|---|---|
| `lane_detector_node.py` | `/camera/image_raw` 발행 추가 (신호등 노드에 프레임 전달) |
| `traffic_light_detector_node.py` | DepthAI 의존성 완전 제거, ultralytics YOLO `.pt` 추론으로 전환 |

---

### 2026-06-20 — 차선 인식 파라미터 최적화

| 파일 | 주요 변경 |
|---|---|
| `lane_detector_node.py` | `conf_thresh` 0.60→0.45, `half_width` 160→203 px, `max_valid_path_span_px` 230→320, `epsi_weight` 0.0→0.3 |
| `lane_recovery_node.py` | `base_speed` 0.15→0.20, `max_wz` 0.5→0.8, `wz_rate_limit` 0.5 신규, `invalid_hold_frames` 4 신규 |
