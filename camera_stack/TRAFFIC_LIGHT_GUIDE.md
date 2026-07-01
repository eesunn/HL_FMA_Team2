# 신호등 인식 및 정지 가이드

> **최종 갱신**: 2026-06-21  
> **담당**: 이현준 (카메라 스택)

---

## 목차

1. [동작 원리](#1-동작-원리)
2. [노드 구성](#2-노드-구성)
3. [실행 명령어](#3-실행-명령어)
4. [파라미터 정리](#4-파라미터-정리)
5. [박스 크기 임계값 보정](#5-박스-크기-임계값-보정)
6. [상태 모니터링](#6-상태-모니터링)
7. [CSV 로그 분석](#7-csv-로그-분석)
8. [동작 검증](#8-동작-검증)
9. [수정 이력](#9-수정-이력)

---

## 1. 동작 원리

### 1-1. 신호등 인식 (traffic_light_detector_node)

```
/camera/image_raw
      │
      ▼
 YOLO .pt 추론 (ultralytics, CPU)
      │  classes: {0: green light, 1: red light}
      ▼
 Voting 버퍼 (최근 10프레임)
      │  RED 비율 >= 0.5 → RED
      │  GREEN 비율 >= 0.5 → GREEN
      │  그 외 → NONE
      ▼
 /traffic_light  (String: RED / GREEN / NONE)

 + RED 박스 면적 EMA 추적
      │  RED 박스 있음 → EMA 갱신 (alpha=0.3)
      │  RED 박스 없음 → EMA 감쇠 (decay=0.9)
      ▼
 /traffic_light_box_area  (Float32: EMA 평활 면적 px²)
```

**Voting 사용 이유**: 단일 프레임 오탐(조명 변화, 부분 가림)을 버퍼로 평균화해 오탐률 감소.

### 1-2. 정지 판단 (lane_recovery_node)

```
/traffic_light == RED
        AND
/traffic_light_box_area >= box_area_threshold
        │
        ▼
     정지 (zero Twist 발행)
```

**박스 크기 기반 정지 사용 이유**:  
YOLO 바운딩박스 면적은 차량-신호등 거리에 반비례합니다. RED만으로 즉시 정지하면 신호등이 멀리 있어도 멈춰버려 정지 위치 제어가 불가능했습니다. 박스 크기 임계값을 두면 차량이 원하는 거리까지 접근한 후 정지할 수 있습니다.

**이중 조건 AND 사용 이유 (오탐 방지)**:
- RED voting만: 색상 오탐 시 예상치 못한 위치에서 정지
- 박스 크기만: 신호등 색 무관하게 가까이 가면 정지
- RED AND 박스 크기: 두 조건 모두 충족해야 정지 → 노이즈 및 오탐에 강함

**데이터 미수신 시 안전 동작**:  
`/traffic_light_box_area` 토픽 미수신(traffic_light_detector 미실행) 상태에서 RED 감지 시 → 즉시 정지 (fail-safe).

---

## 2. 노드 구성

### traffic_light_detector_node

| 항목 | 내용 |
|---|---|
| 파일 | `src/camera_stack/camera_stack/traffic_light_detector_node.py` |
| 구독 | `/camera/image_raw` (lane_detector_node 발행) |
| 발행 | `/traffic_light` (String), `/traffic_light_box_area` (Float32), `/traffic_light_debug` (Image) |
| 모델 | `traffic_light_best.pt` (ultralytics YOLO, CPU 추론) |
| 클래스 | `{0: 'green light', 1: 'red light'}` |
| 추론 방식 | PC CPU, 약 27 fps |
| CSV 로그 | `~/capstone_ws/logs/traffic_light_YYYYMMDD_HHMMSS.csv` |

### lane_recovery_node (정지 판단 담당)

| 항목 | 내용 |
|---|---|
| 파일 | `src/camera_stack/camera_stack/lane_recovery_node.py` |
| 추가 구독 | `/traffic_light` (String), `/traffic_light_box_area` (Float32) |
| 정지 조건 | `RED AND EMA_box_area >= box_area_threshold` |
| CSV 로그 | `~/capstone_ws/logs/lane_recovery_YYYYMMDD_HHMMSS.csv` |

---

## 3. 실행 명령어

> **전제**: 터미널마다 `cd ~/capstone_ws && source install/setup.bash`

### 신호등 인식만 단독 테스트

lane_detector_node가 `/camera/image_raw`를 발행해야 합니다.

```bash
# 터미널 A — 차선 인식 (카메라 공급)
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args -p show_window:=true

# 터미널 B — 신호등 인식
ros2 run camera_stack traffic_light_detector_node \
  --ros-args \
  -p model_path:=$(ros2 pkg prefix camera_stack)/share/camera_stack/models/traffic_light_best.pt
```

### 차선 주행 + 신호등 정지 전체 테스트

```bash
# 터미널 1 — CAN 브리지
ros2 run control_stack can_bridge_node

# 터미널 2 — 메카넘 브리지
ros2 run control_stack mecanum_bridge_node

# 터미널 3 — 차선 인식
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node \
  --ros-args -p show_window:=true

# 터미널 4 — 차선 추종 + 신호등 정지
ros2 run camera_stack lane_recovery_node \
  --ros-args \
  -p output_topic:=/cmd_vel \
  -p auto_enable:=true \
  -p base_speed:=0.10 \
  -p box_area_threshold:=3000.0

# 터미널 5 — 신호등 인식
ros2 run camera_stack traffic_light_detector_node \
  --ros-args \
  -p model_path:=$(ros2 pkg prefix camera_stack)/share/camera_stack/models/traffic_light_best.pt
```

---

## 4. 파라미터 정리

### traffic_light_detector_node

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `model_path` | `share/camera_stack/models/traffic_light_best.pt` | YOLO 모델 경로 |
| `conf_thresh` | `0.50` | 박스 인정 최소 신뢰도. 오탐 많으면 높이기 |
| `vote_buffer_size` | `10` | voting 버퍼 크기 (프레임 수). 클수록 안정, 반응 느림 |
| `min_vote_samples` | `3` | 판정에 필요한 최소 프레임 수 |
| `red_vote_ratio` | `0.5` | 버퍼 내 RED 비율 >= 이 값이면 RED 판정 |
| `green_vote_ratio` | `0.5` | 버퍼 내 GREEN 비율 >= 이 값이면 GREEN 판정 |
| `box_area_ema_alpha` | `0.3` | EMA 평활 계수 (0~1, 클수록 최신 값에 빠르게 반응) |
| `box_area_decay` | `0.9` | RED 박스 없을 때 프레임당 EMA 감쇠율 |
| `input_topic` | `/camera/image_raw` | 입력 이미지 토픽 |
| `show_window` | `False` | 로컬 윈도우 표시 여부 |
| `log_csv` | `True` | CSV 로그 저장 여부 |

### lane_recovery_node (신호등 관련)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `traffic_light_stop` | `True` | 신호등 정지 기능 활성화. False → 신호등 무시 |
| `box_area_threshold` | `2000.0` | 정지 기준 EMA 박스 면적 (px²). **환경에 맞게 보정 필요** |
| `use_box_area` | `True` | False → RED 감지만으로 즉시 정지 (구형 동작) |

---

## 5. 박스 크기 임계값 보정

`box_area_threshold`는 환경(신호등 크기, 카메라 설치 위치)에 따라 달라지므로 실제 측정이 필요합니다.

### 보정 절차

**1단계** — lane_detector_node + traffic_light_detector_node만 실행:

```bash
# 터미널 A
QT_QPA_PLATFORM=xcb ros2 run camera_stack lane_detector_node --ros-args -p show_window:=true

# 터미널 B
ros2 run camera_stack traffic_light_detector_node \
  --ros-args \
  -p model_path:=$(ros2 pkg prefix camera_stack)/share/camera_stack/models/traffic_light_best.pt
```

**2단계** — EMA 값 모니터링:

```bash
ros2 topic echo /traffic_light_box_area
```

**3단계** — 차량을 *정지하고 싶은 위치* 앞에 수동으로 놓고 RED 신호등을 향하게 한다.

**4단계** — 해당 위치에서 출력되는 `box_area` 값을 확인한다.

**5단계** — 그 값의 **약 80~90%** 를 `box_area_threshold`로 설정한다.

```
예시: 정지 위치에서 EMA 값 = 3500 px²
      → box_area_threshold = 3000.0
```

> EMA이므로 값이 안정화되기까지 수 초 기다렸다 확인하세요.  
> `box_area_decay` 감쇠 특성 때문에, RED 박스가 갑자기 사라지면 값이 천천히 줄어듭니다.

---

## 6. 상태 모니터링

```bash
# 신호등 상태 (RED / GREEN / NONE)
ros2 topic echo /traffic_light

# EMA 박스 면적 (px²) — 보정 및 디버깅에 활용
ros2 topic echo /traffic_light_box_area

# 디버그 이미지 (박스 + EMA 면적 + voting 현황 오버레이)
ros2 run rqt_image_view rqt_image_view /traffic_light_debug

# 발행 주기 확인
ros2 topic hz /traffic_light
ros2 topic hz /traffic_light_box_area
```

### lane_recovery_node 로그 해석

| 로그 메시지 | 의미 |
|---|---|
| `RED 접근 중 — box_area=850px²  thr=3000 (정지 미도달)` | RED지만 아직 멀리 있음, 계속 주행 |
| `RED 정지 — box_area=3250px²  thr=3000` | 정지 조건 충족, zero Twist 발행 |
| `[TL] NONE → RED` | 신호등 상태 전환 |
| `[TL] RED → GREEN` | 신호등 해제, 차선 추종 재개 |

---

## 7. CSV 로그 분석

### traffic_light_YYYYMMDD_HHMMSS.csv

| 컬럼 | 설명 |
|---|---|
| `ts_sec` | 수신 타임스탬프 |
| `raw_detect` | 프레임 단위 YOLO 결과 (voting 이전) |
| `state` | voting 후 최종 상태 |
| `state_changed` | 이전 프레임 대비 상태 변경 여부 (0/1) |
| `max_red_conf` | RED 박스 최대 신뢰도 |
| `max_green_conf` | GREEN 박스 최대 신뢰도 |
| `n_red_boxes` | 프레임 내 RED 박스 수 |
| `n_green_boxes` | 프레임 내 GREEN 박스 수 |
| `vote_r / vote_g / vote_n` | 버퍼 내 각 클래스 투표 수 |
| `box_area_ema_px2` | EMA 평활화된 RED 박스 면적 (px²) |

```bash
# 최근 CSV 분석 — RED 구간 box_area 통계
python3 -c "
import pandas as pd, glob, os
f = sorted(glob.glob(os.path.expanduser('~/capstone_ws/logs/traffic_light_*.csv')))[-1]
df = pd.read_csv(f)
red = df[df['state'] == 'RED']
print(f'전체 프레임: {len(df)}, RED 프레임: {len(red)}')
print('--- box_area_ema_px2 (RED 상태) ---')
print(red['box_area_ema_px2'].describe().round(1))
print()
print('--- state 분포 ---')
print(df['state'].value_counts())
"
```

### lane_recovery_YYYYMMDD_HHMMSS.csv (신호등 관련 컬럼)

| 컬럼 | 설명 |
|---|---|
| `traffic_light` | 해당 시점 신호등 상태 |
| `box_area_px2` | 해당 시점 EMA 박스 면적 |
| `reason` | `traffic_light_red_stop` = 신호등으로 인한 정지 |

```bash
# 정지 시점 박스 면적 확인 (임계값 보정에 활용)
python3 -c "
import pandas as pd, glob, os
f = sorted(glob.glob(os.path.expanduser('~/capstone_ws/logs/lane_recovery_*.csv')))[-1]
df = pd.read_csv(f)
stops = df[df['reason'] == 'traffic_light_red_stop']
print(f'신호등 정지 횟수: {len(stops)}')
print(stops[['ts_sec','traffic_light','box_area_px2']].head(20).to_string(index=False))
"
```

---

## 8. 동작 검증

### 시나리오 1 — 박스 크기 조건 없이 단순 동작 확인 (use_box_area=false)

```bash
# lane_recovery_node를 구형 모드로 실행
ros2 run camera_stack lane_recovery_node \
  --ros-args -p output_topic:=/cmd_vel -p auto_enable:=true \
  -p use_box_area:=false

# RED 발행 → 즉시 정지 확인
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'RED'}"

# GREEN 발행 → 재개 확인
ros2 topic pub --once /traffic_light std_msgs/String "{data: 'GREEN'}"
```

### 시나리오 2 — 박스 크기 조건 검증 (토픽 직접 발행)

```bash
# lane_recovery_node 기본 모드로 실행 (use_box_area=true)
ros2 run camera_stack lane_recovery_node \
  --ros-args -p output_topic:=/cmd_vel -p auto_enable:=true \
  -p box_area_threshold:=2000.0

# RED + 작은 박스 → 계속 주행해야 함
ros2 topic pub --rate 10 /traffic_light std_msgs/String "{data: 'RED'}" &
ros2 topic pub --rate 10 /traffic_light_box_area std_msgs/Float32 "{data: 500.0}"

# Ctrl+C 후 — RED + 큰 박스 → 정지해야 함
ros2 topic pub --rate 10 /traffic_light std_msgs/String "{data: 'RED'}" &
ros2 topic pub --rate 10 /traffic_light_box_area std_msgs/Float32 "{data: 3000.0}"
```

### 시나리오 3 — 실제 신호등으로 전체 검증

| 단계 | 확인 사항 | 예상 결과 |
|---|---|---|
| 1 | 신호등 없는 구간 주행 | `/traffic_light=NONE`, 차선 추종 중 |
| 2 | 멀리서 RED 신호등 감지 | `/traffic_light=RED`, `box_area` 작음 → 계속 주행 |
| 3 | 신호등에 접근 | `box_area` 증가 → 임계값 도달 시 정지 |
| 4 | 신호등 GREEN으로 전환 | 차선 추종 자동 재개 |

---

## 9. 수정 이력

### 2026-06-21 — 박스 크기 기반 정지 방식 도입

**변경 전 방식**: `RED` 감지 즉시 정지  
**변경 후 방식**: `RED AND EMA(box_area) >= threshold` 충족 시 정지

**변경 이유**:
- 기존 방식은 신호등이 카메라에 포착되는 순간(먼 거리에서도) 정지
- 정확한 정지 위치 제어가 불가능
- 박스 면적은 거리에 반비례하므로 근접 여부를 직접 측정 가능

**적용된 필터링**:
1. YOLO voting (10프레임 버퍼) — 색상 오탐 평균화
2. EMA 평활 (alpha=0.3) — 면적의 프레임 간 진동 제거
3. RED 박스 없을 때 decay (0.9) — 신호등 사라지면 서서히 0으로 복귀
4. AND 조건 — 색상 + 크기 두 조건 동시 충족 시에만 정지

**변경 파일**:
- `traffic_light_detector_node.py`: `/traffic_light_box_area` 발행, EMA 추적, CSV `box_area_ema_px2` 컬럼 추가
- `lane_recovery_node.py`: `/traffic_light_box_area` 구독, `box_area_threshold` / `use_box_area` 파라미터 추가, 정지 조건 변경, CSV `box_area_px2` 컬럼 추가

### 2026-06-21 — 신호등 연동 초기 구현

- `traffic_light_detector_node.py`: DepthAI 제거 → ultralytics YOLO `.pt` CPU 추론으로 전환, voting 방식 상태 결정
- `lane_recovery_node.py`: `/traffic_light` 구독 추가, RED 감지 시 정지 로직 추가
