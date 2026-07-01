# Camera Stack 통합 계획

작성일: 2026-06-20

---

## 1. OAK-D 하드웨어 제약

depthai 3.x에서 `dai.Pipeline(dai.Device())`는 **단일 프로세스가 디바이스를 배타적으로 점유**한다.

| 시도 | 결과 |
|---|---|
| 두 프로세스(노드)가 각자 `dai.Device()` 호출 | 두 번째 노드 `RuntimeError: Failed to open the device` |
| 한 파이프라인 내에서 `cam_out → NN_A`, `cam_out → NN_B` 동시 연결 | **가능** (depthai 공식 지원) |

즉, **두 개의 독립 ROS2 노드가 각자 blob을 갖고 OAK-D에 접근하는 구조는 불가능**하다.

---

## 2. 현재 camera_stack 구조

```
src/camera_stack/
├── camera_stack/
│   ├── lane_detector_node.py         (1659줄)  실제 사용
│   ├── lane_detector_bev_node.py     (458줄)   미사용 (가이드에 명시)
│   ├── lane_recovery_node.py         (274줄)   차선 추종 cmd_vel 생성
│   ├── traffic_light_detector_node.py (358줄)  신호등 인식
│   └── diag_logger.py                (308줄)   진단 로거 유틸
└── models/
    ├── best_openvino_2022.3_6shave.blob    차선 (YOLOv8-seg)
    └── traffic_light_640x224_6shave.blob   신호등 (YOLOv8-detect)
```

### 각 노드 역할

| 노드 | OAK-D 점유 | 사용 blob | 주요 입력 | 주요 출력 |
|---|---|---|---|---|
| `lane_detector_node` | 직접 점유 | best_openvino_2022.3_6shave | IMU(/imu/data) | `/lane_offset`, `/lane_valid`, `/lane_debug` |
| `traffic_light_detector_node` | 직접 점유 | traffic_light_640x224 | - | `/traffic_light`, `/traffic_light_debug`, `/cmd_vel_traffic` |
| `lane_recovery_node` | 없음 | - | `/lane_offset`, `/lane_valid`, `/lane_recovery_enable` | `/cmd_vel_recovery` |

### 현재 문제점

`lane_detector_node`와 `traffic_light_detector_node`를 **동시에 실행할 수 없다.**
두 노드 모두 `_init_device()`에서 `dai.Pipeline(dai.Device())`를 호출하기 때문이다.

---

## 3. lane_detector_node 내부 파이프라인 구조

```
OAK-D CAM_A
    └── cam_out (640×224, BGR888p, 25fps)
            └── → lane_nn (YOLOv8-seg)
                      └── nn_queue → _process_cb() 30Hz 타이머
```

### 차선 인식 처리 흐름 (lane_detector_node)

```
nn_queue
    └── _parse_yolov8seg()          NN 출력 파싱 (conf_thresh=0.45)
            └── _filter_blobs()     면적/종횡비 필터
                    └── _to_bev()   BEV 원근 변환
                            └── _lane_center_poly()
                                    ├── row-based run 탐색 (min/max_run_width)
                                    ├── robust polyfit (잔차 필터 + inlier ratio)
                                    ├── A 필터: 좌/우 차선 polynomial EMA (poly_alpha=0.25)
                                    └── half-width 학습 (단일 차선 중심 복원)
                                            └── _compute_offset()
                                                    └── C 필터: scalar EMA + HOLD
                                                            └── /lane_offset 발행
```

### 신호등 인식 처리 흐름 (traffic_light_detector_node)

```
OAK-D CAM_A → cam_out (640×224)
    ├── [ML 모드]  → tl_nn (YOLOv8-detect) → nn_queue
    │                   output0 shape: [4+nc, num_anchors] (nc=2, cls0=GREEN, cls1=RED)
    └── [HSV 모드] → rgb_queue → HSV 색상 필터 (ROI crop → morphology → 면적 비교)
                                     └── Voting Buffer (10프레임) → /traffic_light
```

---

## 4. 다음 단계: 통합 노드 설계

### 파이프라인 구조

```
OAK-D CAM_A
    └── cam_out (640×224, BGR888p)
            ├── → lane_nn  (best_openvino_2022.3_6shave.blob)
            │        └── lane_nn_queue
            └── → tl_nn   (traffic_light_640x224_6shave.blob)
                      └── tl_nn_queue
            (+ rgb_queue: 디버그 윈도우 show_window=True 시)
```

`cam_out` 하나를 두 NN 노드 입력에 동시에 링크(fork)한다.
두 NN은 MyriadX 내에서 별도 파이프로 병렬 실행된다.

### 통합 노드 구성안

| 항목 | 내용 |
|---|---|
| 파일명 | `camera_stack/camera_driver_node.py` |
| 기반 | `lane_detector_node.py` 전체 로직 + `traffic_light_detector_node.py` 전체 로직 |
| 타이머 | 30Hz 단일 콜백에서 `lane_nn_queue.tryGet()` + `tl_nn_queue.tryGet()` 동시 폴링 |
| 구독 | `/imu/data` (차선 헤딩 교차검증) |
| 발행 | `/lane_offset`, `/lane_valid`, `/lane_offset_m`, `/lane_curvature`, `/lane_debug` |
|      | `/traffic_light`, `/traffic_light_debug` |

### 기존 노드 처리

| 노드 | 처리 |
|---|---|
| `lane_detector_node.py` | 통합 노드로 대체 (파일은 유지) |
| `traffic_light_detector_node.py` | 통합 노드로 대체 (파일은 유지) |
| `lane_recovery_node.py` | **그대로 유지** (순수 ROS2, 카메라 무관) |
| `lane_detector_bev_node.py` | **그대로 유지** (미사용이지만 삭제하지 않음) |

### 토픽 흐름 (통합 후)

```
[camera_driver_node]
    /lane_offset ──────────→ [lane_recovery_node]
    /lane_valid  ──────────→ [lane_recovery_node]
    /traffic_light ────────→ [mission_fsm]
                                    /cmd_vel_recovery ──┐
                                    /cmd_vel_nav ───────┤→ twist_mux → /cmd_vel
                                    /cmd_vel_zero ──────┘
```

---

## 5. 기타 확인 사항

- `cmd_vel_traffic`: 기존 `traffic_light_detector_node`에서 발행했으나 CLAUDE.md 토픽 표준에 없음 → 통합 노드에서는 `/traffic_light`만 발행, 정지 제어는 `mission_fsm → stop_governor → /cmd_vel_zero` 경로 사용
- `lane_recovery_node`의 `auto_enable` 파라미터는 기본 `False` → `mission_fsm`에서 `/lane_recovery_enable` 토픽으로 활성화
- DiagLogger는 통합 노드에서도 그대로 사용 가능 (독립 유틸 클래스)
