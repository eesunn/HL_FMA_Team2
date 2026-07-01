#!/usr/bin/env python3
"""
BEV 캘리브레이션 툴 (ROS2 불필요).

키보드로 4개 소스 포인트를 실시간 조정하면서
BEV 결과가 올바른지 확인한다.

목표:
  직선 구간에서 양쪽 차선이 BEV에서 수직 평행선으로 보일 것.

실행:
    python3 bev_calibrate.py

키 조작:
    1~4  : 조작할 점 선택 (1=좌상, 2=우상, 3=우하, 4=좌하)
    W/S  : 선택 점 위/아래
    A/D  : 선택 점 좌/우
    Q/E  : BEV dst 좌/우 마진 조정
    R    : 초기값으로 리셋
    P    : 현재 파라미터 출력 (ros2 run 명령어 형태)
    ESC  : 종료

출력: 파라미터 값을 복사해서 ros2 run 인자로 사용하면 됨.
"""

import sys

import cv2
import depthai as dai
import numpy as np

IMG_W, IMG_H = 640, 224

# 초기값 (현재 코드 기본값)
params = {
    'top_y':          0.55,   # bev_src_top_y_ratio
    'top_left':       0.32,   # bev_src_top_left_ratio
    'top_right':      0.68,   # bev_src_top_right_ratio
    'bot_left':       0.00,   # bev_src_bottom_left_ratio
    'bot_right':      1.00,   # bev_src_bottom_right_ratio
    'dst_left':       0.25,   # bev_dst_left_ratio
    'dst_right':      0.75,   # bev_dst_right_ratio
}

STEP_RATIO = 0.01   # 한 번 키 입력 시 변화량

selected = 0   # 현재 선택 점 (0=좌상 1=우상 2=우하 3=좌하)
POINT_LABELS = ['[1]좌상', '[2]우상', '[3]우하', '[4]좌하']
POINT_COLORS = [(0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 255, 200)]


def build_transform(p):
    W = float(IMG_W - 1)
    H = float(IMG_H - 1)
    top_y = p['top_y'] * H

    src = np.float32([
        [p['top_left']  * W, top_y],
        [p['top_right'] * W, top_y],
        [p['bot_right'] * W, H],
        [p['bot_left']  * W, H],
    ])
    dst = np.float32([
        [p['dst_left']  * W, 0],
        [p['dst_right'] * W, 0],
        [p['dst_right'] * W, H],
        [p['dst_left']  * W, H],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, src, dst


def draw_overlay(frame, src_pts, sel):
    vis = frame.copy()
    pts = src_pts.astype(np.int32)
    # 사다리꼴 윤곽
    cv2.polylines(vis, [pts[[0, 1, 2, 3, 0]]], False, (0, 255, 255), 1)
    # 각 꼭짓점
    for i, (pt, label, color) in enumerate(zip(pts, POINT_LABELS, POINT_COLORS)):
        thick = 3 if i == sel else 1
        cv2.circle(vis, tuple(pt), 6, color, thick)
        cv2.putText(vis, label, (pt[0] + 8, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(vis, f'SELECT: {POINT_LABELS[sel]}  WASD=이동  QE=dst  P=출력  R=리셋',
                (4, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return vis


def draw_bev(frame, M, dst_pts):
    bev = cv2.warpPerspective(frame, M, (IMG_W, IMG_H),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
    # dst 세로선 표시
    dl = int(dst_pts[0, 0])
    dr = int(dst_pts[1, 0])
    cv2.line(bev, (dl, 0), (dl, IMG_H), (0, 255, 0), 1)
    cv2.line(bev, (dr, 0), (dr, IMG_H), (0, 255, 0), 1)
    mid = (dl + dr) // 2
    cv2.line(bev, (mid, 0), (mid, IMG_H), (0, 0, 200), 1)
    cv2.putText(bev, 'BEV', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(bev, '목표: 직선 구간에서 차선이 초록선 안에서 수직으로 보일 것',
                (4, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return bev


def print_params(p):
    print('\n===== 현재 BEV 파라미터 =====')
    print(f'  bev_src_top_y_ratio        = {p["top_y"]:.3f}')
    print(f'  bev_src_top_left_ratio     = {p["top_left"]:.3f}')
    print(f'  bev_src_top_right_ratio    = {p["top_right"]:.3f}')
    print(f'  bev_src_bottom_left_ratio  = {p["bot_left"]:.3f}')
    print(f'  bev_src_bottom_right_ratio = {p["bot_right"]:.3f}')
    print(f'  bev_dst_left_ratio         = {p["dst_left"]:.3f}')
    print(f'  bev_dst_right_ratio        = {p["dst_right"]:.3f}')
    print('\n--- ros2 run 인자 ---')
    print(
        f'  -p bev_src_top_y_ratio:={p["top_y"]:.3f} \\\n'
        f'  -p bev_src_top_left_ratio:={p["top_left"]:.3f} \\\n'
        f'  -p bev_src_top_right_ratio:={p["top_right"]:.3f} \\\n'
        f'  -p bev_src_bottom_left_ratio:={p["bot_left"]:.3f} \\\n'
        f'  -p bev_src_bottom_right_ratio:={p["bot_right"]:.3f} \\\n'
        f'  -p bev_dst_left_ratio:={p["dst_left"]:.3f} \\\n'
        f'  -p bev_dst_right_ratio:={p["dst_right"]:.3f}'
    )
    print('=' * 32)


def clamp(v, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, v)))


def main():
    global selected, params

    pipeline = dai.Pipeline(dai.Device())
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        (IMG_W, IMG_H),
        type=dai.ImgFrame.Type.BGR888p,
        fps=25,
    )
    rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    print('BEV 캘리브레이션 툴 시작. 직선 구간에 놓고 P 키로 값 확인.')

    last_frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    while True:
        fm = rgb_queue.tryGet()
        if fm is not None:
            last_frame = fm.getCvFrame()

        try:
            M, src_pts, dst_pts = build_transform(params)
        except Exception as e:
            print(f'변환 오류: {e}')
            if cv2.waitKey(30) & 0xFF == 27:
                break
            continue

        src_vis = draw_overlay(last_frame, src_pts, selected)
        bev_vis = draw_bev(last_frame, M, dst_pts)

        combined = np.hstack([src_vis, bev_vis])

        # 파라미터 수치 HUD
        p = params
        info = (
            f'top_y:{p["top_y"]:.3f}  '
            f'TL:{p["top_left"]:.3f} TR:{p["top_right"]:.3f}  '
            f'BL:{p["bot_left"]:.3f} BR:{p["bot_right"]:.3f}  '
            f'dstL:{p["dst_left"]:.3f} dstR:{p["dst_right"]:.3f}'
        )
        cv2.putText(combined, info, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.imshow('BEV Calibration', combined)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:   # ESC
            break
        elif key == ord('1'):  selected = 0
        elif key == ord('2'):  selected = 1
        elif key == ord('3'):  selected = 2
        elif key == ord('4'):  selected = 3

        # 선택된 점 이동
        elif key == ord('w'):
            if selected in (0, 1):   # 상단점 위로
                params['top_y'] = clamp(params['top_y'] - STEP_RATIO)
            else:                     # 하단점은 항상 맨 아래 → 무시
                pass
        elif key == ord('s'):
            if selected in (0, 1):
                params['top_y'] = clamp(params['top_y'] + STEP_RATIO)
        elif key == ord('a'):
            if selected == 0:
                params['top_left']  = clamp(params['top_left']  - STEP_RATIO)
            elif selected == 1:
                params['top_right'] = clamp(params['top_right'] - STEP_RATIO)
            elif selected == 2:
                params['bot_right'] = clamp(params['bot_right'] - STEP_RATIO)
            elif selected == 3:
                params['bot_left']  = clamp(params['bot_left']  - STEP_RATIO)
        elif key == ord('d'):
            if selected == 0:
                params['top_left']  = clamp(params['top_left']  + STEP_RATIO)
            elif selected == 1:
                params['top_right'] = clamp(params['top_right'] + STEP_RATIO)
            elif selected == 2:
                params['bot_right'] = clamp(params['bot_right'] + STEP_RATIO)
            elif selected == 3:
                params['bot_left']  = clamp(params['bot_left']  + STEP_RATIO)

        # BEV dst 좌우 마진
        elif key == ord('q'):
            params['dst_left']  = clamp(params['dst_left']  - STEP_RATIO)
            params['dst_right'] = clamp(params['dst_right'] + STEP_RATIO)
        elif key == ord('e'):
            params['dst_left']  = clamp(params['dst_left']  + STEP_RATIO)
            params['dst_right'] = clamp(params['dst_right'] - STEP_RATIO)

        elif key == ord('r'):
            params = {
                'top_y': 0.55, 'top_left': 0.32, 'top_right': 0.68,
                'bot_left': 0.00, 'bot_right': 1.00,
                'dst_left': 0.25, 'dst_right': 0.75,
            }
            print('[리셋]')
        elif key == ord('p'):
            print_params(params)

    cv2.destroyAllWindows()
    print_params(params)


if __name__ == '__main__':
    main()
