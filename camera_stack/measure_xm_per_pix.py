#!/usr/bin/env python3
"""
xm_per_pix 측정 도구.

BEV 이미지에서 알려진 거리를 마우스로 클릭하여
xm_per_pix 값을 계산한다.

실행:
    python3 measure_xm_per_pix.py

[줄자 배치 방법]
    차선을 가로질러 (좌우 방향) 줄자를 놓는다.
    예: 왼쪽 차선에서 오른쪽 차선까지, 또는 임의의 가로 50cm

[사용법]
    1. BEV 창에서 LEFT-CLICK  → 첫 번째 점 (P1)
    2.          RIGHT-CLICK  → 두 번째 점 (P2)
    3. 터미널에 실제 거리(cm) 입력 → xm_per_pix 출력
    4. 측정 반복 원하면 다시 클릭
    5. ESC / Q 로 종료
"""

import cv2
import depthai as dai
import numpy as np

IMG_W, IMG_H = 640, 224

# ── 확정된 BEV 파라미터 ────────────────────────────────────────────
BEV = {
    'top_y':    0.660,
    'top_left': 0.350, 'top_right': 0.650,
    'bot_left': 0.000, 'bot_right': 1.000,
    'dst_left': 0.250, 'dst_right': 0.750,
}

W = float(IMG_W - 1)
H = float(IMG_H - 1)

_src = np.float32([
    [BEV['top_left']  * W, BEV['top_y'] * H],
    [BEV['top_right'] * W, BEV['top_y'] * H],
    [BEV['bot_right'] * W, H],
    [BEV['bot_left']  * W, H],
])
_dst = np.float32([
    [BEV['dst_left']  * W, 0],
    [BEV['dst_right'] * W, 0],
    [BEV['dst_right'] * W, H],
    [BEV['dst_left']  * W, H],
])
M = cv2.getPerspectiveTransform(_src, _dst)

DST_L = int(_dst[0, 0])
DST_R = int(_dst[1, 0])
DST_MID = (DST_L + DST_R) // 2

# ── 상태 ─────────────────────────────────────────────────────────
pt1 = None
pt2 = None
last_frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
results = []


def calc_and_print(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist_px = float(np.sqrt(dx * dx + dy * dy))
    print(f'\n  픽셀 거리: {dist_px:.1f} px')
    raw = input('  실제 거리 (cm): ').strip()
    try:
        real_m = float(raw) / 100.0
        xm = real_m / dist_px
        print(f'\n  → xm_per_pix = {xm:.6f}  ({dist_px:.1f}px = {real_m*100:.0f}cm)')
        print(f'     ros2 run ... --ros-args -p xm_per_pix:={xm:.6f}\n')
        results.append((dist_px, real_m, xm))
        return xm
    except ValueError:
        print('  숫자만 입력하세요.')
        return None


def mouse_cb(event, x, y, flags, param):
    global pt1, pt2
    if event == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x, y)
        pt2 = None
        print(f'[P1] x={x}  y={y}')
    elif event == cv2.EVENT_RBUTTONDOWN:
        if pt1 is None:
            print('먼저 LEFT-CLICK으로 P1을 지정하세요.')
            return
        pt2 = (x, y)
        print(f'[P2] x={x}  y={y}')
        calc_and_print(pt1, pt2)


def draw_bev(frame):
    bev = cv2.warpPerspective(frame, M, (IMG_W, IMG_H),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

    # dst 경계선
    cv2.line(bev, (DST_L,   0), (DST_L,   IMG_H), (0, 200, 0), 1)
    cv2.line(bev, (DST_R,   0), (DST_R,   IMG_H), (0, 200, 0), 1)
    cv2.line(bev, (DST_MID, 0), (DST_MID, IMG_H), (80, 80, 220), 1)

    # 클릭 점 및 선
    if pt1 is not None:
        cv2.circle(bev, pt1, 5, (0, 255, 255), -1)
        cv2.putText(bev, 'P1', (pt1[0] + 6, pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    if pt2 is not None:
        cv2.circle(bev, pt2, 5, (255, 120, 0), -1)
        cv2.putText(bev, 'P2', (pt2[0] + 6, pt2[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1)
    if pt1 is not None and pt2 is not None:
        cv2.line(bev, pt1, pt2, (200, 200, 0), 1)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        dist = np.sqrt(dx * dx + dy * dy)
        mx = (pt1[0] + pt2[0]) // 2
        my = (pt1[1] + pt2[1]) // 2 - 8
        cv2.putText(bev, f'{dist:.0f}px', (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    # 최근 측정 결과 표시
    if results:
        last = results[-1]
        txt = f'xm_per_pix={last[2]:.5f}  ({last[0]:.0f}px={last[1]*100:.0f}cm)'
        cv2.putText(bev, txt, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1)

    cv2.putText(bev, 'L=P1  R=P2  ESC/Q=종료',
                (4, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return bev


def main():
    global last_frame

    pipeline = dai.Pipeline(dai.Device())
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput((IMG_W, IMG_H),
                                type=dai.ImgFrame.Type.BGR888p, fps=25)
    q = cam_out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    cv2.namedWindow('xm_per_pix Measurement')
    cv2.setMouseCallback('xm_per_pix Measurement', mouse_cb)

    print('=' * 50)
    print('xm_per_pix 측정 도구')
    print('줄자를 차선과 수직으로 (좌우 방향) 바닥에 놓고')
    print('BEV 창에서 LEFT-CLICK=P1, RIGHT-CLICK=P2')
    print('=' * 50)

    while True:
        fm = q.tryGet()
        if fm is not None:
            last_frame = fm.getCvFrame()

        vis = draw_bev(last_frame)
        cv2.imshow('xm_per_pix Measurement', vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()

    if results:
        print('\n=== 측정 결과 요약 ===')
        xm_values = [r[2] for r in results]
        print(f'  측정 횟수: {len(xm_values)}')
        print(f'  평균 xm_per_pix: {np.mean(xm_values):.6f}')
        if len(xm_values) > 1:
            print(f'  최솟값: {min(xm_values):.6f}')
            print(f'  최댓값: {max(xm_values):.6f}')
        print(f'\n  최종 권장값: -p xm_per_pix:={np.mean(xm_values):.6f}')
    print('종료.')


if __name__ == '__main__':
    main()
