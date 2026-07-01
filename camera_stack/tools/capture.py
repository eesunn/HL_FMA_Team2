"""OAK-D 캡처 도구.

s 키 : 자동 저장 시작/정지 (1초에 1장)
Space : 수동 1장 저장
ESC/q : 종료

저장 경로: ./capture_YYYYMMDD_HHMMSS/0000.jpg ...

사용법:
  python3 capture.py
  python3 capture.py --interval 2      # 2초에 1장
  python3 capture.py --out ~/dataset   # 저장 폴더 지정
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

IMG_W = 640
IMG_H = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=1.0,
                        help='자동 저장 간격(초). 기본 1.0')
    parser.add_argument('--out', type=str, default='',
                        help='저장 폴더 경로. 기본: ./capture_YYYYMMDD_HHMMSS')
    args = parser.parse_args()

    # 저장 폴더
    if args.out:
        save_dir = Path(args.out)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(f'capture_{ts}')
    save_dir.mkdir(parents=True, exist_ok=True)

    interval     = args.interval
    auto_running = False   # s 키로 토글
    count        = 0
    last_saved   = 0.0
    latest_frame = None

    print('━' * 45)
    print(f'  저장 폴더 : {save_dir.resolve()}')
    print(f'  간격      : {interval}초')
    print('  s     : 자동 저장 시작 / 정지')
    print('  Space : 수동 1장 저장')
    print('  ESC/q : 종료')
    print('━' * 45)

    pipeline = dai.Pipeline(dai.Device())
    cam     = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput((IMG_W, IMG_H),
                                type=dai.ImgFrame.Type.BGR888p,
                                fps=25)
    q = cam_out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    print('카메라 워밍업 중...')
    time.sleep(2)

    while True:
            in_frame = q.tryGet()
            if in_frame is not None:
                latest_frame = in_frame.getCvFrame()

            if latest_frame is not None:
                display = latest_frame.copy()

                # 상태 오버레이
                status = 'REC' if auto_running else 'STOP'
                color  = (0, 0, 255) if auto_running else (180, 180, 180)
                cv2.putText(display, status, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, f'saved: {count}', (80, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                if auto_running:
                    remain = max(0.0, interval - (time.time() - last_saved))
                    cv2.putText(display, f'next: {remain:.1f}s', (10, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

                cv2.namedWindow('OAK-D capture', cv2.WINDOW_NORMAL)
                cv2.imshow('OAK-D capture', display)

            key = cv2.waitKey(30) & 0xFF

            # ── s : 자동 저장 토글 ─────────────────────────────────────
            if key == ord('s'):
                auto_running = not auto_running
                last_saved   = time.time()
                print(f'자동 저장 {"시작" if auto_running else "정지"}')

            # ── 자동 저장 ──────────────────────────────────────────────
            now = time.time()
            if auto_running and latest_frame is not None and (now - last_saved) >= interval:
                path = save_dir / f'{count:04d}.jpg'
                cv2.imwrite(str(path), latest_frame)
                print(f'[자동] {path.name}  (총 {count + 1}장)')
                count     += 1
                last_saved = now

            # ── Space : 수동 저장 ──────────────────────────────────────
            if key == 32 and latest_frame is not None:
                path = save_dir / f'{count:04d}.jpg'
                cv2.imwrite(str(path), latest_frame)
                print(f'[수동] {path.name}  (총 {count + 1}장)')
                count     += 1
                last_saved = now

            # ── 종료 ──────────────────────────────────────────────────
            if key in (27, ord('q')):
                break

    pipeline = None
    cv2.destroyAllWindows()
    print(f'\n완료. 총 {count}장 → {save_dir.resolve()}')


if __name__ == '__main__':
    main()
