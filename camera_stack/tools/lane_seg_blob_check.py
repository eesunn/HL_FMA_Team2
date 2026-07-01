#!/usr/bin/env python3
"""OAK-D 차선 세그멘테이션 blob 빠른 확인 도구.

ROS2 없이 OAK-D에서 YOLOv8-seg 차선 blob을 직접 실행하여
마스크 오버레이를 실시간으로 확인합니다.

실행:
    QT_QPA_PLATFORM=xcb python3 tools/lane_seg_blob_check.py
    QT_QPA_PLATFORM=xcb python3 tools/lane_seg_blob_check.py --conf 0.6 --mask-thresh 0.35

키 조작:
  + / =  : conf threshold 올리기 (+0.05)
  -      : conf threshold 낮추기 (-0.05)
  ]      : mask threshold 올리기 (+0.05)
  [      : mask threshold 낮추기 (-0.05)
  s      : 현재 프레임 /tmp 에 저장
  ESC/q  : 종료
"""

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


IMG_W = 640
IMG_H = 224

DEFAULT_BLOB = str(
    Path(__file__).resolve().parent.parent
    / 'models' / 'best_openvino_2022.1_6shave_1.blob'
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def decode_seg(
    tensor0: np.ndarray,
    tensor1: np.ndarray,
    conf_thresh: float,
    mask_thresh: float,
) -> tuple:
    """YOLOv8-seg 두 출력 텐서를 이진 마스크로 디코딩.

    Returns
    -------
    binary   : (H, W) uint8 마스크, 감지 없으면 None
    n_keep   : conf_thresh 통과한 앵커 수
    max_conf : 최대 앵커 신뢰도
    """
    H4 = IMG_H // 4   # 56
    W4 = IMG_W // 4   # 160
    expected0 = 37 * 2940
    expected1 = 32 * H4 * W4

    raw0 = np.asarray(tensor0, dtype=np.float32).flatten()
    raw1 = np.asarray(tensor1, dtype=np.float32).flatten()

    # 텐서 순서가 뒤집힌 경우 자동 교정
    if raw0.size == expected1 and raw1.size == expected0:
        raw0, raw1 = raw1, raw0

    if raw0.size != expected0 or raw1.size != expected1:
        print(f'[WARN] 텐서 크기 불일치: t0={raw0.size}(expect {expected0})  '
              f't1={raw1.size}(expect {expected1})')
        return None, 0, 0.0

    preds  = raw0.reshape(37, 2940).T     # (2940, 37)
    protos = raw1.reshape(32, H4, W4)     # (32, 56, 160)

    scores = preds[:, 4]
    max_conf = float(scores.max()) if scores.size else 0.0
    keep = scores > conf_thresh
    n_keep = int(keep.sum())

    if n_keep == 0:
        return None, 0, max_conf

    mask_coeffs = preds[keep, 5:]                      # (N, 32)
    proto_flat  = protos.reshape(32, H4 * W4)          # (32, 8960)
    masks_raw   = mask_coeffs @ proto_flat             # (N, 8960)
    masks       = sigmoid(masks_raw).reshape(-1, H4, W4)  # (N, 56, 160)
    combined    = masks.max(axis=0)                    # (56, 160)

    combined_resized = cv2.resize(
        combined, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR
    )
    binary = (combined_resized > mask_thresh).astype(np.uint8)
    return binary, n_keep, max_conf


def build_pipeline(blob_path: str):
    pipeline = dai.Pipeline(dai.Device())

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        (IMG_W, IMG_H),
        type=dai.ImgFrame.Type.BGR888p,
        fps=25,
    )

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    cam_out.link(nn.input)

    rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
    nn_queue  = nn.out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    return pipeline, rgb_queue, nn_queue


def draw_hud(
    vis: np.ndarray,
    conf_thresh: float,
    mask_thresh: float,
    fps: float,
    max_conf: float,
    n_keep: int,
    mask_px: int,
) -> None:
    H, W = vis.shape[:2]
    coverage = mask_px / (W * H) * 100.0
    has_mask = mask_px > 0

    cv2.putText(vis,
                f'conf:{conf_thresh:.2f}  mask:{mask_thresh:.2f}  fps:{fps:.1f}',
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.putText(vis,
                f'max_conf:{max_conf:.3f}  anchors:{n_keep}',
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.putText(vis,
                f'mask_px:{mask_px}  coverage:{coverage:.1f}%',
                (5, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if has_mask else (80, 80, 80), 1)

    state_txt   = 'DETECT' if has_mask else 'MISS'
    state_color = (0, 255, 0) if has_mask else (0, 80, 255)
    cv2.putText(vis, state_txt, (W - 85, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    # 조작 도움말
    cv2.putText(vis, '+/-:conf  [/]:mask  s:save  q:quit',
                (5, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description='OAK-D lane seg blob 확인')
    parser.add_argument('--blob', default=DEFAULT_BLOB,
                        help='blob 파일 경로')
    parser.add_argument('--conf', type=float, default=0.70,
                        help='앵커 신뢰도 임계값 (기본 0.70)')
    parser.add_argument('--mask-thresh', type=float, default=0.45,
                        help='마스크 sigmoid 이진화 임계값 (기본 0.45)')
    args = parser.parse_args()

    blob = Path(args.blob).expanduser().resolve()
    if not blob.is_file():
        raise FileNotFoundError(f'blob 파일 없음: {blob}')

    conf_thresh = args.conf
    mask_thresh = args.mask_thresh

    print('=' * 60)
    print(f'blob       : {blob.name}')
    print(f'input      : {IMG_W}x{IMG_H}')
    print(f'conf/mask  : {conf_thresh:.2f} / {mask_thresh:.2f}')
    print('keys       : +/-  conf   [ / ]  mask_thresh   s  save   ESC/q  quit')
    print('=' * 60)

    pipeline, rgb_queue, nn_queue = build_pipeline(str(blob))

    latest_frame  = None
    latest_binary = None
    n_keep        = 0
    max_conf      = 0.0
    fps           = 0.0
    last_t        = time.time()
    save_count    = 0

    try:
        while True:
            rgb_msg = rgb_queue.tryGet()
            nn_msg  = nn_queue.tryGet()

            if rgb_msg is not None:
                latest_frame = rgb_msg.getCvFrame()

            if latest_frame is None:
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                continue

            vis = latest_frame.copy()

            # NN 디코딩
            if nn_msg is not None:
                names = sorted(nn_msg.getAllLayerNames())
                if len(names) >= 2:
                    t0 = np.array(nn_msg.getTensor(names[0]), dtype=np.float32)
                    t1 = np.array(nn_msg.getTensor(names[1]), dtype=np.float32)
                    latest_binary, n_keep, max_conf = decode_seg(
                        t0, t1, conf_thresh, mask_thresh
                    )
                else:
                    print(f'[WARN] 레이어 수 부족: {names}')

                now = time.time()
                dt  = now - last_t
                last_t = now
                if 0 < dt < 5.0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0.0 else 1.0 / dt

            # 마스크 오버레이 (초록 반투명)
            mask_px = 0
            if latest_binary is not None:
                mask_px = int(latest_binary.sum())
                overlay = np.zeros_like(vis)
                overlay[latest_binary > 0] = (0, 220, 0)
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

            # 이미지 중앙 기준선 (빨간)
            cv2.line(vis, (IMG_W // 2, 0), (IMG_W // 2, IMG_H - 1), (0, 0, 255), 1)

            draw_hud(vis, conf_thresh, mask_thresh, fps, max_conf, n_keep, mask_px)

            cv2.imshow('lane_seg_blob_check', vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key in (ord('+'), ord('=')):
                conf_thresh = round(min(conf_thresh + 0.05, 0.95), 2)
                print(f'conf -> {conf_thresh:.2f}')
            elif key == ord('-'):
                conf_thresh = round(max(conf_thresh - 0.05, 0.05), 2)
                print(f'conf -> {conf_thresh:.2f}')
            elif key == ord(']'):
                mask_thresh = round(min(mask_thresh + 0.05, 0.95), 2)
                print(f'mask_thresh -> {mask_thresh:.2f}')
            elif key == ord('['):
                mask_thresh = round(max(mask_thresh - 0.05, 0.05), 2)
                print(f'mask_thresh -> {mask_thresh:.2f}')
            elif key == ord('s'):
                out = Path('/tmp') / f'lane_seg_check_{int(time.time())}_{save_count:04d}.png'
                cv2.imwrite(str(out), vis)
                save_count += 1
                print(f'[저장] {out}')

    finally:
        pipeline = None
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
