#!/usr/bin/env python3
"""OAK-D traffic-light blob quick checker.

This script runs a YOLOv8-detect style traffic-light blob directly on OAK-D
and shows bounding boxes without requiring ROS2.

Keys:
  + / = : increase confidence threshold
  -     : decrease confidence threshold
  s     : save current debug frame to /tmp
  ESC/q : quit
"""

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


IMG_W = 640
IMG_H = 224
DEFAULT_BLOB = (
    '/home/hyeonjun/capstone_ws/src/camera_stack/models/'
    'traffic_light_640x224_6shave.blob'
)

CLASS_NAMES = ('green', 'red')
CLASS_COLORS = ((0, 220, 0), (0, 0, 255))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def decode_yolov8_detect(
    tensor: np.ndarray,
    conf_thresh: float,
    num_classes: int,
    diag: bool = False,
) -> list[tuple[int, int, int, int, float, int]]:
    """Decode YOLOv8 detect output.

    Expected layout is either:
      [4 + nc, anchors] or [anchors, 4 + nc]

    The function also tolerates a leading batch dimension and automatically
    decides whether class scores are logits or already probabilities.
    """
    arr = np.asarray(tensor, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        if arr.size % (4 + num_classes) != 0:
            print(f'[WARN] cannot reshape tensor: shape={tensor.shape}, size={arr.size}')
            return []
        arr = arr.reshape(4 + num_classes, -1)

    features = 4 + num_classes
    if arr.shape[0] == features:
        preds = arr
    elif arr.shape[1] == features:
        preds = arr.T
    else:
        print(f'[WARN] unexpected tensor shape: {tensor.shape} squeezed={arr.shape}')
        return []

    if diag:
        print(f'  decoded layout: features={preds.shape[0]} anchors={preds.shape[1]}')
        labels = ['cx', 'cy', 'w', 'h'] + [f'class_{i}' for i in range(num_classes)]
        for i, label in enumerate(labels[:preds.shape[0]]):
            row = preds[i]
            print(
                f'  row[{i}] {label:8s}: '
                f'min={row.min():8.3f} max={row.max():8.3f} '
                f'mean={row.mean():8.3f}'
            )

    cx, cy, bw, bh = preds[0], preds[1], preds[2], preds[3]
    cls_raw = preds[4:4 + num_classes]
    need_sigmoid = bool(cls_raw.max() > 1.0 or cls_raw.min() < 0.0)
    cls_scores = sigmoid(cls_raw) if need_sigmoid else cls_raw

    if diag:
        print(
            f'  sigmoid={need_sigmoid} '
            f'raw_cls=[{cls_raw.min():.3f},{cls_raw.max():.3f}] '
            f'score=[{cls_scores.min():.3f},{cls_scores.max():.3f}]'
        )

    detections = []
    for cls_id in range(num_classes):
        anchor_idxs = np.where(cls_scores[cls_id] >= conf_thresh)[0]
        for idx in anchor_idxs:
            x1 = int(round(cx[idx] - bw[idx] / 2.0))
            y1 = int(round(cy[idx] - bh[idx] / 2.0))
            x2 = int(round(cx[idx] + bw[idx] / 2.0))
            y2 = int(round(cy[idx] + bh[idx] / 2.0))
            x1 = int(np.clip(x1, 0, IMG_W - 1))
            y1 = int(np.clip(y1, 0, IMG_H - 1))
            x2 = int(np.clip(x2, 0, IMG_W - 1))
            y2 = int(np.clip(y2, 0, IMG_H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((x1, y1, x2, y2, float(cls_scores[cls_id, idx]), cls_id))
    return detections


def nms(
    detections: list[tuple[int, int, int, int, float, int]],
    conf_thresh: float,
    iou_thresh: float,
) -> list[tuple[int, int, int, int, float, int]]:
    if not detections:
        return []

    result = []
    for cls_id in sorted(set(det[5] for det in detections)):
        cls_dets = [det for det in detections if det[5] == cls_id]
        boxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in cls_dets]
        scores = [d[4] for d in cls_dets]
        idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        for i in (idxs.flatten() if len(idxs) > 0 else []):
            result.append(cls_dets[int(i)])
    return result


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float, int]],
    conf_thresh: float,
    fps: float,
) -> str:
    state = 'NONE'

    for x1, y1, x2, y2, score, cls_id in detections:
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class{cls_id}'
        color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 0)
        label = f'{name} {score:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        if name.lower().startswith('red') or cls_id == 1:
            state = 'RED'
        elif state != 'RED':
            state = 'GREEN'

    state_color = {
        'RED': (0, 0, 255),
        'GREEN': (0, 220, 0),
        'NONE': (150, 150, 150),
    }[state]
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, state, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
    cv2.putText(
        frame, f'conf:{conf_thresh:.2f} fps:{fps:.1f} dets:{len(detections)}',
        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
    )
    return state


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
    nn_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    return pipeline, rgb_queue, nn_queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--blob', default=DEFAULT_BLOB)
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--diag-frames', type=int, default=5)
    args = parser.parse_args()

    blob = Path(args.blob).expanduser()
    if not blob.is_file():
        raise FileNotFoundError(f'blob not found: {blob}')

    print('=' * 60)
    print(f'blob       : {blob}')
    print(f'input size : {IMG_W}x{IMG_H}')
    print(f'classes    : {args.classes}  names={CLASS_NAMES}')
    print(f'conf/iou   : {args.conf:.2f}/{args.iou:.2f}')
    print('keys       : +/- conf, s save, ESC/q quit')
    print('=' * 60)

    pipeline, rgb_queue, nn_queue = build_pipeline(str(blob))
    latest = None
    last_t = time.time()
    fps = 0.0
    diag_left = args.diag_frames

    try:
        while True:
            rgb_msg = rgb_queue.tryGet()
            nn_msg = nn_queue.tryGet()

            if rgb_msg is not None:
                latest = rgb_msg.getCvFrame()

            if latest is None:
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                continue

            vis = latest.copy()

            if nn_msg is not None:
                names = nn_msg.getAllLayerNames()
                tensors = [np.array(nn_msg.getTensor(name), dtype=np.float32) for name in names]
                diag = diag_left > 0
                if diag:
                    print(f'\n[NN] layers={names}')
                    for name, tensor in zip(names, tensors):
                        print(f'  {name}: shape={tensor.shape} size={tensor.size}')
                    diag_left -= 1

                detections = []
                for tensor in tensors:
                    detections.extend(
                        decode_yolov8_detect(tensor, args.conf, args.classes, diag=diag)
                    )
                    diag = False
                detections = nms(detections, args.conf, args.iou)

                now = time.time()
                dt = max(now - last_t, 1e-6)
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0.0 else 1.0 / dt
                last_t = now

                state = draw_detections(vis, detections, args.conf, fps)
                if detections:
                    best = max(detections, key=lambda det: det[4])
                    name = CLASS_NAMES[best[5]] if best[5] < len(CLASS_NAMES) else str(best[5])
                    print(
                        f'[{state}] best={name} {best[4]:.2f} '
                        f'box={best[:4]} n={len(detections)} conf={args.conf:.2f}'
                    )
            else:
                draw_detections(vis, [], args.conf, fps)

            cv2.imshow('traffic_light_blob_check', vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in (ord('+'), ord('=')):
                args.conf = min(args.conf + 0.05, 0.95)
                print(f'conf -> {args.conf:.2f}')
            elif key == ord('-'):
                args.conf = max(args.conf - 0.05, 0.05)
                print(f'conf -> {args.conf:.2f}')
            elif key == ord('s'):
                out = Path('/tmp') / f'traffic_light_blob_check_{int(time.time())}.png'
                cv2.imwrite(str(out), vis)
                print(f'saved: {out}')
    finally:
        pipeline = None
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
