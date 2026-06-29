#!/usr/bin/env python3

"""
Copyright 2026 NXP

SPDX-License-Identifier: BSD-3-Clause

Headless (no GUI) test harness for object detection models running on the
i.MX95 Neutron NPU.

Supports two output formats, auto-selected by output-tensor count:
  * SSD MobileNet (two outputs: boxes + scores), decoded with box_priors.txt;
  * YOLOv8 (one raw (4 + num_classes, num_anchors) output), decoded in-script
    with confidence thresholding + per-class NMS (no priors needed).

Unlike object_detection.py, this script does not open a GTK window. It loads the
quantized TFLite detection model through the Neutron external delegate, runs
inference on a single image (or a frame captured from a camera), decodes the
output, and prints the detections and inference timing to the console. Useful
for bring-up, regression checks, and benchmarking the NPU without a display.

Example:
    python3 object_detection_neutron_headless.py --image bus.jpg
    python3 object_detection_neutron_headless.py --camera /dev/video0 --runs 30
    python3 object_detection_neutron_headless.py --backend cpu --image bus.jpg
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# Import utils (shared asset downloader)
sys.path.append("/root/gopoint-apps/scripts/")
import utils

MODELS_PATH = "/root/gopoint-apps/downloads/"

# Default assets (mirror the names used by object_detection.py)
DEFAULT_NEUTRON_MODEL = (
    "yolov8s_neutron.tflite"
)
DEFAULT_CPU_MODEL = (
    "ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess.tflite"
)
DEFAULT_LABELS = "coco_labels_list.txt"
DEFAULT_PRIORS = "box_priors.txt"

NEUTRON_DELEGATE = "/usr/lib/libneutron_delegate.so"

# SSD box decoding scales (same constants the NNStreamer mobilenet-ssd decoder uses)
Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

# Map of utils.download_file() negative return codes to messages
DOWNLOAD_ERRORS = {
    -1: "Cannot find file in downloads database (downloads.json).",
    -2: "Download failed. Check the target's internet connection and retry.",
    -3: "Downloaded file is corrupted. Clean /root/gopoint-apps/downloads and retry.",
}


def fetch(name):
    """Resolve an asset to a local path, or exit.

    Prefers a file that already exists locally (a direct path, or a file already
    present in the downloads folder) so models not registered in downloads.json
    can still be used. Otherwise falls back to utils.download_file().
    """
    if os.path.isfile(name):
        return name
    local = os.path.join(MODELS_PATH, name)
    if os.path.isfile(local):
        return local

    result = utils.download_file(name)
    if isinstance(result, int) and result < 0:
        sys.exit(f"Error fetching '{name}': {DOWNLOAD_ERRORS.get(result, result)}")
    return result


def load_interpreter(model_path, backend):
    """Create a TFLite interpreter, optionally with the Neutron delegate."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import (  # type: ignore
                Interpreter,
                load_delegate,
            )
        except ImportError:
            sys.exit(
                "No TFLite runtime found. Install tflite-runtime or tensorflow."
            )

    delegates = []
    if backend == "neutron":
        if not os.path.exists(NEUTRON_DELEGATE):
            sys.exit(
                f"Neutron delegate not found at {NEUTRON_DELEGATE}. "
                "This backend is only available on i.MX95."
            )
        delegates.append(load_delegate(NEUTRON_DELEGATE))

    interpreter = Interpreter(
        model_path=model_path, experimental_delegates=delegates
    )
    interpreter.allocate_tensors()
    return interpreter


def load_labels(path):
    """Read the COCO label list (one label per line)."""
    with open(path, encoding="utf-8") as label_file:
        return [line.strip() for line in label_file]


def load_priors(path):
    """Read box_priors.txt as a (4, num_boxes) float array.

    Rows are [y_center, x_center, height, width].
    """
    priors = np.loadtxt(path, dtype=np.float32)
    if priors.shape[0] != 4:
        sys.exit(f"Unexpected box_priors shape {priors.shape}; expected 4 rows.")
    return priors


def read_image(args, width, height):
    """Return an (height, width, 3) uint8 RGB array from --image or --camera."""
    if args.camera:
        image_path = capture_frame(args.camera)
    else:
        image_path = args.image

    try:
        import cv2

        bgr = cv2.imread(image_path)
        if bgr is None:
            sys.exit(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        return np.asarray(resized, dtype=np.uint8)
    except ImportError:
        from PIL import Image

        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((width, height))
            return np.asarray(img, dtype=np.uint8)


def capture_frame(device):
    """Grab a single frame from a V4L2 camera using GStreamer, return a JPEG path."""
    out_path = os.path.join(tempfile.gettempdir(), "neutron_test_frame.jpg")
    pipeline = (
        f"gst-launch-1.0 -e v4l2src device={device} num-buffers=1 ! "
        "videoconvert ! jpegenc ! filesink location=" + out_path
    )
    result = subprocess.run(pipeline, shell=True, capture_output=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        sys.exit(
            f"Failed to capture frame from {device}:\n"
            + result.stderr.decode("utf-8", "ignore")
        )
    return out_path


def decode_detections(boxes, scores, priors, labels, score_threshold, iou_threshold):
    """Decode raw SSD outputs into a list of detections.

    boxes  -- (num_boxes, 4) float32, encoded as [dy, dx, dh, dw]
    scores -- (num_boxes, num_classes) float32 class probabilities
    Returns a list of dicts: {label, class_id, score, box=(ymin, xmin, ymax, xmax)}.
    """
    y_center = priors[0] + (boxes[:, 0] / Y_SCALE) * priors[2]
    x_center = priors[1] + (boxes[:, 1] / X_SCALE) * priors[3]
    half_h = np.exp(boxes[:, 2] / H_SCALE) * priors[2] / 2.0
    half_w = np.exp(boxes[:, 3] / W_SCALE) * priors[3] / 2.0

    ymin = y_center - half_h
    xmin = x_center - half_w
    ymax = y_center + half_h
    xmax = x_center + half_w
    decoded_boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)

    # The "no_postprocess" model emits raw logits; convert to probabilities.
    scores = 1.0 / (1.0 + np.exp(-scores))

    detections = []
    num_classes = scores.shape[1]
    # Class 0 is background; skip it.
    for class_id in range(1, num_classes):
        class_scores = scores[:, class_id]
        candidates = np.where(class_scores >= score_threshold)[0]
        if candidates.size == 0:
            continue
        keep = non_max_suppression(
            decoded_boxes[candidates], class_scores[candidates], iou_threshold
        )
        for idx in keep:
            box_idx = candidates[idx]
            label = labels[class_id] if class_id < len(labels) else str(class_id)
            detections.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "score": float(class_scores[box_idx]),
                    "box": tuple(float(v) for v in decoded_boxes[box_idx]),
                }
            )
    detections.sort(key=lambda det: det["score"], reverse=True)
    return detections


def non_max_suppression(boxes, scores, iou_threshold):
    """Greedy NMS. boxes are (N, 4) as [ymin, xmin, ymax, xmax]."""
    ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, ymax - ymin) * np.maximum(0.0, xmax - xmin)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        inter = np.maximum(0.0, yy2 - yy1) * np.maximum(0.0, xx2 - xx1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolov8(prediction, labels, score_threshold, iou_threshold, width, height):
    """Decode a single YOLOv8 detect output into detections.

    YOLOv8 emits one tensor of shape (4 + num_classes, num_anchors) -- e.g.
    (84, 18900) for COCO -- with no embedded NMS:
      rows 0:4 = box as (cx, cy, w, h) in *input-pixel* units (0..width/height),
      rows 4:  = per-class confidences, already sigmoid-activated by the head.
    Returns the same dict shape as decode_detections(), with boxes normalized to
    [0, 1] as (ymin, xmin, ymax, xmax).
    """
    preds = np.asarray(prediction, dtype=np.float32)
    # Orient to (num_anchors, 4 + num_classes): the anchor axis is the longer one.
    if preds.ndim != 2:
        sys.exit(f"Unexpected YOLOv8 output shape {preds.shape}; expected 2-D.")
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    # Best class per anchor, then threshold (no objectness term in YOLOv8).
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep_mask = confidences >= score_threshold
    boxes_xywh = boxes_xywh[keep_mask]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    if boxes_xywh.shape[0] == 0:
        return []

    # (cx, cy, w, h) in input pixels -> normalized (ymin, xmin, ymax, xmax).
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    xmin = (cx - bw / 2.0) / width
    xmax = (cx + bw / 2.0) / width
    ymin = (cy - bh / 2.0) / height
    ymax = (cy + bh / 2.0) / height
    decoded_boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)

    # Per-class NMS. YOLOv8 COCO classes are contiguous 0..79 (no background),
    # so labels must be the 80-class COCO list (person..toothbrush).
    detections = []
    for class_id in np.unique(class_ids):
        cls_mask = class_ids == class_id
        cls_boxes = decoded_boxes[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = non_max_suppression(cls_boxes, cls_scores, iou_threshold)
        cls_indices = np.where(cls_mask)[0]
        for idx in keep:
            box_idx = cls_indices[idx]
            label = labels[class_id] if class_id < len(labels) else str(class_id)
            detections.append(
                {
                    "label": label,
                    "class_id": int(class_id),
                    "score": float(confidences[box_idx]),
                    "box": tuple(float(v) for v in decoded_boxes[box_idx]),
                }
            )
    detections.sort(key=lambda det: det["score"], reverse=True)
    return detections


def read_output(interpreter, detail):
    """Return an output tensor as float32, dequantizing if the model is quantized.

    Neutron-converted models often keep int8/uint8 outputs; the raw integers must
    be mapped back with real = scale * (q - zero_point) before decoding.
    """
    tensor = interpreter.get_tensor(detail["index"])
    scale, zero_point = detail["quantization"]
    if scale:
        return (tensor.astype(np.float32) - zero_point) * scale
    return tensor.astype(np.float32)


def split_outputs(interpreter):
    """Return (boxes, scores) arrays from the interpreter's output tensors.

    The box tensor is identified by a trailing dimension of 4.
    """
    boxes = scores = None
    for detail in interpreter.get_output_details():
        tensor = read_output(interpreter, detail)
        flat = np.squeeze(tensor)
        if flat.ndim == 2 and flat.shape[1] == 4:
            boxes = flat.astype(np.float32)
        else:
            scores = flat.astype(np.float32)
    if boxes is None or scores is None:
        sys.exit("Could not identify box/score output tensors from the model.")
    return boxes, scores


def is_yolo_model(interpreter):
    """Decide YOLO vs SSD from output shapes alone (before inference).

    YOLO emits either one (4+nc, anchors) tensor or -- when the head is split for
    int8 -- two tensors shaped (1, 4, anchors) and (1, nc, anchors): a size-4 axis
    that is NOT the last dim, with a larger trailing anchor dim. SSD's box tensor
    instead has its 4 as the LAST dim (1, num_boxes, 4).
    """
    shapes = [list(d["shape"]) for d in interpreter.get_output_details()]
    if len(shapes) == 1:
        return True
    if len(shapes) == 2:
        if any(s[-1] == 4 for s in shapes):
            return False  # SSD: box tensor has trailing dim 4
        # YOLO split (box+score) or raw head (box DFL + class logits): both are
        # two (1, channels, anchors) tensors.
        return all(len(s) == 3 for s in shapes)
    return False


def make_anchors(height, width, strides=(8, 16, 32), offset=0.5):
    """YOLOv8/11 anchor points (grid centers) and per-anchor strides.

    Mirrors ultralytics make_anchors: scales of stride 8/16/32, grid centers at
    cell+0.5, flattened row-major, concatenated in stride order. Returns
    (anchor_points [N, 2] as (x, y), strides [N]).
    """
    pts, strd = [], []
    for s in strides:
        nh, nw = height // s, width // s
        sx = np.arange(nw, dtype=np.float32) + offset
        sy = np.arange(nh, dtype=np.float32) + offset
        gy, gx = np.meshgrid(sy, sx, indexing="ij")
        pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        strd.append(np.full((nh * nw,), float(s), dtype=np.float32))
    return np.concatenate(pts, 0), np.concatenate(strd, 0)


def decode_raw_head(box_raw, cls_raw, height, width):
    """Decode a raw YOLO head (pre-decode export) on the host.

    box_raw -- (anchors, 4*reg_max) DFL logits; cls_raw -- (anchors, num_classes)
    class logits. Does DFL (softmax over reg_max bins -> expected distance),
    dist2bbox with anchor grid + strides, and sigmoid on the classes. Returns
    (anchors, 4 + num_classes) with box as (cx, cy, w, h) in input pixels -- the
    same layout decode_yolov8() expects. This keeps the precision-sensitive box
    decode in float on the CPU, off the NPU's int8 kernels.
    """
    n, reg = box_raw.shape[0], box_raw.shape[1] // 4
    anchors, strides = make_anchors(height, width)
    if anchors.shape[0] != n:
        sys.exit(f"anchor count {anchors.shape[0]} != {n} outputs; bad stride layout.")
    bins = box_raw.reshape(n, 4, reg)
    bins = bins - bins.max(axis=2, keepdims=True)
    soft = np.exp(bins)
    soft /= soft.sum(axis=2, keepdims=True)
    dist = (soft * np.arange(reg, dtype=np.float32)).sum(axis=2)  # [N,4] = l,t,r,b
    x1y1 = anchors - dist[:, :2]
    x2y2 = anchors + dist[:, 2:]
    cxcy = (x1y1 + x2y2) / 2.0
    wh = x2y2 - x1y1
    box_px = np.concatenate([cxcy, wh], axis=1) * strides[:, None]  # cx,cy,w,h px
    cls = 1.0 / (1.0 + np.exp(-cls_raw))
    return np.concatenate([box_px, cls], axis=1)


def yolo_predictions(interpreter):
    """Assemble YOLO outputs into one (num_anchors, 4 + num_classes) float array.

    Handles three head layouts, all oriented to (anchors, channels):
      * one tensor -- already-decoded (4+nc, anchors);
      * two tensors with a 4-channel box -- decoded split box/score;
      * two tensors, box channels = 4*reg_max -- raw head, decoded here on the host
        (box first, class second; see --yolo-raw-head in the converter).
    """
    outs = []
    for detail in interpreter.get_output_details():
        arr = np.squeeze(read_output(interpreter, detail)).astype(np.float32)
        if arr.ndim != 2:
            sys.exit(f"Unexpected YOLO output shape {arr.shape}; expected 2-D.")
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        outs.append(arr)
    if len(outs) == 1:
        return outs[0]
    if len(outs) != 2 or outs[0].shape[0] != outs[1].shape[0]:
        sys.exit("Could not identify YOLO box/score output tensors from the model.")
    chans = [o.shape[1] for o in outs]
    if 4 in chans:  # decoded split: box (4) + class (nc)
        box = next(o for o in outs if o.shape[1] == 4)
        cls = next(o for o in outs if o.shape[1] != 4)
        return np.concatenate([box, cls], axis=1)
    # raw head: box DFL logits (4*reg_max) first, class logits second.
    _, height, width, _ = interpreter.get_input_details()[0]["shape"]
    return decode_raw_head(outs[0], outs[1], int(height), int(width))


def main():
    parser = argparse.ArgumentParser(
        description="Headless Neutron NPU test for SSDLite MobileNet v2 detection."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to an input image file.")
    source.add_argument(
        "--camera", help="V4L2 device to capture one frame from (e.g. /dev/video0)."
    )
    parser.add_argument(
        "--backend",
        choices=["neutron", "cpu"],
        default="neutron",
        help="Inference backend (default: neutron).",
    )
    parser.add_argument("--model", help="Override the TFLite model path.")
    parser.add_argument("--labels", help="Override the labels file path.")
    parser.add_argument("--priors", help="Override the box_priors file path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score to report a detection (default: 0.5).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression (default: 0.45).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed inference iterations for benchmarking (default: 1).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print input/output tensor shapes, dtypes and value ranges, then "
        "skip SSD decoding. Use this to identify a model's output format.",
    )
    parser.add_argument(
        "--input-range",
        choices=["0-1", "0-255"],
        default="0-1",
        help="Float input scaling for YOLO models: '0-1' divides by 255 "
        "(ultralytics default), '0-255' feeds raw pixels (default: 0-1).",
    )
    args = parser.parse_args()

    # Resolve assets (download on first use, like the GUI demo).
    default_model = (
        DEFAULT_NEUTRON_MODEL if args.backend == "neutron" else DEFAULT_CPU_MODEL
    )
    model_path = args.model or fetch(default_model)
    labels_path = args.labels or fetch(DEFAULT_LABELS)
    priors_path = args.priors or fetch(DEFAULT_PRIORS)

    print(f"Backend     : {args.backend}")
    print(f"Model       : {model_path}")

    interpreter = load_interpreter(model_path, args.backend)
    input_detail = interpreter.get_input_details()[0]
    _, height, width, _ = input_detail["shape"]

    # Decide YOLO vs SSD from output shapes (YOLO may have a split box/score head
    # of two tensors). This drives both the input normalization and the decoder.
    is_yolo = is_yolo_model(interpreter)

    labels = load_labels(labels_path)
    priors = None if is_yolo else load_priors(priors_path)

    image = read_image(args, width, height)
    input_data = np.expand_dims(image, axis=0)
    if input_detail["dtype"] != np.uint8:
        if is_yolo:
            # YOLOv8 expects RGB; ultralytics default is [0, 1] (divide by 255),
            # but some exports bake normalization in and want raw [0, 255].
            divisor = 255.0 if args.input_range == "0-1" else 1.0
            input_data = input_data.astype(np.float32) / divisor
        else:
            # SSD float input model: normalize to [-1, 1] (centeredReduced).
            input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    # Warm-up run (first inference includes graph setup / NPU compile).
    interpreter.set_tensor(input_detail["index"], input_data)
    interpreter.invoke()

    # Timed runs.
    timings = []
    for _ in range(max(1, args.runs)):
        start = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], input_data)
        interpreter.invoke()
        timings.append((time.perf_counter() - start) * 1000.0)

    avg_ms = sum(timings) / len(timings)
    print(f"Input shape : {width}x{height}")

    if args.inspect:
        in_d = input_detail
        print(
            f"Input tensor: '{in_d['name']}' shape={list(in_d['shape'])} "
            f"dtype={np.dtype(in_d['dtype']).name} quant={in_d['quantization']}"
        )
        for i, detail in enumerate(interpreter.get_output_details()):
            raw = interpreter.get_tensor(detail["index"])
            deq = read_output(interpreter, detail)
            print(
                f"Output[{i}]  : '{detail['name']}' shape={list(raw.shape)} "
                f"dtype={raw.dtype} quant={detail['quantization']}"
            )
            print(
                f"             dequant min={deq.min():.4f} "
                f"max={deq.max():.4f} mean={deq.mean():.4f}"
            )

        if is_yolo:
            preds = yolo_predictions(interpreter)
            box_rows = preds[:, :4]
            cls_rows = preds[:, 4:]
            best = cls_rows.max(axis=1)
            print(
                f"YOLO split  : {preds.shape[1] - 4} classes, "
                f"{preds.shape[0]} anchors"
            )
            print(
                f"  box (cols 0:4) min={box_rows.min():.3f} max={box_rows.max():.3f}"
            )
            print(
                f"  cls (cols 4:)  min={cls_rows.min():.6e} max={cls_rows.max():.6e}"
            )
            print(f"  best class score per anchor: max={best.max():.6e}")
            for thr in (0.01, 0.05, 0.1, 0.25, 0.5):
                print(f"    anchors with best >= {thr:<4}: {int((best >= thr).sum())}")
        return

    if is_yolo:
        prediction = yolo_predictions(interpreter)
        detections = decode_yolov8(
            prediction, labels, args.threshold, args.iou, width, height
        )
    else:
        boxes, scores = split_outputs(interpreter)
        detections = decode_detections(
            boxes, scores, priors, labels, args.threshold, args.iou
        )

    print(
        f"Inference   : avg {avg_ms:.2f} ms  "
        f"(min {min(timings):.2f} / max {max(timings):.2f} ms over {len(timings)} runs)"
    )
    print(f"Throughput  : {1000.0 / avg_ms:.1f} FPS")
    print(f"Detections  : {len(detections)} above threshold {args.threshold}")
    for det in detections:
        ymin, xmin, ymax, xmax = det["box"]
        print(
            f"  {det['label']:<16} {det['score'] * 100:5.1f}%  "
            f"box=[{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]"
        )


if __name__ == "__main__":
    main()
