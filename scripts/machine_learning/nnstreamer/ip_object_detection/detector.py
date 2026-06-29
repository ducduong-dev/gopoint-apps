#!/usr/bin/env python3

"""
Copyright 2026 NXP

SPDX-License-Identifier: BSD-3-Clause

YOLOv8 object detector for the i.MX95 Neutron NPU.

This is the inference core for the IP-camera object detection + tracking demo. It
is a direct port of the decode pipeline validated in
``object_detection_neutron_headless.py`` (same repo), repackaged as a reusable
``Detector`` class so the GUI app can run inference per frame in-process and feed
the results to ByteTrack.

Pipeline per frame:
  1. resize RGB frame to the model input, normalize to [0, 1] (YOLOv8 default);
  2. invoke the TFLite interpreter (Neutron external delegate on i.MX95, or CPU);
  3. dequantize outputs, assemble the YOLO head (handles decoded, split, and
     raw-DFL exports), decode to boxes, run per-class NMS;
  4. return boxes in *pixel* coordinates of the original frame, ready for the
     tracker.

Only numpy + a TFLite runtime are required; no host-only deps.
"""

import os
import sys

import numpy as np

# Import utils (shared asset downloader)
sys.path.append("/root/gopoint-apps/scripts/")
import utils

MODELS_PATH = "/root/gopoint-apps/downloads/"

DEFAULT_NEUTRON_MODEL = "yolov8s_640_neutron.tflite"
DEFAULT_CPU_MODEL = "yolov8s_640_int8.tflite"
DEFAULT_LABELS = "coco80_labels.txt"

NEUTRON_DELEGATE = "/usr/lib/libneutron_delegate.so"

# utils.download_file() negative return codes -> human messages.
DOWNLOAD_ERRORS = {
    -1: "Cannot find file in downloads database (downloads.json).",
    -2: "Download failed. Check the target's internet connection and retry.",
    -3: "Downloaded file is corrupted. Clean /root/gopoint-apps/downloads and retry.",
}


def fetch(name):
    """Resolve an asset to a local path, downloading on first use.

    Prefers a file that already exists (direct path or already in the downloads
    folder) so models not registered in downloads.json can still be used.
    Raises RuntimeError on failure so the GUI can surface the message.
    """
    if os.path.isfile(name):
        return name
    local = os.path.join(MODELS_PATH, name)
    if os.path.isfile(local):
        return local

    result = utils.download_file(name)
    if isinstance(result, int) and result < 0:
        raise RuntimeError(
            f"Error fetching '{name}': {DOWNLOAD_ERRORS.get(result, result)}"
        )
    return result


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


def make_anchors(height, width, strides=(8, 16, 32), offset=0.5):
    """YOLOv8/11 anchor points (grid centers) and per-anchor strides."""
    pts, strd = [], []
    for s in strides:
        nh, nw = height // s, width // s
        sx = np.arange(nw, dtype=np.float32) + offset
        sy = np.arange(nh, dtype=np.float32) + offset
        gy, gx = np.meshgrid(sy, sx, indexing="ij")
        pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        strd.append(np.full((nh * nw,), float(s), dtype=np.float32))
    return np.concatenate(pts, 0), np.concatenate(strd, 0)


def decode_raw_head(box_raw, cls_raw, height, width, score_threshold=0.0):
    """Decode a raw YOLO head (pre-decode export) on the host (DFL + dist2bbox).

    box_raw -- (anchors, 4*reg_max) DFL logits; cls_raw -- (anchors, num_classes)
    class logits. Returns (M, 4 + num_classes) with box cols as (cx, cy, w, h) in
    input pixels and class cols sigmoid-activated.

    The DFL softmax/dist2bbox is the dominant host cost, so it is run *only* on
    anchors whose best class score already clears ``score_threshold``: the cheap
    per-anchor sigmoid+max prefilter typically leaves a few dozen of the ~8400
    anchors, cutting the expensive box decode by ~100x. Pass score_threshold=0 to
    decode every anchor (original behavior).
    """
    n, reg = box_raw.shape[0], box_raw.shape[1] // 4
    anchors, strides = make_anchors(height, width)
    if anchors.shape[0] != n:
        raise RuntimeError(
            f"anchor count {anchors.shape[0]} != {n} outputs; bad stride layout."
        )

    # Cheap prefilter before the costly DFL box decode. Sigmoid is monotonic, so
    # sigmoid(x) >= t  <=>  x >= logit(t): threshold on the *raw* class logits to
    # keep surviving anchors, avoiding an exp() over all ~8400 x num_classes
    # scores. Only the survivors are then sigmoid-activated.
    num_classes = cls_raw.shape[1]
    if score_threshold > 0.0:
        clamped = min(score_threshold, 1.0 - 1e-6)
        logit_threshold = np.log(clamped / (1.0 - clamped))
        keep = cls_raw.max(axis=1) >= logit_threshold
        if not keep.any():
            return np.zeros((0, 4 + num_classes), dtype=np.float32)
        box_raw = box_raw[keep]
        cls_raw = cls_raw[keep]
        anchors = anchors[keep]
        strides = strides[keep]

    cls = 1.0 / (1.0 + np.exp(-cls_raw))
    m = box_raw.shape[0]
    bins = box_raw.reshape(m, 4, reg)
    bins = bins - bins.max(axis=2, keepdims=True)
    soft = np.exp(bins)
    soft /= soft.sum(axis=2, keepdims=True)
    dist = (soft * np.arange(reg, dtype=np.float32)).sum(axis=2)  # [M,4]=l,t,r,b
    x1y1 = anchors - dist[:, :2]
    x2y2 = anchors + dist[:, 2:]
    cxcy = (x1y1 + x2y2) / 2.0
    wh = x2y2 - x1y1
    box_px = np.concatenate([cxcy, wh], axis=1) * strides[:, None]
    return np.concatenate([box_px, cls], axis=1)


def decode_yolov8(prediction, score_threshold, iou_threshold, width, height):
    """Decode a (num_anchors, 4 + num_classes) YOLOv8 output into detections.

    Box rows are (cx, cy, w, h) in input-pixel units; class columns are already
    sigmoid-activated. Returns three arrays (boxes_norm, scores, class_ids) where
    boxes_norm is (K, 4) as normalized (ymin, xmin, ymax, xmax) in [0, 1].
    """
    preds = np.asarray(prediction, dtype=np.float32)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep_mask = confidences >= score_threshold
    boxes_xywh = boxes_xywh[keep_mask]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    if boxes_xywh.shape[0] == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    cx, cy = boxes_xywh[:, 0], boxes_xywh[:, 1]
    bw, bh = boxes_xywh[:, 2], boxes_xywh[:, 3]
    xmin = (cx - bw / 2.0) / width
    xmax = (cx + bw / 2.0) / width
    ymin = (cy - bh / 2.0) / height
    ymax = (cy + bh / 2.0) / height
    decoded_boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)

    out_boxes, out_scores, out_classes = [], [], []
    for class_id in np.unique(class_ids):
        cls_mask = class_ids == class_id
        cls_boxes = decoded_boxes[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = non_max_suppression(cls_boxes, cls_scores, iou_threshold)
        for idx in keep:
            out_boxes.append(cls_boxes[idx])
            out_scores.append(cls_scores[idx])
            out_classes.append(int(class_id))
    return (
        np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(out_scores, dtype=np.float32),
        np.asarray(out_classes, dtype=np.int32),
    )


class Detector:
    """Loads a YOLOv8 TFLite model and runs detection on RGB frames."""

    def __init__(self, backend="neutron", model_path=None, labels_path=None):
        self.backend = backend
        model = model_path or fetch(
            DEFAULT_NEUTRON_MODEL if backend == "neutron" else DEFAULT_CPU_MODEL
        )
        labels = labels_path or fetch(DEFAULT_LABELS)
        self.labels = self._load_labels(labels)
        self.model_path = model

        self.interpreter = self._load_interpreter(model, backend)
        self.input_detail = self.interpreter.get_input_details()[0]
        _, self.height, self.width, _ = self.input_detail["shape"]
        self.is_float_input = self.input_detail["dtype"] != np.uint8

        # Warm-up (first invoke compiles the NPU graph).
        dummy = np.zeros(
            (1, self.height, self.width, 3), dtype=self.input_detail["dtype"]
        )
        self.interpreter.set_tensor(self.input_detail["index"], dummy)
        self.interpreter.invoke()

    @staticmethod
    def _load_labels(path):
        with open(path, encoding="utf-8") as label_file:
            return [line.strip() for line in label_file if line.strip()]

    @staticmethod
    def _load_interpreter(model_path, backend):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            from tensorflow.lite.python.interpreter import (  # type: ignore
                Interpreter,
                load_delegate,
            )

        delegates = []
        if backend == "neutron":
            if not os.path.exists(NEUTRON_DELEGATE):
                raise RuntimeError(
                    f"Neutron delegate not found at {NEUTRON_DELEGATE}. "
                    "This backend is only available on i.MX95."
                )
            delegates.append(load_delegate(NEUTRON_DELEGATE))

        interpreter = Interpreter(
            model_path=model_path, experimental_delegates=delegates
        )
        interpreter.allocate_tensors()
        return interpreter

    def _read_output(self, detail):
        """Output tensor as float32, dequantized if the model is quantized."""
        tensor = self.interpreter.get_tensor(detail["index"])
        scale, zero_point = detail["quantization"]
        if scale:
            return (tensor.astype(np.float32) - zero_point) * scale
        return tensor.astype(np.float32)

    def _yolo_predictions(self, score_threshold=0.0):
        """Assemble outputs into one (num_anchors, 4 + num_classes) array.

        For a raw-head model, score_threshold is forwarded to decode_raw_head so
        the expensive DFL box decode only runs on anchors that clear it.
        """
        outs = []
        for detail in self.interpreter.get_output_details():
            arr = np.squeeze(self._read_output(detail)).astype(np.float32)
            if arr.ndim != 2:
                raise RuntimeError(
                    f"Unexpected YOLO output shape {arr.shape}; expected 2-D."
                )
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            outs.append(arr)
        if len(outs) == 1:
            return outs[0]
        if len(outs) != 2 or outs[0].shape[0] != outs[1].shape[0]:
            raise RuntimeError("Could not identify YOLO box/score output tensors.")
        chans = [o.shape[1] for o in outs]
        if 4 in chans:  # decoded split: box (4) + class (nc)
            box = next(o for o in outs if o.shape[1] == 4)
            cls = next(o for o in outs if o.shape[1] != 4)
            return np.concatenate([box, cls], axis=1)
        # Raw head: one output is class logits (num_classes channels), the other
        # is box DFL logits (4*reg_max). Both can be divisible by 4 (e.g. 80 and
        # 64), so identify the class tensor by the label count rather than order.
        num_classes = len(self.labels)
        cls_out = next((o for o in outs if o.shape[1] == num_classes), None)
        if cls_out is None:
            box_out, cls_out = outs[0], outs[1]  # fall back to positional order
        else:
            box_out = next(o for o in outs if o is not cls_out)
        return decode_raw_head(
            box_out, cls_out, int(self.height), int(self.width), score_threshold
        )

    def detect(self, frame_rgb, score_threshold=0.5, iou_threshold=0.45):
        """Run detection on an RGB frame (H, W, 3 uint8).

        Returns (boxes_xyxy, scores, class_ids) with boxes in *pixel*
        coordinates of the supplied frame, ready for the tracker.
        """
        import cv2

        orig_h, orig_w = frame_rgb.shape[:2]
        # Skip the resize when the caller already supplies a model-sized frame
        # (e.g. a hardware-scaled inference branch); boxes then come back in that
        # frame's pixel space.
        if (orig_h, orig_w) == (self.height, self.width):
            resized = frame_rgb
        else:
            resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(resized, axis=0)
        if self.is_float_input:
            input_data = input_data.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_detail["index"], input_data)
        self.interpreter.invoke()

        prediction = self._yolo_predictions(score_threshold)
        boxes_norm, scores, class_ids = decode_yolov8(
            prediction, score_threshold, iou_threshold, self.width, self.height
        )
        if boxes_norm.shape[0] == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                scores,
                class_ids,
            )

        # normalized (ymin, xmin, ymax, xmax) -> pixel (x1, y1, x2, y2).
        ymin, xmin, ymax, xmax = (
            boxes_norm[:, 0],
            boxes_norm[:, 1],
            boxes_norm[:, 2],
            boxes_norm[:, 3],
        )
        boxes_xyxy = np.stack(
            [xmin * orig_w, ymin * orig_h, xmax * orig_w, ymax * orig_h], axis=1
        ).astype(np.float32)
        return boxes_xyxy, scores, class_ids

    def label_for(self, class_id):
        """Human-readable label for a class id."""
        return self.labels[class_id] if class_id < len(self.labels) else str(class_id)
