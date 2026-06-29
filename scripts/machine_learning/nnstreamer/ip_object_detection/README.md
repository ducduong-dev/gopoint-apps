<!--
Copyright 2026 NXP
SPDX-License-Identifier: BSD-3-Clause
-->

# IP Camera Object Detection + Tracking

Real-time object detection and multi-object tracking on a single **RTSP / IP
camera** stream, for *GoPoint for i.MX Applications Processors*.

The demo decodes one configurable RTSP source with the board's hardware video
decoder, runs **YOLOv8** object detection on the **i.MX95 Neutron NPU**, and
tracks the detected objects across frames with **ByteTrack**. A GTK
configuration window sets up the stream and a separate OpenCV window shows the
annotated, real-time results (boxes, class labels, per-object track IDs and
motion trails, plus an FPS / track-count overlay).

## Supported platforms

| SoC      | Backend         | Notes                              |
| -------- | --------------- | ---------------------------------- |
| i.MX95   | Neutron NPU/CPU | Primary target (`yolov8s_neutron`) |

## Files

| File                       | Purpose                                                        |
| -------------------------- | ------------------------------------------------------------- |
| `ip_object_detection.py`   | Main app: GTK launcher + capture/inference/track loop          |
| `ip_object_detection.glade`| Configuration window definition                                |
| `gst_camera.py`            | Tee'd RTSP pipeline: full-res display + 640 inference branches |
| `detector.py`              | `Detector` — YOLOv8 decode on the Neutron NPU (numpy + TFLite) |
| `byte_tracker.py`          | ByteTrack tracker (`STrack`, `BYTETracker`), per-class ids     |
| `kalman_filter.py`         | ByteTrack Kalman filter (numpy + scipy)                        |
| `matching.py`              | IoU cost + linear assignment (SciPy Hungarian / numpy greedy)  |

## Pipeline

The RTSP source is decoded once (hardware `v4l2*dec`) and split with a `tee`
into two `imxvideoconvert_g2d` branches, so the i.MX 2D engine does all
color-convert/scale work off the CPU:

```text
rtspsrc/uridecodebin → decode → tee ┬→ g2d → BGRx WxH (display) → appsink
                                    └→ g2d → RGBA 640×640 (infer) → appsink
```

Both branches hand g2d an explicit output size (g2d's negotiation under a tee is
unreliable when unconstrained on i.MX) and emit 4-channel BGRx/RGBA so the alpha
byte is dropped in numpy with no software `videoconvert`. Detection runs on the
pre-scaled 640×640 branch (no `cv2.resize`); the inference-space boxes/tracks are
scaled up onto the display frame for drawing. The host-side YOLOv8 decode
thresholds on raw class logits *before* the DFL box decode, so only the few
surviving anchors are decoded.

The two branches are consumed by two threads: a **detection** thread (640 branch
→ YOLOv8 + ByteTrack → publishes a track snapshot) and a **display** thread
(display branch at camera rate → draws the latest snapshot). This decouples the
on-screen frame rate from inference, so slow detection does not stutter the
video. Measured live: display ≈ 12 FPS while detection ≈ 9 FPS, concurrently.

Measured on i.MX95 against a 2560×1440 H.265 camera: inference ≈ 41 ms (the model
delegates only 1 of 18 nodes to Neutron, the rest fall back to CPU/XNNPACK), host
decode ≈ 8 ms, end-to-end ≈ 9 FPS at a 1280×720 display (the display branch size
noticeably affects throughput — 1080p is ~half the rate). The display-resolution
control caps the display copy; detection is unaffected by it.

## Usage

The demo is normally launched from the GoPoint launcher (entry **"IP Camera
Detection"** under *Machine Learning → NNStreamer*). To run it directly on the
target:

```sh
python3 /root/gopoint-apps/scripts/machine_learning/nnstreamer/ip_object_detection/ip_object_detection.py
```

1. Wait for the model to download and warm up (the status bar shows progress;
   the first run is slow while the NPU graph compiles).
2. Enter the camera's **RTSP URL**, e.g.
   `rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101`.
3. Pick the stream **codec** and **backend** (NPU on i.MX95, or CPU). Note the
   codec is the *actual* stream codec, which may not match the URL name — pick
   **H.265** if H.264 yields no video. *Auto* (uridecodebin) is best-effort and
   can fail on cameras that also publish an audio track; prefer the explicit
   codec.
4. Pick the **Display** resolution. The display copy is hardware-scaled to this
   exact size, so choose the entry matching the camera's aspect ratio (most IP
   cameras are 16:9). A larger display lowers throughput; detection is unaffected
   (it always uses the 640×640 branch).
5. Adjust the **Confidence** and **NMS IoU** thresholds if needed, toggle
   **Show trails**, and press **Start**. Press **Stop** to return to the
   configuration window.

## Models and assets

These assets are bundled in this repo under `downloads/` (so they install to
`/root/gopoint-apps/downloads/` with the rest of the package). They are also
registered in `downloads.json` pointing at this repo's GitHub raw URLs, so
`utils.download_file()` can re-fetch them if missing:

| Asset                        | Used by              |
| ---------------------------- | -------------------- |
| `yolov8s_640_neutron.tflite` | NPU backend (i.MX95) |
| `yolov8s_640_int8.tflite`    | CPU backend          |
| `coco80_labels.txt`          | 80-class COCO labels |

## Dependencies

Runs against the board's preinstalled BSP: GStreamer (with `v4l2h264dec` /
`v4l2h265dec` and `imxvideoconvert_g2d`), OpenCV (`cv2`) built with the
GStreamer backend, a TFLite runtime + the Neutron external delegate
(`/usr/lib/libneutron_delegate.so`), GTK3 / PyGObject, NumPy and SciPy.

> **SciPy is required** by the ByteTrack Kalman filter (`kalman_filter.py`) and
> is not part of the default i.MX BSP — install it on the target if missing
> (`pip install scipy`). The assignment step in `matching.py` additionally
> degrades to a numpy greedy fallback when SciPy is absent, but the tracker as a
> whole still needs SciPy.

## Attribution

`byte_tracker.py`, `kalman_filter.py` and `matching.py` are vendored and trimmed
from [ByteTrack](https://github.com/ifzhang/ByteTrack) (Zhang et al., ECCV 2022),
used under the MIT License. The YOLOv8 decode pipeline is ported from
`../object_detection/object_detection_neutron_headless.py` in this repo.
