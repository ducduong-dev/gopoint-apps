#!/usr/bin/env python3

"""
Copyright 2026 NXP

SPDX-License-Identifier: BSD-3-Clause

IP-camera object detection + tracking demo for GoPoint (i.MX95).

Streams from a single RTSP/IP camera, decodes with the board's hardware video
decoder, runs YOLOv8 object detection on the Neutron NPU (see detector.py), and
tracks the detected objects across frames with ByteTrack (see byte_tracker.py).
A GTK+Glade window configures the stream (RTSP URL, codec, backend, thresholds)
and a separate OpenCV window shows the annotated, real-time results.

The configuration window follows the same shape as the other NNStreamer demos in
this repo: a .glade file defines the widgets, a threaded preload downloads and
warms up the model, and UI updates are marshalled back onto the GTK main loop
with GLib.idle_add.

Capture/inference/display run on two worker threads so a slow inference never
stutters the video: a detection thread pulls the 640 branch, runs YOLOv8 +
ByteTrack, and publishes an immutable track snapshot; a display thread pulls the
display branch at camera rate and draws the latest snapshot. The GTK main loop
only does cv2.imshow of the most recent annotated frame.
"""

import os
import sys
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gtk as gtk
from gi.repository import GLib, Gst

# Local modules (same directory, on the absolute install path).
sys.path.append(
    "/root/gopoint-apps/scripts/machine_learning/nnstreamer/ip_object_detection"
)
from detector import Detector
from byte_tracker import BYTETracker, BaseTrack
from gst_camera import GstCamera, CODEC_DECODE

GLADE_FILE = (
    "/root/gopoint-apps/scripts/machine_learning/nnstreamer/"
    "ip_object_detection/ip_object_detection.glade"
)

WINDOW_NAME = "i.MX IP Camera Object Detection + Tracking"

# Codec choices for the UI; the depay/parse/decode mapping lives in gst_camera.
CODECS = list(CODEC_DECODE)

TRAIL_LENGTH = 32
"""Number of past center points kept per track for the motion trail."""

# Display branch resolution options (label -> (width, height)). The display copy
# is hardware-scaled by imxvideoconvert_g2d to exactly this size, so a high-res
# sensor does not dominate draw/imshow; detection is unaffected (separate 640
# branch). Pick the entry matching the camera's aspect (most IP cameras are 16:9).
DISPLAY_OPTIONS = {
    "1920x1080": (1920, 1080),
    "1280x720": (1280, 720),
    "960x540": (960, 540),
    "640x640": (640, 640),
}
DEFAULT_DISPLAY = "1280x720"


def threaded(fn):
    """Run a method on its own thread, off the GTK main loop."""

    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

    return wrapper


def color_for_id(track_id):
    """Deterministic, well-spread BGR color for a track id."""
    # Golden-ratio hue stepping gives visually distinct colors per id.
    hue = int((track_id * 0.61803398875 * 180) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class IPObjectDetection:
    """GTK launcher + capture/inference/track loop for the IP-camera demo."""

    def __init__(self):
        self.builder = gtk.Builder()
        self.builder.add_from_file(GLADE_FILE)
        self.builder.connect_signals(self)

        # Widgets
        self.window = self.builder.get_object("main-window")
        self.rtsp_entry = self.builder.get_object("rtsp-entry")
        self.codec_list = self.builder.get_object("codec-list")
        self.display_list = self.builder.get_object("display-list")
        self.backend_list = self.builder.get_object("backend-list")
        self.score_scale = self.builder.get_object("score-scale")
        self.iou_scale = self.builder.get_object("iou-scale")
        self.trails_check = self.builder.get_object("trails-check")
        self.start_button = self.builder.get_object("start-button")
        self.stop_button = self.builder.get_object("stop-button")
        self.status_bar = self.builder.get_object("status-bar")
        self.progress_bar = self.builder.get_object("progress-bar")
        self.about_dialog = self.builder.get_object("about-dialog")

        # Platform (selects available backends).
        try:
            with open("/sys/devices/soc0/soc_id", encoding="utf-8") as soc:
                self.platform = soc.read().strip()
        except OSError:
            self.platform = "unknown"

        # Populate codec choices. Default to H.265 to match the preset camera
        # (its stream is H.265 despite the ".264" name); change per camera.
        for codec in CODECS:
            self.codec_list.append_text(codec)
        default_codec = "H.265" if "H.265" in CODECS else CODECS[0]
        self.codec_list.set_active(CODECS.index(default_codec))

        # Populate display-resolution choices.
        for label in DISPLAY_OPTIONS:
            self.display_list.append_text(label)
        self.display_list.set_active(list(DISPLAY_OPTIONS).index(DEFAULT_DISPLAY))

        # Populate backends. Neutron NPU is the i.MX95 fast path; CPU always works.
        backends = []
        if self.platform == "i.MX95":
            backends.append("NPU")
        backends.append("CPU")
        for backend in backends:
            self.backend_list.append_text(backend)
        self.backend_list.set_active(0)

        # Runtime state.
        self.detector = None
        self.tracker = None
        self.camera = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.display_timeout_id = None
        self.trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

        # Latest detection snapshot published by the detection thread and drawn
        # by the (faster) display thread. Decoupling the two keeps the on-screen
        # video at camera rate even though inference is slower.
        self.snapshot = []  # list of dicts: box (infer px), id, label, score, trail
        self.snapshot_lock = threading.Lock()
        self.detect_fps = 0.0

        # Progress-bar pulse bookkeeping.
        self.pulsing = False
        self.progress_bar.set_show_text(False)

        self.window.connect("delete-event", self.quit_app)
        self.window.show()

        # Preload model off the main thread.
        threading.Thread(target=self.preload, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Preload / model warm-up
    # ------------------------------------------------------------------ #

    def on_timeout(self):
        """Pulse the progress bar while a long operation runs."""
        if self.pulsing:
            self.progress_bar.set_show_text(True)
            self.progress_bar.pulse()
            return True
        self.progress_bar.set_show_text(False)
        self.progress_bar.set_fraction(0.0)
        return False

    def set_status(self, text):
        """Thread-safe status-bar update."""
        GLib.idle_add(self.status_bar.set_text, text)

    def preload(self):
        """Download + load + warm up the detector (runs in a worker thread)."""
        self.set_controls(False)
        self.pulsing = True
        GLib.timeout_add(50, self.on_timeout)

        backend = "neutron" if self.backend_list.get_active_text() == "NPU" else "cpu"
        self.set_status("Downloading and loading YOLOv8 model (first run is slow)...")
        try:
            self.detector = Detector(backend=backend)
        except Exception as error:  # surface any download/delegate failure
            self.pulsing = False
            self.set_status(f"Model load failed: {error}")
            return

        self.pulsing = False
        self.set_status("Ready. Enter an RTSP URL and press Start.")
        self.set_controls(True)

    def set_controls(self, sensitive):
        """Enable/disable the configuration widgets together."""
        for widget in (
            self.start_button,
            self.rtsp_entry,
            self.codec_list,
            self.display_list,
            self.backend_list,
        ):
            GLib.idle_add(widget.set_sensitive, sensitive)

    # ------------------------------------------------------------------ #
    # Start / stop
    # ------------------------------------------------------------------ #

    def start(self, widget):
        """Open the stream and kick off capture + display (GTK main thread)."""
        if self.detector is None:
            self.set_status("Model not ready yet, please wait...")
            return

        url = self.rtsp_entry.get_text().strip()
        if not url or url == "rtsp://":
            self.set_status("Please enter a valid RTSP URL first.")
            return

        codec = self.codec_list.get_active_text()
        display_size = DISPLAY_OPTIONS.get(
            self.display_list.get_active_text(), (1280, 720)
        )
        self.set_status("Connecting to camera...")
        try:
            self.camera = GstCamera(
                url,
                codec,
                infer_size=self.detector.width,
                display_size=display_size,
            )
            self.camera.start()
        except Exception as error:
            self.set_status(f"Could not open the stream: {error}")
            self.camera = None
            return

        # Fresh tracker + trails per session so ids restart at 1.
        BaseTrack.reset_id()
        self.tracker = BYTETracker(
            track_thresh=self.score_scale.get_value() / 100.0,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
        )
        self.trails.clear()
        with self.snapshot_lock:
            self.snapshot = []
        self.detect_fps = 0.0

        self.running = True
        self.set_controls(False)
        self.stop_button.set_sensitive(True)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Detection and display run on separate threads so a slow inference does
        # not stutter the on-screen video; the main loop only does imshow.
        self.detection_loop()
        self.display_loop()
        self.display_timeout_id = GLib.timeout_add(33, self.update_display)

    def stop(self, widget):
        """Stop the stream and tear down the display."""
        self.running = False
        if self.display_timeout_id is not None:
            GLib.source_remove(self.display_timeout_id)
            self.display_timeout_id = None
        time.sleep(0.1)
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        cv2.destroyAllWindows()
        with self.frame_lock:
            self.current_frame = None
        self.set_controls(True)
        self.stop_button.set_sensitive(False)
        self.set_status("Stopped. Ready to start again.")

    # ------------------------------------------------------------------ #
    # Detection thread (640 branch) and display thread (display branch)
    # ------------------------------------------------------------------ #

    @threaded
    def detection_loop(self):
        """Pull the 640 branch, detect + track, publish a snapshot for display."""
        fps = 0.0
        last = time.perf_counter()

        while self.running and self.camera is not None:
            infer_rgb = self.camera.read_infer()
            if infer_rgb is None:
                if not self.running:
                    break
                self.set_status("Stream interrupted, reconnecting...")
                time.sleep(0.5)
                if not self._reconnect():
                    break
                continue

            score_thr = self.score_scale.get_value() / 100.0
            iou_thr = self.iou_scale.get_value() / 100.0

            # infer_rgb is already model-sized RGB (hardware-scaled), so detect()
            # skips the resize; boxes come back in inference (infer_size) pixels.
            boxes, scores, class_ids = self.detector.detect(
                infer_rgb, score_threshold=score_thr, iou_threshold=iou_thr
            )
            tracks = self.tracker.update(boxes, scores, class_ids)

            snapshot = self._build_snapshot(tracks)
            with self.snapshot_lock:
                self.snapshot = snapshot

            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)
            last = now
            self.detect_fps = fps

    @threaded
    def display_loop(self):
        """Pull the display branch at camera rate and draw the latest snapshot."""
        fps = 0.0
        last = time.perf_counter()
        backend = self.backend_list.get_active_text()
        infer_size = self.detector.width

        while self.running and self.camera is not None:
            frame = self.camera.read_display()
            if frame is None:
                if not self.running:
                    break
                time.sleep(0.05)  # detection thread owns reconnect; just wait
                continue

            disp_h, disp_w = frame.shape[:2]
            scale_x = disp_w / infer_size
            scale_y = disp_h / infer_size
            with self.snapshot_lock:
                snapshot = self.snapshot
            self.annotate(
                frame, snapshot, self.trails_check.get_active(), scale_x, scale_y
            )

            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)
            last = now
            self.draw_hud(frame, self.detect_fps, fps, len(snapshot), backend)

            with self.frame_lock:
                self.current_frame = frame

    def _build_snapshot(self, tracks):
        """Build an immutable per-track snapshot (boxes/trails in inference px).

        Trails are maintained here at detection rate; the published list is never
        mutated after hand-off, so the display thread can read it lock-free.
        """
        active_ids = set()
        snapshot = []
        for track in tracks:
            active_ids.add(track.track_id)
            x1, y1, x2, y2 = (float(v) for v in track.tlbr)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.trails[track.track_id].append((cx, cy))
            snapshot.append(
                {
                    "box": (x1, y1, x2, y2),
                    "id": track.track_id,
                    "label": self.detector.label_for(track.class_id),
                    "score": float(track.score),
                    "trail": list(self.trails[track.track_id]),
                }
            )
        for stale in [tid for tid in self.trails if tid not in active_ids]:
            del self.trails[stale]
        return snapshot

    def _reconnect(self):
        """Try to reopen the stream after an interruption."""
        if self.camera is not None:
            self.camera.stop()
        url = self.rtsp_entry.get_text().strip()
        codec = self.codec_list.get_active_text()
        display_size = DISPLAY_OPTIONS.get(
            self.display_list.get_active_text(), (1280, 720)
        )
        try:
            self.camera = GstCamera(
                url,
                codec,
                infer_size=self.detector.width,
                display_size=display_size,
            )
            self.camera.start()
        except Exception as error:
            self.set_status(f"Reconnect failed ({error}). Press Stop and retry.")
            self.camera = None
            return False
        self.set_status("Reconnected.")
        return True

    def annotate(self, frame, snapshot, show_trails, scale_x=1.0, scale_y=1.0):
        """Draw a detection snapshot, scaling inference-space coords to ``frame``."""
        for det in snapshot:
            bx1, by1, bx2, by2 = det["box"]
            x1, y1 = int(bx1 * scale_x), int(by1 * scale_y)
            x2, y2 = int(bx2 * scale_x), int(by2 * scale_y)
            color = color_for_id(det["id"])
            caption = f"{det['label']} {det['score'] * 100:.0f}% #{det['id']}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                frame,
                caption,
                (x1 + 1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            if show_trails:
                pts = det["trail"]
                for i in range(1, len(pts)):
                    p0 = (int(pts[i - 1][0] * scale_x), int(pts[i - 1][1] * scale_y))
                    p1 = (int(pts[i][0] * scale_x), int(pts[i][1] * scale_y))
                    cv2.line(frame, p0, p1, color, 2)

    def draw_hud(self, frame, detect_fps, display_fps, n_tracks, backend):
        """Overlay detection/display FPS, track count and backend (top-left)."""
        text = (
            f"Det: {detect_fps:4.1f}  Disp: {display_fps:4.1f}  "
            f"Tracks: {n_tracks}  {backend}"
        )
        cv2.rectangle(frame, (0, 0), (470, 28), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    # Display (GTK main thread)
    # ------------------------------------------------------------------ #

    def update_display(self):
        """Show the latest annotated frame; called on the GTK main loop."""
        if not self.running:
            return False
        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()
        if frame is not None:
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)
        return True

    # ------------------------------------------------------------------ #
    # Misc UI handlers
    # ------------------------------------------------------------------ #

    def about_button_activate(self, widget):
        """Show the About dialog."""
        self.about_dialog.run()
        self.about_dialog.hide()
        return True

    def quit_app(self, *args):
        """Tear down everything and quit."""
        self.running = False
        time.sleep(0.1)
        if self.camera is not None:
            self.camera.stop()
        cv2.destroyAllWindows()
        gtk.main_quit()


if __name__ == "__main__":
    os.environ.setdefault(
        "VIV_VX_CACHE_BINARY_GRAPH_DIR", "/root/gopoint-apps/downloads"
    )
    os.environ.setdefault("VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "1")
    Gst.init(None)
    APP = IPObjectDetection()
    gtk.main()
