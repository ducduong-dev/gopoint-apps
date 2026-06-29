#!/usr/bin/env python3

"""
Copyright 2026 NXP

SPDX-License-Identifier: BSD-3-Clause

Dual-output RTSP camera for the IP-camera object detection demo.

A single RTSP source is decoded once (hardware `v4l2*dec`) and split with a
`tee` into two branches, each finished by `imxvideoconvert_g2d` so the i.MX 2D
engine does the color-convert and scaling instead of OpenCV on the CPU:

  * **display** branch -- full-resolution BGR frames for the on-screen window;
  * **inference** branch -- frames hardware-scaled to the model's square input
    (default 640x640) in RGB, ready to hand straight to the detector.

`read()` pulls one frame from each branch (latest-only: the appsinks are
configured `drop=true max-buffers=1`), so the detector never pays for a
`cv2.resize`, and detection results in inference-space are simply scaled up to
the display frame for drawing. Only Gst + numpy are imported.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np

# Per-codec depay/parse/hardware-decode fragments. "Auto" uses uridecodebin to
# negotiate the whole RTSP -> raw path when the codec is unknown.
CODEC_DECODE = {
    "H.264": "rtph264depay ! h264parse ! v4l2h264dec",
    "H.265": "rtph265depay ! h265parse ! v4l2h265dec",
    "Auto": None,
}


class GstCamera:
    """Tee'd RTSP pipeline exposing a display and an inference appsink."""

    def __init__(self, url, codec, infer_size=640, display_size=(1280, 720)):
        self.infer_size = infer_size
        self.display_size = display_size

        width, height = display_size
        self.pipeline = Gst.parse_launch(
            self._build(url, codec, infer_size, int(width), int(height))
        )
        self.display_sink = self.pipeline.get_by_name("displaysink")
        self.infer_sink = self.pipeline.get_by_name("infersink")
        for sink in (self.display_sink, self.infer_sink):
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
        self.bus = self.pipeline.get_bus()

    @staticmethod
    def _source(url, codec):
        """RTSP source + depay/parse/hardware-decode fragment for a codec."""
        decode = CODEC_DECODE.get(codec)
        if decode is None:
            return f'uridecodebin uri="{url}" source::latency=200'
        return f'rtspsrc location="{url}" latency=200 protocols=tcp ! {decode}'

    @staticmethod
    def _build(url, codec, n, disp_w, disp_h):
        """Construct the gst-launch pipeline string.

        Both branches give imxvideoconvert_g2d an explicit output width+height:
        unconstrained, g2d negotiation under a tee is unreliable on i.MX (it can
        collapse to a small default), so the display size is a fixed choice the
        caller makes (`disp_w` x `disp_h`) rather than derived from the source.
        """
        # g2d outputs 4-channel BGRx/RGBA directly (no software videoconvert):
        # the alpha byte is dropped in numpy, keeping the whole path on the 2D
        # engine. Display is BGRx -> BGR, inference is RGBA -> RGB.
        display = (
            "queue max-size-buffers=2 leaky=downstream ! imxvideoconvert_g2d ! "
            f"video/x-raw,format=BGRx,width={disp_w},height={disp_h} ! "
            "appsink name=displaysink"
        )
        infer = (
            "queue max-size-buffers=2 leaky=downstream ! imxvideoconvert_g2d ! "
            f"video/x-raw,format=RGBA,width={n},height={n} ! appsink name=infersink"
        )
        return f"{GstCamera._source(url, codec)} ! tee name=t  t. ! {display}  t. ! {infer}"

    def start(self):
        """Set the pipeline rolling."""
        self.pipeline.set_state(Gst.State.PLAYING)

    @staticmethod
    def _sample_to_array(sample):
        """Convert a 4-channel BGRx/RGBA appsink sample to an (H, W, 3) array.

        g2d emits packed 32-bit pixels, so the row stride is width*4 (always
        4-byte aligned, no padding). The trailing x/alpha byte is dropped, giving
        BGR for the display branch and RGB for the inference branch.
        """
        buf = sample.get_buffer()
        struct = sample.get_caps().get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            flat = np.frombuffer(info.data, dtype=np.uint8, count=width * height * 4)
            frame = flat.reshape(height, width, 4)[:, :, :3]
            return frame.copy()
        finally:
            buf.unmap(info)

    def _pull(self, sink, timeout_ms):
        """Pull the latest frame from an appsink, or None on timeout."""
        sample = sink.emit("try-pull-sample", timeout_ms * Gst.MSECOND)
        if sample is None:
            return None
        return self._sample_to_array(sample)

    def read_infer(self, timeout_ms=2000):
        """Pull the latest 640x640 RGB inference frame, or None on timeout."""
        return self._pull(self.infer_sink, timeout_ms)

    def read_display(self, timeout_ms=2000):
        """Pull the latest display-resolution BGR frame, or None on timeout."""
        return self._pull(self.display_sink, timeout_ms)

    def has_error(self):
        """True if the pipeline bus reported an ERROR or end-of-stream."""
        msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        return msg is not None

    def stop(self):
        """Tear the pipeline down."""
        self.pipeline.set_state(Gst.State.NULL)
