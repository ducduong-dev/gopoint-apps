# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the **example applications package** for *GoPoint for i.MX Applications Processors* (formerly NXP Demo Experience). It is **not a standalone application** — it is a collection of self-contained demo apps (Python scripts, GTK/Glade UIs, shell scripts, and assets) that get installed to `/root/gopoint-apps/` on an i.MX EVK (Evaluation Kit) board and are launched by a separate `demoexperience` binary (the launcher GUI/TUI, which lives in the [nxp-demo-experience](https://github.com/nxp-imx-support/nxp-demo-experience) repo).

There is **no build system, package manager, or test suite** in this repo — no `package.json`, `Makefile`, `pyproject.toml`, or `requirements.txt`. Demos run on the embedded target, against the board's preinstalled Linux BSP libraries (GStreamer, NNStreamer, TensorFlow Lite, OpenGL/G2D, GTK3, PyGObject). You generally cannot run them on the host; treat changes as edits to scripts that execute on hardware.

## Target device sync

This repo is synced with a copy of the same repo on the i.MX device at `root@192.168.1.27:/root/gopoint-apps`. Any change made in this repo should be synced over to the device copy so the two stay in step. The device has no `rsync`; use `scp` (SSH key auth works non-interactively).

## Installing packages on the target

The board can be missing Python packages a demo needs (e.g. `scipy`). You **may install Python packages on the target when necessary** to make a demo run — prefer the system package manager, else `pip`. Keep new runtime dependencies minimal and documented in the demo's README, and note that a clean board may need the same install.

## Architecture

**`demos.json` is the registry.** Every demo the launcher shows is an entry here, grouped as `Category -> Subcategory -> [demos]`. Categories include Machine Learning (NNStreamer), Multimedia (GStreamer), TSN, Security (ELE), Communication (CAN Bus), and GPU (OpenVG 2D). Each demo entry carries:
- `executable` — the command the launcher runs, with **hardcoded `/root/gopoint-apps/...` paths** (e.g. `python3 /root/gopoint-apps/scripts/.../foo.py`).
- `compatible` — comma-separated list of SoCs the demo supports (e.g. `imx8mp, imx93, imx95`). Keep this accurate; the launcher uses it to filter what is shown per board.
- `screenshot` / `icon` — filenames resolved against the `screenshot/` and `icon/` directories.

When you **add or rename a demo script, you must update `demos.json`** (path, `compatible`, assets) or the launcher will not find it.

**Asset directories** are flat and referenced by filename from JSON, not by relative path: `screenshot/` (preview images named in `demos.json`), `icon/` (SVG icons named in `demos.json`), `data/` (UI imagery used by demos), `downloads/` (pre-bundled ML models / label files), `licenses/`.

**Runtime asset downloading.** Large models and media are fetched on first run rather than committed. `downloads.json` maps a filename to `{url, alt_url, sha}`. `scripts/utils.py:download_file()` downloads into `/root/gopoint-apps/downloads/`, falling back to `alt_url` (the `nxp-demo-experience-assets` mirror), and verifies the SHA-1. When a demo needs a new model/asset, add it to `downloads.json` and fetch it via `utils.download_file(name)` — do not commit large binaries.

**`scripts/utils.py` is the shared helper module.** Demos import it via:
```python
sys.path.append("/root/gopoint-apps/scripts/")
import utils
```
It provides `download_file()` (asset fetch + SHA check) and `run_check()` (detects working V4L2 capture devices under `/dev/video*`). Reuse these instead of reimplementing camera probing or downloads.

**GTK + Glade demo pattern.** Most GUI demos follow the same shape: a `.glade` file defines the window/widgets; the Python launcher does `Gtk.Builder().add_from_file(<absolute glade path>)`, `builder.get_object(...)` for each widget, and `builder.connect_signals(self)`. UI work runs off the main GTK thread via a local `@threaded` decorator (spawns a `threading.Thread`), and updates the UI back on the main loop with `GLib.idle_add`. The glade file path is hardcoded absolute (`/root/gopoint-apps/scripts/.../foo.glade`). When editing UI, change both the `.glade` and the matching widget IDs in the `.py`.

**TUI entry point.** `scripts/run_tui.sh` is how the launcher is started in text mode on a board: it ensures `/root/gopoint-apps/downloads` exists and runs `/usr/bin/demoexperience tui`.

## Conventions

- **Python style is `black`** (see README badge). Format any Python you touch with `black` before considering it done.
- **License headers** are required on source files. Scripts/shell use `SPDX-License-Identifier: BSD-3-Clause` with an `NXP` copyright line and year range; preserve and update the year when editing. Note the repository as a whole is under the proprietary `LA_OPT_Online Code Hosting` license (`LICENSE.txt`) — these coexist.
- **Absolute install paths.** Scripts assume they live under `/root/gopoint-apps/`. New scripts should follow the same absolute-path convention for sibling scripts, glade files, and downloads rather than computing paths relative to `__file__`.
- Demos are organized by domain under `scripts/`: `machine_learning/` (with a large `nnstreamer/` subtree), `multimedia/`, `audio/`, `communication/`, `opengl/`, `TSN/`. Many demo subdirectories carry their own `README.md` with device-specific setup — read it before modifying that demo.
