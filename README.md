# PocketNet Face Recognition Overlay

This project provides a real-time face recognition demo using the lightweight PocketNet model. It captures a selected window on the desktop, detects faces with MTCNN and identifies them using embeddings from PocketNet. Recognition results are drawn on a transparent PyQt overlay.

## Features

- **Real-time detection and recognition** from any chosen window using screen capture.
- **PocketNet** model for compact face embeddings loaded from `src/face_engine/models`.
- **MTCNN** based face detector with optional GPU acceleration.
- **SQLite user database** managed by `src/db/db_manager.py`.
- **Face tracking and automatic re-evaluation** to keep identities stable over time.
- Sample images under `src/test_images` and a helper script to register them as users.

## Repository layout

- `src/face_engine/pocketnet_face.py` – wraps the PocketNet model and performs detection and embedding extraction.
- `src/live_feed_capture/capture_window/MTCNN_FD.py` – main overlay logic for capturing the screen, running recognition and drawing results.
- `src/db/db_manager.py` – simple database manager with functions to register and query users.
- `src/main.py` – launches the overlay targeting a window title (default `"explorer"`).
- `PocketNetCode` – original PocketNet training code (included for reference).

## Getting started

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Populate the user database with the provided sample images:

```bash
python src/db/db_manager.py
```

3. Run the overlay demo:

```bash
python src/main.py
```

The script searches for a visible window containing `"explorer"` in its title and overlays recognition results on top of it. Edit the constant `target_window_title` in `src/main.py` to select a different window. The recognition threshold can be changed via `THRESHOLD` in `src/live_feed_capture/capture_window/MTCNN_FD.py`.

GPU acceleration is used automatically when a CUDA device is available.
