# VR Face Recognition System

This repository contains a prototype that shows how face detection and recognition can be visualised directly inside a VR headset.  A small Python application performs the heavy lifting of detecting faces on the PC screen and matching them against a database.  A Unity project receives the recognition results via UDP and renders a frame and the detected name around each head in 3D space.

## Overview

1. **Python face detection and recognition**
   - The screen of the PC is captured using `mss` and converted to RGB for further processing.
   - Faces are detected with **MTCNN**.  Each detection is cropped and converted to a tensor that is passed to the face recognition model.
   - A pre‑trained **PocketNet S128** model generates embeddings which are compared to templates stored in an SQLite database.  Only the model path and threshold are configurable here – the actual PocketNet implementation is left untouched and has its own documentation.
   - For every frame a JSON payload with normalised face centre, size, colour and name is sent to Unity via UDP.

2. **Unity visualisation**
   - Unity listens on the UDP port and spawns a `FaceUI` object for each unique name received.
   - `FaceUIManager` creates OpenVR overlays that draw a rectangular frame and a name label in the headset.
   - Face positions are updated whenever a new UDP packet arrives while the headset is not moving.

3. **HMD movement handling**
   - `FaceTrackingManager` tracks the headset rotation and keeps a simple state machine with two states: **Moving** and **Still**.
   - When the headset rotates quickly, the state switches to *Moving* and existing face overlays keep their last position, effectively locking them in world space.
   - Once the headset is still for a short time, the state returns to *Still* and incoming face data once again updates the overlay positions each frame.

## Repository layout

- `src/face_detection/MTCNN_FD.py` – captures the chosen window, performs detection and recognition and sends UDP packages to Unity.
- `src/face_recognition/pocketnet_face.py` – wrapper around the PocketNet S128 model used to generate embeddings.
- `src/db/db_manager.py` – SQLite database helper to register users and fetch their templates.
- `src/Unity Project/HMD_FaceRecognition` – Unity project that visualises the received faces in VR.
- `src/main.py` – entry point that starts the Python overlay and selects the target window to capture.

## Getting started

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Register example images as users:

```bash
python src/db/db_manager.py
```

3. Launch the Python overlay:

```bash
python src/main.py
```

4. Open the Unity project in the `src/Unity Project/HMD_FaceRecognition` folder and press **Play**.  As the Python script sends detections, you will see coloured frames and names appear around each recognised head in VR.

GPU acceleration is used automatically when CUDA is available.
