# === Standard Library Imports ===
import sys
import socket
import json
import time

# === Third-Party Imports ===
import cv2
import numpy as np
import torch
import mss
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from facenet_pytorch import MTCNN

# === Project-Specific Imports ===
from src.face_recognition.pocketnet_face import FaceEngine
from src.db.db_manager import DBManager
from config.paths import DB_PATH

# === Configuration Flags ===
debug = True
PRINT_ON_SCREEN = False  # Show face boxes on the screen

# === Recognition & Tracking Parameters ===

PROBATION_PERIOD = 3              # Frames a face must be consistently detected before confirmation
TRACKING_DISTANCE_THRESHOLD = 180 # Max pixel distance to match detections to existing trackers
MAX_MISSING_FRAMES = 3            # Max consecutive frames a face can go undetected before being discarded

MATCH_DECAY = 50                  # Frames to wait before re-evaluating a known face
NO_MATCH_DECAY = 20               # Frames to wait before retrying recognition on an "Unknown" face

THRESHOLD = 0.9242                # Cosine similarity threshold to consider a face match


# === Initialize Logging Helper ===
def DEBUG_log(message):
    if debug:
        print(f"[DEBUG] {message}")


# === Initialize Database ===
db = DBManager(DB_PATH)

# === Device Setup ===
print("CUDA available:", torch.cuda.is_available())
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"GPU: {device_name}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG_log(f"Using device: {device}")

# === Initialize Face Detection (MTCNN) ===
mtcnn = MTCNN(keep_all=True, min_face_size=40, device=device)

# === Initialize Face Recognition Engine ===
DEBUG_log("Initializing Face Recognition Engine...")
try:
    engine = FaceEngine(device=str(device))
except Exception as e:
    print(f"[FATAL ERROR] Could not initialize the face recognition system: {e}")
    sys.exit(1)


# --- Helper Classes & Functions ---

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2))


def find_best_user_match(
        embedding: np.ndarray,
        db: DBManager,
        debug_log: callable
) -> (dict, float):
    """
    Compare `embedding` to every template of every user in `db`,
    compute each user's **maximum** similarity, debug-print them,
    and return the best-matching user dict and its score.
    """
    best_user = None
    best_score = -1.0

    for user in db.get_all_users():
        temps = user['templates']
        if not temps:
            debug_log(f"[SKIP] User {user['id']} '{user['name']}' has no templates.")
            continue

        # compute per-template similarities
        sims = [_cosine_similarity(embedding, t) for t in temps]
        max_sim = max(sims)

        # Debug: show max similarity instead of average
        debug_log(
            f"User {user['id']} '{user['name']}' max sim: {max_sim:.4f} "
            f"(from {len(sims)} templates)"
        )

        if max_sim > best_score:
            best_score = max_sim
            best_user = user

        # Uncomment to use average instead
        # With the current model (pocketNetS_128_pretrained.pth), images of the same user under
        # different conditions (lighting, angle) produce rather different templates. As a result,
        # averaging ultimately changes the task to "find the user with the most templates under similar conditions."
        # Ideally, we would have multiple templates of each user in different lighting conditions to
        # increase the likelihood of one matching the current inference conditions.
        # Therefore, we use maximum instead of average for now.

        # avg_sim = sum(sims) / len(sims)
        # debug_log(
        #     f"User {user['id']} '{user['name']}' avg sim: {avg_sim:.4f} "
        #     f"(from {len(sims)} templates)"
        # )
        # if avg_sim > best_score:
        #     best_score = avg_sim
        #     best_user = user

    return best_user, best_score


class FaceTracker:
    """A class to track the state and properties of a detected face."""

    WINDOW_SIZE = 15  # Number of frames to keep in history for confidence

    def __init__(self, box):
        self.box = box
        self.center = self.get_center(box)
        self.id = None
        self.frames_since_seen = 0
        self.confirmed = False
        self.hits = 1
        self.history = [True]
        # --- State for recognition ---
        self.recognized = False
        self.name = None
        self.sim_score = 0
        self.frames_since_recognition = 0
        self.user = None

    def get_center(self, box):
        x, y, w, h = box
        return x + w // 2, y + h // 2

    def update(self, box):
        self.box = box
        self.center = self.get_center(box)
        self.frames_since_seen = 0
        self.hits += 1
        self._append_history(True)

    def register_miss(self):
        self.frames_since_seen += 1
        self._append_history(False)

    def _append_history(self, value: bool):
        self.history.append(value)
        if len(self.history) > self.WINDOW_SIZE:
            self.history.pop(0)

    def detection_confidence(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# --- Main Application Overlay ---

class Overlay(QtWidgets.QWidget):
    def __init__(self, target_win):
        super().__init__(None, QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        self.target = target_win
        self.active_trackers = []
        self.next_face_id = 1
        self.max_missing_frames = MAX_MISSING_FRAMES
        self.tracking_threshold = TRACKING_DISTANCE_THRESHOLD
        self.probation_period = PROBATION_PERIOD

        self.window_minimized_known = False
        self.window_closed_known = False

        # ── bring up UDP socket for Unity comms ──
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_ip = "127.0.0.1"
        self.unity_port = 5005

        DEBUG_log(f"Overlay initialized. UDP socket: {self.unity_ip}:{self.unity_port}")
        QtCore.QTimer(self, interval=50, timeout=self.sync_position).start()
        QtCore.QTimer(self, interval=50, timeout=self.capture_and_detect).start()

    def sync_position(self):
        try:
            geom = self.target.box
            self.setGeometry(geom.left, geom.top, geom.width, geom.height)
        except Exception as e:
            if not self.window_closed_known:
                DEBUG_log(f"sync error: {e}")
                print("[ERROR] The selected window was closed!")
                self.window_closed_known = True
            self.close()
            quit()

    def capture_and_detect(self):
        try:
            # 1. CAPTURE SCREEN
            geom = self.target.box
            capture_area = {"top": geom.top, "left": geom.left, "width": geom.width, "height": geom.height}
            with mss.mss() as sct:
                sct_img = sct.grab(capture_area)

            frame_bgr = np.array(sct_img)[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 2. DETECT FACES WITH MTCNN
            try:
                boxes, _ = mtcnn.detect(frame_rgb)
                ts = time.time()
            except Exception as e:
                if 'non-empty list' in str(e):
                    if not self.window_minimized_known:
                        DEBUG_log("mtcnn.detect internal error (no initial faces found), handled.")
                        print("[ERROR] The selected window was minimized!")
                        self.window_minimized_known = True
                        quit()

                    boxes = None
                else:
                    DEBUG_log(f"An unexpected error occurred in mtcnn.detect: {e}")
                    boxes = None

            current_detections = []
            if boxes is not None:
                self.window_minimized_known = False
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    current_detections.append([x1, y1, w, h])

            # 3. APPLY TRACKING LOGIC
            unmatched_detections = list(current_detections)
            for tracker in self.active_trackers:
                best_match_box = None
                min_dist = self.tracking_threshold

                tracker_center = tracker.get_center(tracker.box)
                for detection in unmatched_detections:
                    det_center = tracker.get_center(detection)
                    dist = distance(tracker_center, det_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_match_box = detection

                if best_match_box is not None:
                    tracker.update(best_match_box)
                    unmatched_detections.remove(best_match_box)
                else:
                    tracker.register_miss()

            for detection in unmatched_detections:
                self.active_trackers.append(FaceTracker(detection))

            self.active_trackers = [t for t in self.active_trackers if t.frames_since_seen <= self.max_missing_frames]

            # 4. PERFORM RECOGNITION AND RE-EVALUATION
            for tracker in self.active_trackers:
                # First, confirm new trackers if they pass the probation period
                if not tracker.confirmed and tracker.hits >= self.probation_period:
                    tracker.confirmed = True
                    tracker.id = self.next_face_id
                    self.next_face_id += 1
                    DEBUG_log(f"New face confirmed: ID {tracker.id}")

                # If a face has already been recognized, start its decay counter.
                if tracker.recognized:
                    tracker.frames_since_recognition += 1

                    is_match = (tracker.name != "Unknown")
                    decay_threshold = MATCH_DECAY if is_match else NO_MATCH_DECAY

                    # Check if the decay period has been reached
                    if tracker.frames_since_recognition >= decay_threshold:
                        # Now, wait for a high-confidence frame to perform the re-check
                        if tracker.detection_confidence() == 1.0:
                            DEBUG_log(
                                f"Face ID {tracker.id} ('{tracker.name}') met decay. Re-evaluating on high-confidence frame.")
                            tracker.recognized = False  # Flag for re-recognition

                # --- RECOGNITION LOGIC (Handles both initial and re-evaluations) ---
                # If a face is confirmed but not yet recognized (or flagged for re-recognition)
                if tracker.confirmed and not tracker.recognized:
                    DEBUG_log(f"Running recognition for Face ID {tracker.id}...")
                    x, y, w, h = map(int, tracker.box)
                    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                    crop_y1, crop_y2 = max(0, y - pad_y), min(frame_rgb.shape[0], y + h + pad_y)
                    crop_x1, crop_x2 = max(0, x - pad_x), min(frame_rgb.shape[1], x + w + pad_x)
                    face_crop_img = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

                    if face_crop_img.size > 0:
                        pil_img = Image.fromarray(face_crop_img)
                        try:
                            face_tensor = mtcnn(pil_img, save_path=None)
                        except Exception as e:
                            face_tensor = None
                            if 'non-empty list' in str(e):
                                DEBUG_log(f"mtcnn() error on crop for Face ID {tracker.id}: Face too small. [handled]")
                            else:
                                DEBUG_log(f"An unexpected error in mtcnn() on crop: {e}")

                        embedding = engine.embed_tensor(face_tensor)

                        if embedding is not None:

                            best_user, score = find_best_user_match(embedding, db, DEBUG_log)

                            tracker.sim_score = score
                            if best_user and score >= THRESHOLD:
                                # Existing user
                                tracker.name = best_user['name']
                                tracker.user = best_user

                            else:
                                """
                                # This would save Unknown users in the DB so we can reidentify them even though 
                                # they are not registered
                                # No match → register as "Unknown"
                                unknown_color = "#888888"
                                new_id = db.register_user(
                                    name="Unknown",
                                    color=unknown_color,
                                    templates=[embedding],
                                    registered_user=False,  # flag as unregistered placeholder
                                    last_seen=None
                                )
                                # build the same shape of dict as get_all_users returns
                                tracker.user = {
                                    'id': new_id,
                                    'name': "Unknown",
                                    'color': unknown_color,
                                    'templates': [embedding],
                                    'registered_user': False,
                                    'last_seen': None
                                }
                                
                                """
                                tracker.name = "Unknown"

                            DEBUG_log(
                                f"Face ID {tracker.id} processed. "
                                f"Name: '{tracker.name}', Score: {tracker.sim_score:.4f}"
                            )

                        else:
                            tracker.name = "Unknown"
                            tracker.sim_score = 0.0
                    else:
                        tracker.name = "Unknown"
                        tracker.sim_score = 0.0

                    # Mark as recognized and reset the decay counter
                    tracker.recognized = True
                    tracker.frames_since_recognition = 0

            # build & send one JSON packet per frame ──
            h, w = frame_rgb.shape[:2]

            # build one batch per frame:
            payload = []
            ts = time.time()
            for t in self.active_trackers:
                if not t.confirmed: continue

                # Get face center and bounding box dimensions
                box = t.box
                face_w = box[2]
                face_h = box[3]
                cx, cy = t.center

                # normalize all screen-space values to [-1,1] for center, and [0,1] for size
                norm_x = (cx / w - 0.5) * 2.0
                norm_y = -(cy / h - 0.5) * 2.0  # Y is flipped to match Unity's screen space
                norm_w = face_w / w
                norm_h = face_h / h

                # Get user color if available, else fallback to gray
                color = "#888888"
                if t.user and "color" in t.user:
                    color = t.user["color"]

                # --- PAYLOAD ---
                payload.append({
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4),
                    "w": round(norm_w, 4),
                    "h": round(norm_h, 4),
                    "name": t.name,
                    "color": color,
                    "timestamp": round(ts, 4)
                })

            packet = json.dumps(payload).encode("utf-8")
            self.sock.sendto(packet, (self.unity_ip, self.unity_port))

            if payload:
                # uncomment for UDP debugging
                # DEBUG_log(f"Sent payload: {payload[0]}")
                pass

            self.update()  # trigger paintEvent()

        except Exception as e:
            if not self.window_closed_known:
                DEBUG_log(f"capture_and_detect error: {e}")

    def paintEvent(self, event):
        # display the detected faces on the PC Screen (not related to the HMD visualization)
        if PRINT_ON_SCREEN:
            painter = QtGui.QPainter(self)

            for tracker in self.active_trackers:
                x, y, w, h = map(int, tracker.box)
                font_size = max(8, int(h * 0.13))
                font = QtGui.QFont("Arial", font_size, QtGui.QFont.Bold)
                painter.setFont(font)

                if tracker.confirmed:
                    painter.setPen(QtGui.QPen(QtCore.Qt.green, 3))
                    if tracker.frames_since_seen > 0:
                        painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 3))
                    painter.drawRect(x, y, w, h)
                    painter.setPen(QtCore.Qt.white)
                    label_name = QtCore.QRectF(x, y - font_size - 10, w, font_size + 5)
                    label_text = tracker.name or f"Face_{tracker.id}"
                    painter.drawText(label_name, QtCore.Qt.AlignLeft, label_text)
                    confidence_text = f"{tracker.sim_score:.4f}"
                    label_conf = QtCore.QRectF(x, y + h + 3, w, font_size + 5)
                    painter.drawText(label_conf, QtCore.Qt.AlignLeft, confidence_text)

                else:
                    pen = QtGui.QPen(QtCore.Qt.yellow, 2)
                    pen.setStyle(QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawRect(x, y, w, h)
                    painter.setPen(QtCore.Qt.white)
                    confidence_text = f"{tracker.detection_confidence():.2f}"
                    label_conf = QtCore.QRectF(x, y + h + 3, w, font_size + 5)
                    painter.drawText(label_conf, QtCore.Qt.AlignLeft, confidence_text)
