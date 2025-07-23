import sys
import os
import pygetwindow as gw
from PyQt5 import QtWidgets

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
pocketnet_path = os.path.join(project_root, 'PocketNetCode')

# Add the 'PocketNetCode' directory to the Python path
if pocketnet_path not in sys.path:
    sys.path.insert(0, pocketnet_path)

# Import the main overlay class which now handles all logic
from src.face_detection.MTCNN_FD import Overlay, DEBUG_log

if __name__ == "__main__":
    DEBUG_log("Starting application...")

    # --- 1. Find the target window to overlay ---
    # You can change this to any part of a window title
    target_window_title = "vr view" # IMPORTANT always use lowcase

    wins = gw.getAllWindows()
    matches = [w for w in wins if target_window_title in w.title.lower() and w.visible and not w.isMinimized]

    if not matches:
        print(f"Error: Window containing '{target_window_title}' not found or not visible.")
        print("Please ensure the target window is open and not minimized.")
        # List available windows to help user
        print("\nAvailable windows:")
        for w in wins:
            if w.title: print(f"- {w.title}")
        sys.exit(1)

    target_win = matches[0]
    DEBUG_log(f"Targeting window: '{target_win.title}'")

    # --- 2. Initialize and run the PyQt Application ---
    app = QtWidgets.QApplication(sys.argv)

    # The Overlay class internally initializes the face engine and tracker
    overlay = Overlay(target_win)
    overlay.show()

    sys.exit(app.exec_())
