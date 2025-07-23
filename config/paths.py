# config/paths.py
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "face_recognition" / "models"
MODEL_PATH = MODELS_DIR / "pocketNetS_128_pretrained.pth"

IMAGE_DIR = PROJECT_ROOT / "res" / "images_to_register"

DB_DIR = PROJECT_ROOT / "src" / "db"
DB_PATH = DB_DIR / "users.db"
