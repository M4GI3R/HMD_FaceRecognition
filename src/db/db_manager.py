# db_manager.py

import os
import sqlite3
import pickle
from typing import List, Optional, Dict, Any
import numpy as np

from src.face_recognition.pocketnet_face import FaceEngine
from config.paths import IMAGE_DIR, DB_PATH


class DBManager:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        # ensure directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

        # Check for existing users
        if not self._has_users():
            print("[WARNING] Database is initialized but contains no registered users.")

    def _has_users(self) -> bool:
        """
        Returns True if there is at least one user in the database.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        return count > 0

    def _create_table(self):
        """Create the users table if it doesn't already exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            color TEXT NOT NULL,
            templates BLOB,
            registered_user BOOLEAN NOT NULL,
            last_seen TEXT
        );
        """
        self.conn.execute(sql)
        self.conn.commit()

    def register_user(
            self,
            name: str,
            color: str,
            templates: List[bytes],
            registered_user: bool = True,
            last_seen: Optional[str] = None
    ) -> int:
        """
        Insert a new user record.
        :param name:            Person's name.
        :param color:           Hex color string, e.g. "#008ECC".
        :param templates:       List of face‐template arrays (NumPy ndarrays).
        :param registered_user: Flag indicating a registered user.
        :param last_seen:       ISO timestamp string, or None.
        :return:                The new user's auto‐assigned ID.
        """
        blob = pickle.dumps(templates)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO users (name, color, templates, registered_user, last_seen) VALUES (?, ?, ?, ?, ?)",
            (name, color, blob, int(registered_user), last_seen),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_user_templates(self, user_id: int) -> List:
        """
        Retrieve and unpickle the templates for a given user.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT templates FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row or row[0] is None:
            return []
        return pickle.loads(row[0])

    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all users in the DB, each as a dict:
          {
            'id': int,
            'name': str,
            'color': str,
            'templates': List[np.ndarray],
            'registered_user': bool,
            'last_seen': Optional[str]
          }
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, color, templates, registered_user, last_seen FROM users")
        rows = cur.fetchall()

        users = []
        for uid, name, color, blob, reg_flag, last_seen in rows:
            templates = pickle.loads(blob) if blob is not None else []
            users.append({
                'id': uid,
                'name': name,
                'color': color,
                'templates': templates,
                'registered_user': bool(reg_flag),
                'last_seen': last_seen,
            })
        return users

    def update_last_seen(self, user_id: int, timestamp: str):
        """
        Update the last_seen column for a user (use ISO‐format datetime).
        """
        self.conn.execute(
            "UPDATE users SET last_seen = ? WHERE id = ?",
            (timestamp, user_id),
        )
        self.conn.commit()

    def reset_database(self):
        """
        Delete the current DB file (if exists), reinitialize the connection and recreate the users table.
        """
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f"[INFO] Deleted database file: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        print("[INFO] Reinitialized database and recreated tables.")



def register_user(db: DBManager, engine: FaceEngine, name: str, image_filenames: list[str], color: str):
    templates = []

    # Extract templates and check for validity
    for image_filename in image_filenames:
        img_path = IMAGE_DIR / image_filename
        template = engine.extract_template(img_path)
        if template is None:
            print(f"[ERROR] Could not extract template from {img_path}")
            continue

        # Duplicate check against existing users
        duplicate_found = False
        for user in db.get_all_users():
            for existing_template in user['templates']:
                if np.allclose(template, existing_template):
                    print(
                        f"[SKIP] Image {image_filename} is already used for user "
                        f"'{user['name']}' (ID {user['id']})."
                    )
                    duplicate_found = True
                    break
            if duplicate_found:
                break

        if not duplicate_found:
            templates.append(template)

    if not templates:
        print("[ABORT] No unique templates found. User not registered.")
        return

    # Register new user with all unique templates
    user_id = db.register_user(
        name=name,
        color=color,
        templates=templates,
        registered_user=True,
        last_seen=None
    )
    print(f"[OK] Registered user {name} with ID {user_id} using {len(templates)} image(s)")


def main():
    # Initialize DB + face engine once
    db = DBManager(DB_PATH)
    engine = FaceEngine()

    # Reset/clear DB before registering users
    db.reset_database()

    # Register new user
    # register_user(db, engine, "NAME", images: ["IMAGE_1.png", "IMAGE_2.jpg", ..."], color: "#3881ff")
    register_user(db, engine, "Simon", ["Simon_2.png", "Simon_3.jpg", "Simon_1.jpg"], "#3881ff")


if __name__ == "__main__":
    main()
