import os
from typing import Optional

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from PocketNetCode.backbones.augment_cnn import AugmentCNN
import PocketNetCode.backbones.genotypes as gt
from PocketNetCode.config import config_PocketNetS128

from config.paths import MODEL_PATH

DEBUG = False


def DEBUG_log(message):
    if DEBUG:
        print(f"[DEBUG] [pocketnet_face.py] {message}")


class FaceEngine:
    """Utility class for PocketNet based face recognition."""

    def __init__(
            self, device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        DEBUG_log(f"Using device: {self.device}")

        # Initialize MTCNN. It will use the GPU for detection if self.device is cuda.
        self.detector = MTCNN(keep_all=False, min_face_size=40,
                              thresholds=[0.6, 0.7, 0.7], device=self.device)
        DEBUG_log("MTCNN initialized")

        cfg = config_PocketNetS128.config

        genotype = gt.from_str(cfg.genotypes["softmax_casia"])
        # Load the PocketNet model onto the specified device
        self.model = AugmentCNN(
            C=cfg.channel,
            n_layers=cfg.n_layers,
            genotype=genotype,
            stem_multiplier=4,
            emb=cfg.embedding_size,
        ).to(self.device)

        # Check if the model file exists before loading
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"PocketNet model not found at {os.path.abspath(MODEL_PATH)}")

        state = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        DEBUG_log(f"PocketNet model loaded from {MODEL_PATH}")

    @staticmethod
    def _preprocess(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - 127.5) * 0.0078125

    def embed_tensor(self, face_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Generates an embedding from a face tensor (output of MTCNN).
        """
        if face_tensor is None:
            return None

        # MTCNN might return the tensor on CPU even if initialized with CUDA.
        # We MUST ensure the input tensor is on the same device as the model (self.device).
        if face_tensor.device != self.device:
            face_tensor = face_tensor.to(self.device)

        if face_tensor.dim() == 3:  # Safety check for unbatched input
            face_tensor = face_tensor.unsqueeze(0)

        # Ensure we have a 4D tensor before preprocessing
        if face_tensor.dim() != 4:
            print(f"[ERROR] embed_tensor received an invalid tensor shape: {face_tensor.shape}")
            return None

        face = self._preprocess(face_tensor)
        with torch.no_grad():
            emb = self.model(face)

        template = emb[0].cpu().numpy()
        DEBUG_log(f"Template extracted from tensor with shape {template.shape}")
        return template

    def extract_template(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extracts a template from a given image file path.
        Used for registering users.
        """
        DEBUG_log(f"Reading image: {image_path}")
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            return None

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not open or read image file: {image_path}. Error: {e}")
            return None

        # Use the MTCNN detector to get the face tensor
        face_tensor = self.detector(img)

        if face_tensor is None:
            print(f"[ERROR] No face detected in {image_path}")
            return None

        # Generate the embedding from the detected face tensor
        return self.embed_tensor(face_tensor)