import io
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import setup_model_and_loss

_CHECKPOINT_PATH = "best_model.pth"
_CLASS_WEIGHTS_PATH = "class_weights.json"


class ModelService:
    def __init__(self, device: str = "cpu"):
        self.device = device

        # Load ordered class names from class_weights.json (same sort order as training)
        with open(_CLASS_WEIGHTS_PATH, "r") as f:
            weights = json.load(f)
        self.class_names = sorted(weights.keys())

        # Initialize model architecture and move to device
        self.model, _ = setup_model_and_loss(device)

        # Load best checkpoint (dict-style format from Phase 2b update)
        checkpoint = torch.load(_CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback for legacy plain state_dict
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Inference transforms — identical to validation / evaluate.py
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on raw image bytes and return the top prediction.

        Args:
            image_bytes: Raw bytes of a JPEG or PNG image.

        Returns:
            {"prediction": class_name, "confidence": float}
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probabilities = F.softmax(logits, dim=1).squeeze(0)
        confidence, class_idx = torch.max(probabilities, dim=0)

        return {
            "prediction": self.class_names[class_idx.item()],
            "confidence": round(confidence.item(), 4),
        }
