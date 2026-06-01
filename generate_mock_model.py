"""
Generate a mock model.pt — a randomly initialized EfficientNet-B0
with the same architecture as the real model so the backend can load
and run inference without crashing.

Usage:
    python generate_mock_model.py

Output:
    backend/app/models/model.pt
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def generate_mock_model():
    # Build the same architecture used in training and inference
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    # Save the state dict (randomly initialized weights)
    save_path = os.path.join(
        os.path.dirname(__file__), "backend", "app", "models", "model.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ Mock model saved to: {save_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print(f"   Architecture: EfficientNet-B0 with classifier head → Linear(1280, 1)")
    print(f"   Weights: randomly initialized (NOT trained)")
    print(f"   Replace with trained weights before production use.")

if __name__ == "__main__":
    generate_mock_model()
