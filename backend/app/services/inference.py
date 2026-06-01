import torch
import torch.nn as nn
from torchvision import models
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.pt")
THRESHOLD = 0.5

_model = None

def load_model():
    global _model
    if _model is None:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        _model = model
    return _model

def run_inference(tensor: torch.Tensor) -> tuple[str, float]:
    model = load_model()
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()
    label = "Malignant" if prob >= THRESHOLD else "Benign/Normal"
    return label, prob
