"""
FAHM Mammogram Classification — Evaluation Script
Loads trained model, runs on the test split, and produces metrics + plots.

Usage:
    cd d:/fahm/model
    python evaluate.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import config
from dataset import MammogramDataset, prepare_dataframe


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    """Reconstruct the EfficientNet-B0 architecture (no pretrained weights)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model.to(device)


eval_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Check model exists ──
    if not os.path.isfile(config.MODEL_SAVE_PATH):
        print(f"ERROR: Trained model not found at {config.MODEL_SAVE_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    # ── 2. Rebuild test split (same seed → same split) ──
    full_df = prepare_dataframe(config.DATA_ROOT, config.CSV_ROOT)

    train_df, temp_df = train_test_split(
        full_df,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=full_df["label"],
        random_state=config.SEED,
    )
    relative_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    _, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=config.SEED,
    )

    print(f"Test set size: {len(test_df)}")

    test_ds = MammogramDataset(test_df, transform=eval_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # ── 3. Load model ──
    model = build_model(device)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # ── 4. Inference ──
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    all_preds = (all_probs_np >= 0.5).astype(int)

    # ── 5. Metrics ──
    accuracy = accuracy_score(all_labels_np, all_preds)
    sensitivity = recall_score(all_labels_np, all_preds, pos_label=1, zero_division=0)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc_roc = roc_auc_score(all_labels_np, all_probs_np)

    print("\n" + "=" * 50)
    print("       EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (Recall / TPR)")
    print(f"  Specificity: {specificity:.4f}  (TNR)")
    print(f"  AUC-ROC:     {auc_roc:.4f}")
    print("-" * 50)
    print("  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print("=" * 50)

    # ── 6. ROC Curve ──
    fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    roc_path = os.path.join(docs_dir, "roc_curve.png")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FAHM — ROC Curve (Mammogram Classification)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\nROC curve saved to: {roc_path}")


if __name__ == "__main__":
    main()
