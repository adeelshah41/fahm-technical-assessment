"""
FAHM Mammogram Classification — Training Script
EfficientNet-B0 binary classifier (Malignant vs Benign/Normal).

Usage:
    cd d:/fahm/model
    python train.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from dataset import MammogramDataset, prepare_dataframe


# ────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────
# Transforms
# ────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    """EfficientNet-B0 with a single-output (sigmoid) classifier head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model.to(device)


# ────────────────────────────────────────────────────────
# Training helpers
# ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        labels_t = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels_t)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())

    avg_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return avg_loss, auc


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Prepare data ──
    full_df = prepare_dataframe(config.DATA_ROOT, config.CSV_ROOT)

    if len(full_df) == 0:
        print("ERROR: No images found. Check DATA_ROOT and CSV_ROOT in config.py.")
        sys.exit(1)

    # Stratified train / (val+test) split
    train_df, temp_df = train_test_split(
        full_df,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=full_df["label"],
        random_state=config.SEED,
    )
    # Split temp into val / test  (50-50 of the remaining 30%)
    relative_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=config.SEED,
    )

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # ── 2. Datasets & loaders ──
    train_ds = MammogramDataset(train_df, transform=train_transforms)
    val_ds = MammogramDataset(val_df, transform=val_transforms)
    test_ds = MammogramDataset(test_df, transform=val_transforms)

    # WeightedRandomSampler to handle class imbalance in training
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # ── 3. Model, loss, optimizer ──
    model = build_model(device)

    # pos_weight compensates for class imbalance in the loss as well
    num_neg = int((train_labels == 0).sum())
    num_pos = int((train_labels == 1).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight = {pos_weight.item():.3f}  (neg={num_neg}, pos={num_pos})")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ── 4. Training loop ──
    best_auc = 0.0
    save_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved (AUC={best_auc:.4f})")

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")

    # ── 5. Quick test-set evaluation ──
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  |  Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
