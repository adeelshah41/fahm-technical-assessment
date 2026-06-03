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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
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
        nn.Dropout(p=0.4, inplace=True),
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

    auc_roc = roc_auc_score(all_labels_np, all_probs_np)

    # ── 5. Default Threshold (0.5) — Baseline ──
    default_preds = (all_probs_np >= 0.5).astype(int)
    default_acc = accuracy_score(all_labels_np, default_preds)
    default_prec = precision_score(all_labels_np, default_preds, pos_label=1, zero_division=0)
    default_recall = recall_score(all_labels_np, default_preds, pos_label=1, zero_division=0)
    default_f1 = f1_score(all_labels_np, default_preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels_np, default_preds, labels=[0, 1]).ravel()
    default_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 70)
    print("       BASELINE RESULTS (Default Threshold = 0.50)")
    print("=" * 70)
    print(f"  Accuracy:    {default_acc:.4f}")
    print(f"  Precision:   {default_prec:.4f}  (PPV)")
    print(f"  Recall:      {default_recall:.4f}  (Sensitivity / TPR)")
    print(f"  Specificity: {default_spec:.4f}  (TNR)")
    print(f"  F1-Score:    {default_f1:.4f}")
    print(f"  AUC-ROC:     {auc_roc:.4f}")
    print("-" * 70)
    print(f"  Confusion Matrix:  TN={tn}  FP={fp}  |  FN={fn}  TP={tp}")
    print("=" * 70)

    # ── 6. Threshold Sweep ──
    print("\n" + "=" * 70)
    print("       THRESHOLD SWEEP ANALYSIS")
    print("=" * 70)
    print(f"  {'Thresh':>7} | {'Sens':>7} | {'Spec':>7} | {'Acc':>7} | {'Prec':>7} | {'F1':>7} | {'Youden J':>8} | {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}")
    print("-" * 95)

    best_youden_j = -1
    best_youden_thresh = 0.5
    best_sens_thresh = None

    sweep_thresholds = np.arange(0.25, 0.56, 0.01)
    sweep_results = []

    for thresh in sweep_thresholds:
        preds = (all_probs_np >= thresh).astype(int)
        acc = accuracy_score(all_labels_np, preds)
        prec = precision_score(all_labels_np, preds, pos_label=1, zero_division=0)
        sens = recall_score(all_labels_np, preds, pos_label=1, zero_division=0)
        f1_val = f1_score(all_labels_np, preds, pos_label=1, zero_division=0)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(all_labels_np, preds, labels=[0, 1]).ravel()
        spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        youden_j = sens + spec - 1

        marker = ""
        if thresh == 0.50:
            marker = " <-- DEFAULT"

        print(f"  {thresh:>7.2f} | {sens:>6.4f} | {spec:>6.4f} | {acc:>6.4f} | {prec:>6.4f} | {f1_val:>6.4f} | {youden_j:>8.4f} | {tn_t:>4} {fp_t:>4} {fn_t:>4} {tp_t:>4}{marker}")

        sweep_results.append({
            'thresh': thresh, 'sens': sens, 'spec': spec, 'acc': acc,
            'prec': prec, 'f1': f1_val, 'youden_j': youden_j,
            'tn': tn_t, 'fp': fp_t, 'fn': fn_t, 'tp': tp_t
        })

        if youden_j > best_youden_j:
            best_youden_j = youden_j
            best_youden_thresh = thresh

        if sens >= 0.85 and (best_sens_thresh is None or thresh > best_sens_thresh):
            best_sens_thresh = thresh

    print("-" * 95)

    # ── 7. Summary ──
    print("\n" + "=" * 70)
    print("       OPTIMAL THRESHOLD RECOMMENDATIONS")
    print("=" * 70)

    def print_at_threshold(label, thresh):
        preds = (all_probs_np >= thresh).astype(int)
        acc = accuracy_score(all_labels_np, preds)
        prec = precision_score(all_labels_np, preds, pos_label=1, zero_division=0)
        sens = recall_score(all_labels_np, preds, pos_label=1, zero_division=0)
        f1_val = f1_score(all_labels_np, preds, pos_label=1, zero_division=0)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(all_labels_np, preds, labels=[0, 1]).ravel()
        spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        youden = sens + spec - 1
        print(f"\n  [{label}]  Threshold = {thresh:.2f}")
        print(f"    Sensitivity: {sens:.4f}   Specificity: {spec:.4f}")
        print(f"    Accuracy:    {acc:.4f}   Precision:   {prec:.4f}")
        print(f"    F1-Score:    {f1_val:.4f}   Youden J:    {youden:.4f}")
        print(f"    Confusion:   TN={tn_t}  FP={fp_t}  |  FN={fn_t}  TP={tp_t}")
        return sens, spec

    print_at_threshold("DEFAULT", 0.50)
    youden_sens, youden_spec = print_at_threshold("YOUDEN OPTIMAL (best balanced)", best_youden_thresh)

    if best_sens_thresh is not None:
        clin_sens, clin_spec = print_at_threshold("CLINICAL (sensitivity >= 85%)", best_sens_thresh)
    else:
        closest_thresh = min(sweep_results, key=lambda x: abs(x['sens'] - 0.85))['thresh']
        clin_sens, clin_spec = print_at_threshold("NEAREST TO 85% SENSITIVITY", closest_thresh)
        best_sens_thresh = closest_thresh

    print("\n" + "=" * 70)
    print(f"  >>> RECOMMENDED THRESHOLD FOR DEPLOYMENT: {best_youden_thresh:.2f}")
    print(f"  >>> (Youden-optimal balances sensitivity and specificity)")
    print("=" * 70)

    # ── 8. Plots ──
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs_dir, exist_ok=True)

    fpr, tpr, roc_thresholds = roc_curve(all_labels_np, all_probs_np)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve with operating points
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_roc:.4f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", alpha=0.5)
    ax1.plot(1 - default_spec, default_recall, 'rs', markersize=12, label=f"Default (t=0.50)", zorder=5)
    ax1.plot(1 - youden_spec, youden_sens, 'g^', markersize=12, label=f"Youden (t={best_youden_thresh:.2f})", zorder=5)
    ax1.plot(1 - clin_spec, clin_sens, 'bD', markersize=10, label=f"Clinical (t={best_sens_thresh:.2f})", zorder=5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax1.set_title("ROC Curve with Operating Points", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # Threshold vs Metrics
    ax2 = axes[1]
    threshs = [r['thresh'] for r in sweep_results]
    sens_vals = [r['sens'] for r in sweep_results]
    spec_vals = [r['spec'] for r in sweep_results]
    acc_vals = [r['acc'] for r in sweep_results]
    f1_vals = [r['f1'] for r in sweep_results]

    ax2.plot(threshs, sens_vals, 'r-o', markersize=3, lw=2, label="Sensitivity")
    ax2.plot(threshs, spec_vals, 'b-s', markersize=3, lw=2, label="Specificity")
    ax2.plot(threshs, acc_vals, 'g-^', markersize=3, lw=1.5, label="Accuracy")
    ax2.plot(threshs, f1_vals, 'm-d', markersize=3, lw=1.5, label="F1-Score")
    ax2.axvline(x=0.50, color='red', linestyle=':', alpha=0.5, label="Default (0.50)")
    ax2.axvline(x=best_youden_thresh, color='green', linestyle=':', alpha=0.5, label=f"Youden ({best_youden_thresh:.2f})")
    ax2.axhline(y=0.85, color='gray', linestyle='--', alpha=0.4, label="85% target")
    ax2.set_xlim([0.25, 0.55])
    ax2.set_ylim([0.4, 1.0])
    ax2.set_xlabel("Classification Threshold", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.set_title("Metrics vs. Classification Threshold", fontsize=14)
    ax2.legend(loc="center left", fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    roc_path = os.path.join(docs_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nROC + Threshold analysis saved to: {roc_path}")

    # Classification report at recommended threshold
    print("\n" + "=" * 70)
    print(f"  Classification Report at Recommended Threshold ({best_youden_thresh:.2f})")
    print("=" * 70)
    rec_preds = (all_probs_np >= best_youden_thresh).astype(int)
    print(classification_report(all_labels_np, rec_preds, target_names=["Benign", "Malignant"]))


if __name__ == "__main__":
    main()
