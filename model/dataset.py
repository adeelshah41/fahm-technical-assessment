"""
FAHM Mammogram Classification — Dataset
Handles CBIS-DDSM path mapping, label encoding, and PyTorch Dataset.

Path mapping logic:
  1. Load dicom_info.csv which maps SeriesInstanceUID → jpeg image_path.
  2. For each case CSV row, extract the SeriesInstanceUID (uid2) from
     'cropped image file path' (format: casename/uid1/uid2/file.dcm).
  3. Look up uid2 in the dicom_info mapping to find the actual jpeg path.
  4. Combine with data_root to get the full filesystem path.
"""

import os
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# Path-mapping helpers
# ────────────────────────────────────────────────────────


def _build_dicom_lookup(csv_root: str) -> dict:
    """Build a lookup from SeriesInstanceUID → relative jpeg path.

    dicom_info.csv has:
      - SeriesInstanceUID: the UID that appears as the folder name in jpeg/
      - image_path: e.g. 'CBIS-DDSM/jpeg/<uid>/<filename>.jpg'
      - SeriesDescription: 'cropped images', 'ROI mask images', or
        'full mammogram images'

    We only want "cropped images" rows.  For each SeriesInstanceUID we store
    *all* matching jpeg filenames so that we can pick the first one (typically
    only one cropped-image JPEG exists per series).

    Returns:
        dict  {uid_str: relative_jpeg_path}   e.g.
              {'1.3.6...304': '1.3.6...304/1-172.jpg'}
    """
    dicom_info_path = os.path.join(csv_root, "dicom_info.csv")
    df = pd.read_csv(dicom_info_path)

    # Filter to cropped images only (16-bit, not ROI masks)
    cropped = df[df["SeriesDescription"].str.strip() == "cropped images"].copy()

    lookup: dict[str, str] = {}
    for _, row in cropped.iterrows():
        uid = str(row["SeriesInstanceUID"]).strip()
        img_path = str(row["image_path"]).strip()  # e.g. CBIS-DDSM/jpeg/uid/file.jpg
        # Extract relative path under jpeg/  →  uid/file.jpg
        parts = img_path.replace("\\", "/").split("/")
        # Expected: ['CBIS-DDSM', 'jpeg', '<uid>', '<file>.jpg']
        if len(parts) >= 3:
            rel = "/".join(parts[2:])  # uid/file.jpg
        else:
            rel = img_path
        if uid not in lookup:
            lookup[uid] = rel

    logger.info("DICOM lookup built: %d cropped-image series found.", len(lookup))
    return lookup


def _extract_series_uid(cropped_path: str) -> str | None:
    """Extract the SeriesInstanceUID (uid2) from a cropped image file path.

    The CSV gives paths like:
      Mass-Training_P_00001_LEFT_CC_1/uid1/uid2/000000.dcm
    uid2 is the 3rd segment (0-indexed: [casename, uid1, uid2, file]).
    """
    # Clean up quoting / newlines that appear in calc CSVs
    clean = cropped_path.strip().strip('"').strip()
    segments = clean.replace("\\", "/").split("/")
    if len(segments) >= 3:
        return segments[2]
    return None


# ────────────────────────────────────────────────────────
# DataFrame preparation
# ────────────────────────────────────────────────────────

PATHOLOGY_MAP = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

CSV_FILES = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
]


def prepare_dataframe(data_root: str, csv_root: str) -> pd.DataFrame:
    """Combine all four case CSVs, map DICOM paths → JPEG, encode labels.

    Returns a DataFrame with columns: ['image_path', 'label']
    where image_path is an absolute (or data_root-relative) path to a JPEG.
    """
    lookup = _build_dicom_lookup(csv_root)

    records: list[dict] = []
    missing = 0

    for csv_name in CSV_FILES:
        csv_path = os.path.join(csv_root, csv_name)
        if not os.path.isfile(csv_path):
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue

        df = pd.read_csv(csv_path)

        # The column name for cropped path
        crop_col = "cropped image file path"
        if crop_col not in df.columns:
            logger.warning("Column '%s' missing in %s, skipping.", crop_col, csv_name)
            continue

        for _, row in df.iterrows():
            raw_crop = str(row[crop_col])
            pathology = str(row["pathology"]).strip()

            if pathology not in PATHOLOGY_MAP:
                continue

            uid2 = _extract_series_uid(raw_crop)
            if uid2 is None:
                missing += 1
                continue

            # Look up jpeg path from dicom_info
            rel_jpeg = lookup.get(uid2)
            if rel_jpeg is None:
                missing += 1
                continue

            full_path = os.path.join(data_root, rel_jpeg)

            if not os.path.isfile(full_path):
                # Try to find any jpg in the uid2 folder as fallback
                uid_folder = os.path.join(data_root, uid2)
                if os.path.isdir(uid_folder):
                    jpgs = [f for f in os.listdir(uid_folder) if f.lower().endswith((".jpg", ".jpeg"))]
                    if jpgs:
                        full_path = os.path.join(uid_folder, jpgs[0])
                    else:
                        missing += 1
                        continue
                else:
                    missing += 1
                    continue

            records.append(
                {
                    "image_path": full_path,
                    "label": PATHOLOGY_MAP[pathology],
                }
            )

    if missing > 0:
        logger.warning(
            "%d rows skipped (missing UID mapping or image file).", missing
        )

    result = pd.DataFrame(records)

    # De-duplicate (same image may appear in mass & calc overlaps)
    result = result.drop_duplicates(subset="image_path").reset_index(drop=True)

    logger.info(
        "Dataset prepared: %d images  |  Malignant=%d  Benign=%d",
        len(result),
        int((result["label"] == 1).sum()),
        int((result["label"] == 0).sum()),
    )
    return result


# ────────────────────────────────────────────────────────
# PyTorch Dataset
# ────────────────────────────────────────────────────────


class MammogramDataset(Dataset):
    """PyTorch Dataset for CBIS-DDSM cropped mammogram ROIs.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain 'image_path' and 'label' columns.
    transform : callable, optional
        torchvision-style transforms to apply to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"])

        # Apply CLAHE to enhance contrast of mammogram ROIs
        import cv2
        import numpy as np
        img_np = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        image = Image.fromarray(img_clahe).convert("RGB")

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

