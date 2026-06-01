# 🤖 AI Agent Master Context — FAHM Mammogram Classification Project

> **Purpose:** This document gives a fully self-contained context brief for an AI coding agent to plan, scaffold, implement, and deploy the FAHM Biotechnology technical assessment end-to-end.
> **Deadline:** Sunday, 7 June 2026 — 11:59 PM Arabia Standard Time (GMT+3)
> **Estimated Build Time:** 16–18 focused hours across 2 days

---

## 1. PROJECT OBJECTIVE

Build and deploy a **full-stack AI web application** that:
- Accepts a mammogram image (DICOM, PNG, or JPEG) uploaded by a clinician
- Runs binary classification inference: **Malignant vs. Benign/Normal**
- Returns the predicted label + a **confidence score** (0–100%)
- Is deployed live on a cloud platform (GCP preferred)

---

## 2. TECHNOLOGY STACK DECISIONS

| Layer | Choice | Reason |
|---|---|---|
| ML Framework | PyTorch + torchvision | Flexible, great pretrained model support |
| Pretrained Model | EfficientNet-B0 | Lightweight, strong performance on medical imaging |
| Backend | FastAPI (Python 3.11) | Async, fast, OpenAPI docs auto-generated |
| Frontend | React.js + TailwindCSS | Clean UI, easy to deploy |
| Containerization | Docker | Reproducibility, cloud-ready |
| Cloud Platform | GCP Cloud Run | Serverless, HTTPS by default, free tier |
| Frontend Hosting | Vercel | Free, instant deploy from GitHub |
| Dataset | CBIS-DDSM (Kaggle PNG version) | Standardized, large, publicly available |

---

## 3. REPOSITORY STRUCTURE

Scaffold the project exactly as follows:

```
fahm-mammogram/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routers/
│   │   │   └── predict.py       # /predict endpoint
│   │   ├── services/
│   │   │   ├── preprocessor.py  # Image preprocessing pipeline
│   │   │   └── inference.py     # Model loading + inference logic
│   │   ├── utils/
│   │   │   └── dicom_utils.py   # DICOM → PNG conversion
│   │   └── models/
│   │       └── .gitkeep         # model.pt goes here (not committed)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── model/
│   ├── train.py                 # Full training script
│   ├── evaluate.py              # Metrics + ROC curve generation
│   ├── dataset.py               # PyTorch Dataset class for CBIS-DDSM
│   ├── config.py                # All hyperparameters in one place
│   └── requirements_train.txt   # Training-only deps (heavier)
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadZone.jsx   # Drag-and-drop image upload
│   │   │   ├── ResultCard.jsx   # Classification result display
│   │   │   └── Disclaimer.jsx   # Medical disclaimer banner
│   │   ├── services/
│   │   │   └── api.js           # axios calls to backend
│   │   └── index.css
│   ├── package.json
│   └── .env.example             # VITE_API_URL=...
│
├── docs/
│   └── report.md                # Technical report draft
│
├── docker-compose.yml           # Local full-stack run
├── .github/
│   └── workflows/
│       └── ci.yml               # Optional: lint + test on push
└── README.md
```

---

## 4. PHASE-BY-PHASE IMPLEMENTATION INSTRUCTIONS

---

### PHASE 1 — Dataset & Training Pipeline (`/model`)

#### 4.1 Dataset

- **Dataset:** CBIS-DDSM — use the **preprocessed PNG version from Kaggle**
  - URL: `https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset`
  - Labels: Mass cases → Malignant / Benign; Normal cases → Benign/Normal
  - Binary target: `1 = Malignant`, `0 = Benign/Normal`

#### 4.2 `model/config.py`

```python
# config.py — all hyperparameters in one place
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "../backend/app/models/model.pt"
DATA_ROOT = "./data/"
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
```

#### 4.3 `model/dataset.py`

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])  # 1=Malignant, 0=Benign
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
```

#### 4.4 `model/train.py` — Key Logic

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import MammogramDataset
from config import *

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Model ---
def build_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

# --- Class Imbalance: Weighted Loss ---
# Count class frequencies in training set and pass to BCEWithLogitsLoss
# pos_weight = num_negatives / num_positives

# --- Training Loop ---
# Use BCEWithLogitsLoss with pos_weight
# Use Adam optimizer
# Save best model based on validation AUC
# Log train loss, val loss, val AUC per epoch
```

#### 4.5 `model/evaluate.py` — Metrics to Generate

Generate and save all of the following:

```python
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# MUST compute and report:
# 1. Accuracy
# 2. Sensitivity = Recall = TP / (TP + FN)   ← CRITICAL for screening
# 3. Specificity = TN / (TN + FP)
# 4. AUC-ROC score
# 5. ROC curve plot → save as docs/roc_curve.png

# Threshold: default 0.5 — also consider tuning for higher sensitivity

def plot_roc_curve(y_true, y_probs, save_path="docs/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="darkorange", lw=2)
    plt.plot([0,1],[0,1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve — Mammogram Binary Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
```

---

### PHASE 2 — Backend API (`/backend`)

#### 4.6 `backend/requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
torch==2.3.0
torchvision==0.18.0
Pillow==10.3.0
pydicom==2.4.4
numpy==1.26.4
scikit-learn==1.5.0
python-dotenv==1.0.1
```

#### 4.7 `backend/app/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict

app = FastAPI(
    title="FAHM Mammogram Classifier API",
    description="Binary classification: Malignant vs Benign/Normal",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod: restrict to your frontend domain
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

#### 4.8 `backend/app/routers/predict.py`

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.preprocessor import preprocess_image
from app.services.inference import run_inference

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "application/dicom"]
    # Also allow .dcm by extension check
    filename = file.filename.lower()
    if not (filename.endswith(('.jpg','.jpeg','.png','.dcm'))):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()

    # PRIVACY: image is processed in memory — never written to disk
    tensor = preprocess_image(contents, filename)
    label, confidence = run_inference(tensor)

    return {
        "prediction": label,           # "Malignant" or "Benign/Normal"
        "confidence": round(confidence * 100, 2),  # percentage
        "disclaimer": "For research purposes only. Not a clinical diagnostic tool."
    }
```

#### 4.9 `backend/app/services/preprocessor.py`

```python
from PIL import Image
import torchvision.transforms as T
import torch
import io
from app.utils.dicom_utils import dicom_to_pil

IMG_SIZE = 224

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_image(file_bytes: bytes, filename: str) -> torch.Tensor:
    if filename.endswith(".dcm"):
        pil_img = dicom_to_pil(file_bytes)
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    tensor = transform(pil_img).unsqueeze(0)  # add batch dim
    return tensor
```

#### 4.10 `backend/app/utils/dicom_utils.py`

```python
import pydicom
import numpy as np
from PIL import Image
import io

def dicom_to_pil(file_bytes: bytes) -> Image.Image:
    """Convert DICOM bytes to PIL Image. Strips all patient metadata."""
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    
    # ANONYMIZATION: Only access pixel data — never read/log patient metadata
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Normalize to 0-255
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(pixel_array).convert("RGB")
    return pil_img
```

#### 4.11 `backend/app/services/inference.py`

```python
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
```

#### 4.12 `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### PHASE 3 — Frontend UI (`/frontend`)

#### 4.13 React App Overview

Use **Vite + React + TailwindCSS**. Bootstrap:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer
npm install axios
npx tailwindcss init -p
```

#### 4.14 `frontend/src/App.jsx` — Structure

```jsx
import { useState } from "react"
import UploadZone from "./components/UploadZone"
import ResultCard from "./components/ResultCard"
import Disclaimer from "./components/Disclaimer"

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-800">FAHM Mammogram Classifier</h1>
        <p className="text-gray-500 mt-2">Upload a mammogram to receive an AI classification</p>
      </header>

      <Disclaimer />

      <UploadZone
        setResult={setResult}
        setLoading={setLoading}
        setPreview={setPreview}
        setError={setError}
      />

      {preview && (
        <img src={preview} alt="Uploaded mammogram preview"
          className="mt-6 max-w-xs rounded-lg shadow-md border border-gray-200" />
      )}

      {loading && (
        <div className="mt-6 text-blue-600 font-medium animate-pulse">
          Analyzing image...
        </div>
      )}

      {error && (
        <div className="mt-4 text-red-500 text-sm">{error}</div>
      )}

      {result && !loading && <ResultCard result={result} />}
    </div>
  )
}
```

#### 4.15 `frontend/src/components/ResultCard.jsx`

```jsx
export default function ResultCard({ result }) {
  const isMalignant = result.prediction === "Malignant"

  return (
    <div className={`mt-8 p-6 rounded-2xl shadow-lg w-full max-w-sm border-2
      ${isMalignant ? "border-red-400 bg-red-50" : "border-green-400 bg-green-50"}`}>

      <h2 className="text-xl font-bold text-gray-800 mb-1">Classification Result</h2>

      <p className={`text-3xl font-extrabold mt-2
        ${isMalignant ? "text-red-600" : "text-green-600"}`}>
        {result.prediction}
      </p>

      <div className="mt-4">
        <p className="text-sm text-gray-500 mb-1">Confidence</p>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`h-3 rounded-full transition-all duration-700
              ${isMalignant ? "bg-red-500" : "bg-green-500"}`}
            style={{ width: `${result.confidence}%` }}
          />
        </div>
        <p className="text-right text-sm font-semibold text-gray-700 mt-1">
          {result.confidence}%
        </p>
      </div>

      <p className="text-xs text-gray-400 mt-4 italic">{result.disclaimer}</p>
    </div>
  )
}
```

#### 4.16 `frontend/src/components/UploadZone.jsx`

```jsx
import { useRef } from "react"
import { predictImage } from "../services/api"

export default function UploadZone({ setResult, setLoading, setPreview, setError }) {
  const inputRef = useRef()

  const handleFile = async (file) => {
    if (!file) return
    setError(null)
    setResult(null)
    setPreview(URL.createObjectURL(file))
    setLoading(true)
    try {
      const data = await predictImage(file)
      setResult(data)
    } catch (err) {
      setError("Error analyzing image. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
      className="mt-4 w-full max-w-md border-2 border-dashed border-blue-400
        rounded-2xl p-10 text-center cursor-pointer hover:bg-blue-50 transition-colors"
    >
      <p className="text-gray-500">Drag & drop a mammogram here, or click to upload</p>
      <p className="text-xs text-gray-400 mt-1">Supports PNG, JPEG, DICOM (.dcm)</p>
      <input
        ref={inputRef} type="file"
        accept=".png,.jpg,.jpeg,.dcm"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  )
}
```

#### 4.17 `frontend/src/services/api.js`

```javascript
import axios from "axios"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080"

export async function predictImage(file) {
  const formData = new FormData()
  formData.append("file", file)

  const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" }
  })
  return response.data
}
```

#### 4.18 `frontend/.env.example`

```
VITE_API_URL=https://your-cloud-run-url.run.app
```

---

### PHASE 4 — Docker Compose (Local Full-Stack)

#### 4.19 `docker-compose.yml`

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports:
      - "8080:8080"
    volumes:
      - ./backend/app/models:/app/app/models
    environment:
      - ENV=development

  frontend:
    build: ./frontend
    ports:
      - "5173:80"
    environment:
      - VITE_API_URL=http://localhost:8080
    depends_on:
      - backend
```

---

### PHASE 5 — Cloud Deployment

#### 4.20 GCP Cloud Run — Backend

```bash
# Step 1: Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Step 2: Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fahm-backend ./backend

# Step 3: Deploy to Cloud Run
gcloud run deploy fahm-backend \
  --image gcr.io/YOUR_PROJECT_ID/fahm-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080

# Output: https://fahm-backend-XXXX.run.app  ← use this as VITE_API_URL
```

#### 4.21 Vercel — Frontend

```bash
# From /frontend directory
npm install -g vercel
vercel

# Set environment variable in Vercel dashboard:
# VITE_API_URL = https://fahm-backend-XXXX.run.app
```

---

## 5. PRIVACY & SECURITY IMPLEMENTATION

Implement and document all of the following:

| Requirement | Implementation |
|---|---|
| **HTTPS in transit** | GCP Cloud Run enforces TLS 1.2+ automatically |
| **No data at rest** | Images processed purely in memory — never written to disk or DB |
| **DICOM anonymization** | `dicom_utils.py` reads ONLY pixel data. Patient metadata (name, DOB, ID) is never accessed, logged, or stored |
| **No logging of PHI** | FastAPI access logs disabled for `/predict` endpoint; only status codes logged |
| **Saudi PDPL compliance** | No personal data collected. No cookies. No user accounts. Data minimization principle applied. |
| **CORS restriction** | Production: restrict `allow_origins` to frontend domain only |
| **Dockerfile security** | Non-root user in Dockerfile for backend container |

Add to `backend/app/main.py` for production:

```python
# Disable request body logging in production
import logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

---

## 6. README.md TEMPLATE

The agent must generate a complete `README.md` at the project root:

```markdown
# FAHM Mammogram Classifier

Binary classification of mammogram images: **Malignant vs. Benign/Normal**

## Live Demo
- **Frontend:** https://fahm-mammogram.vercel.app
- **API Docs:** https://fahm-backend-XXXX.run.app/docs

## Quick Start (Local)

### Prerequisites
- Docker + Docker Compose
- Python 3.11+ (for training only)
- Node.js 18+ (for frontend dev)

### Run with Docker Compose
git clone https://github.com/YOUR_USERNAME/fahm-mammogram
cd fahm-mammogram

# Add your trained model
cp /path/to/model.pt backend/app/models/model.pt

docker-compose up --build
# Frontend: http://localhost:5173
# API:      http://localhost:8080/docs

## Model Training

cd model
pip install -r requirements_train.txt
# Download CBIS-DDSM dataset from Kaggle → place in model/data/
python train.py
python evaluate.py  # generates ROC curve in docs/

## Architecture
[Frontend (Vercel)] → HTTPS → [FastAPI on Cloud Run] → [EfficientNet-B0 model]

## Privacy
- Images processed in-memory only — never persisted
- DICOM metadata stripped before processing
- HTTPS enforced end-to-end
- No PHI logged or stored
```

---

## 7. TECHNICAL REPORT OUTLINE (`/docs/report.md`)

Generate a PDF report covering these exact sections:

```
1. Architecture Overview
   - System diagram: Frontend → API → Model
   - Technology choices and rationale

2. Dataset
   - CBIS-DDSM description and source
   - Class distribution (malignant vs benign counts)
   - Preprocessing steps applied

3. ML Model
   - EfficientNet-B0 with transfer learning
   - Class imbalance handling (weighted loss)
   - Training setup (epochs, optimizer, LR)

4. Performance Metrics
   - Accuracy: XX%
   - Sensitivity (Recall): XX%     ← PRIORITIZE THIS
   - Specificity: XX%
   - AUC-ROC: X.XX
   - [Insert ROC curve image]
   - Discussion: Why accuracy alone fails for screening tasks
     (class imbalance → a model predicting all benign gets high accuracy
      but zero sensitivity — catastrophic for cancer screening)

5. Privacy & Data Security Framework
   - In-transit security (TLS/HTTPS via Cloud Run)
   - At-rest security (no persistence)
   - DICOM anonymization protocol
   - Saudi PDPL compliance rationale

6. Trade-offs & Future Work
   - Threshold tuning for higher sensitivity
   - Grad-CAM explainability heatmaps
   - DICOM viewer integration
   - Proper audit logging for compliance
   - Larger model (EfficientNet-B4) with more compute
```

---

## 8. SUBMISSION CHECKLIST

Before the deadline, verify all three deliverables:

- [ ] **GitHub Repo** — public, clean code, working `Dockerfile`, complete `README.md`
- [ ] **Live URL** — Vercel frontend loads, uploads work, results display correctly
- [ ] **PDF Report** — covers all 5 sections above, includes ROC curve image
- [ ] Model file committed or instructions to obtain it are clear
- [ ] `.env.example` files present (never commit real `.env`)
- [ ] API `/health` endpoint returns `{"status": "ok"}`
- [ ] API `/docs` (Swagger UI) accessible on deployed URL
- [ ] Sensitivity and AUC prominently reported (not just accuracy)
- [ ] Medical disclaimer visible in the UI

---

## 9. CRITICAL CONSTRAINTS — DO NOT VIOLATE

1. **Never write uploaded images to disk** — memory only
2. **Never log or print patient data from DICOM headers**
3. **Never commit model weights if file > 100MB** — use Git LFS or document download steps
4. **Never hardcode API keys or credentials** — use `.env` files
5. **Never report only accuracy** — sensitivity and AUC are the primary metrics for this task
6. **Frontend must show disclaimer** on every result: *"For research purposes only"*

---

## 10. AGENT EXECUTION ORDER

Execute phases strictly in this order:

```
1. Scaffold full repo structure
2. Write model/dataset.py + model/config.py
3. Write model/train.py (complete, runnable)
4. Write model/evaluate.py (generates all metrics + ROC curve)
5. Write all backend files (main.py → routers → services → utils)
6. Write backend/Dockerfile
7. Write all frontend files (App.jsx → components → services)
8. Write docker-compose.yml
9. Write README.md
10. Write docs/report.md template
11. Run deployment steps (Cloud Run + Vercel)
12. Verify checklist
```

---

*This context document is self-contained. The agent should not need to ask clarifying questions — all decisions have been made above. If a conflict arises between sections, prefer the most privacy-preserving and security-conscious option.*
