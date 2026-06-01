# FAHM Mammogram Classifier

Binary classification of mammogram images: **Malignant vs. Benign/Normal**

## Live Demo
- **Frontend:** [https://fahm-mammogram-classifier.vercel.app](https://fahm-mammogram-classifier.vercel.app)
- **API Docs:** [https://adeel41-mammogram-classifier.hf.space/docs](https://adeel41-mammogram-classifier.hf.space/docs)

## Quick Start (Local)

### Prerequisites
- Docker + Docker Compose
- Python 3.11+ (for training only)
- Node.js 18+ (for frontend dev)

### Run with Docker Compose
```bash
git clone https://github.com/adeelshah41/fahm-mammogram-classifier
cd fahm-mammogram-classifier

# Model weights are tracked via Git LFS, ensure they are pulled:
git lfs pull

docker-compose up --build
# Frontend: http://localhost:5173
# API:      http://localhost:8080/docs
```

## Model Training

```bash
cd model
pip install -r requirements_train.txt
# Download CBIS-DDSM dataset from Kaggle → place in model/data/
python train.py
python evaluate.py  # generates ROC curve in docs/
```

## Architecture
[Frontend (Vercel)] → HTTPS → [FastAPI on Hugging Face Spaces] → [EfficientNet-B0 model]

## Privacy & Security
- Images processed in-memory only — never persisted
- DICOM metadata stripped before processing
- HTTPS enforced end-to-end
- No PHI logged or stored
