---
title: FAHM Mammogram Classifier
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# FAHM — AI-Powered Mammogram Classification API

Binary classification of breast mammogram images as **Malignant** or **Benign/Normal** using an EfficientNet-B0 deep learning model trained on the CBIS-DDSM dataset.

## API Endpoints

- `GET /health` — Health check
- `POST /predict` — Upload a mammogram image (JPG, PNG, or DICOM) and receive a classification result

## Tech Stack

- **FastAPI** — High-performance Python web framework
- **PyTorch** — Deep learning inference engine
- **EfficientNet-B0** — Lightweight CNN architecture
- **CBIS-DDSM** — Curated Breast Imaging Subset of the Digital Database for Screening Mammography
