<div align="center">

# 🩺 FAHM Mammogram Classifier

**AI-Powered Binary Classification of Mammogram Images (Malignant vs. Benign/Normal)**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Frontend-00D8FF?style=for-the-badge&logo=react&logoColor=white)](https://fahm-mammogram-classifier.vercel.app)
[![API Docs](https://img.shields.io/badge/API_Docs-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://adeel41-mammogram-classifier.hf.space/docs)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

This project is a secure, decoupled full-stack AI application that processes mammogram scans and outputs clinical predictions with confidence scores.

</div>

---

## 🏗️ System Architecture

The application separates concerns between a highly responsive React client and an isolated, containerized Deep Learning inference engine, ensuring maximum performance, zero UI-blocking, and strict data security.

```mermaid
graph LR
    subgraph Client Environment
        A[Vercel CDN] -->|Serves Static UI| B(Client Browser<br/>React UI)
    end
    
    subgraph Hugging Face Spaces Engine
        C(HF Spaces<br/>FastAPI) -->|Bytes Buffer| D[Image Preprocessor<br/>1. Strip PHI<br/>2. Apply CLAHE]
        D -->|Normalized Tensor| E[PyTorch<br/>EfficientNet-B0]
    end
    
    B -->|HTTPS POST<br/>Image or DICOM| C
    E -.->|Probability| C
    C -.->|JSON Response<br/>Prediction & Confidence| B
    
    style A fill:#000,stroke:#333,color:#fff
    style B fill:#111,stroke:#444,color:#fff
    style C fill:#222,stroke:#555,color:#fff
    style D fill:#222,stroke:#555,color:#fff
    style E fill:#222,stroke:#555,color:#fff
```

## 🔄 Project Workflow Phases

The development of this system followed a rigorous end-to-end Machine Learning lifecycle:

```mermaid
flowchart TD
    A([🔍 1. Research & Data Collection<br/>CBIS-DDSM Dataset Analysis]) --> B([🧹 2. Data Preprocessing<br/>DICOM Parsing & CLAHE Enhancement])
    B --> C([🧠 3. Model Training<br/>EfficientNet-B0 & Hyperparameter Tuning])
    C --> D([⚙️ 4. Backend API Development<br/>FastAPI & PyTorch Inference Integration])
    D --> E([🎨 5. Frontend UI Development<br/>React, Tailwind, & UX Design])
    E --> F([🔗 6. Full-Stack Integration<br/>CORS, API Connectivity, Dockerization])
    F --> G([🚀 7. Cloud Deployment<br/>Vercel CDN & Hugging Face Spaces])
    G --> H([📝 8. Documentation<br/>Technical Reports & Final Delivery])
    
    style A fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style B fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style C fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style D fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style E fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style F fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style G fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
    style H fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- **Docker + Docker Compose**
- **Python 3.11+** (for local model training)
- **Node.js 18+** (for frontend development)

### Run with Docker Compose
The entire stack can be spun up locally using containers.

```bash
git clone https://github.com/adeelshah41/fahm-technical-assessment
cd fahm-technical-assessment

# Model weights are tracked via Git LFS, ensure they are pulled:
git lfs pull

# Start both frontend and backend services
docker-compose up --build
```
* **Frontend:** `http://localhost:5173`
* **Backend API Docs:** `http://localhost:8080/docs`

---

## 🧠 Model Training

To retrain the EfficientNet-B0 classifier from scratch:

```bash
cd model
pip install -r requirements_train.txt

# 1. Download CBIS-DDSM dataset from Kaggle
# 2. Place files in the model/data/ directory

python train.py      # Starts the PyTorch training loop
python evaluate.py   # Generates metrics and ROC curves in docs/
```

---

## 🔒 Privacy & Security Protocols

Medical imaging applications demand strict adherence to data protection laws (e.g., Saudi PDPL, HIPAA). This architecture guarantees compliance through:

- **Zero Persistence Processing:** Images are processed entirely in-memory using ephemeral byte streams. Nothing is written to disk or databases.
- **DICOM Anonymization:** Extracted pixel arrays are completely stripped of Protected Health Information (PHI) before hitting the inference engine.
- **Transport Security:** End-to-end TLS 1.2+ encryption (HTTPS) prevents packet interception.
- **Data Minimization:** No access logs contain payload contents.
