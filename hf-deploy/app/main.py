from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
import logging

# Disable request body logging in production
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title="FAHM Mammogram Classifier API",
    description="Binary classification: Malignant vs Benign/Normal",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

from app.services.inference import load_model
import torch

@app.on_event("startup")
async def startup_event():
    print("Pre-loading and warming up model...")
    model = load_model()
    dummy_input = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        _ = model(dummy_input)
    print("Model loaded and warmed up successfully.")

app.include_router(predict.router)

@app.get("/health")
def health():
    return {"status": "ok"}
