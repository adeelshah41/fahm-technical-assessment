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

app.include_router(predict.router)

@app.get("/health")
def health():
    return {"status": "ok"}
