from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.preprocessor import preprocess_image
from app.services.inference import run_inference

router = APIRouter()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    clahe: bool = Query(default=True)
):
    filename = file.filename.lower()
    if not (filename.endswith(('.jpg','.jpeg','.png','.dcm'))):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    tensor = preprocess_image(contents, filename, apply_clahe=clahe)
    label, confidence = run_inference(tensor)

    return {
        "prediction": label,
        "confidence": confidence,
        "disclaimer": "For research purposes only. Not a clinical diagnostic tool."
    }
