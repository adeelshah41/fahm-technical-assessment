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
        pil_img = Image.open(io.BytesIO(file_bytes))
    
    # Apply CLAHE to enhance contrast of mammogram (train/inference parity)
    import cv2
    import numpy as np
    img_np = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_np)
    pil_img = Image.fromarray(img_clahe).convert("RGB")
    
    tensor = transform(pil_img).unsqueeze(0)
    return tensor

