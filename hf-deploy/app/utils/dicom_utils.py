import pydicom
import numpy as np
from PIL import Image
import io

def dicom_to_pil(file_bytes: bytes) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
    pixel_array = (pixel_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(pixel_array).convert("RGB")
    return pil_img
