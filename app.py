from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import imghdr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Cancer Detection API", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: Restrict to local UI only ---
# In production, replace with your actual frontend domain.
ALLOWED_ORIGINS = [
    "null",                   # file:// origin (local HTML file)
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

MODEL_PATH = "cancer_detection_cnn (1).h5"
model = None

# --- Validation Constants ---
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB hard cap
MAX_IMAGE_DIMENSION = 4096               # pixels — blocks decompression bombs
ALLOWED_MIME_MAGIC = {"png", "jpeg", "gif", "bmp", "webp"}  # imghdr types
CONFIDENCE_UNCERTAIN_THRESHOLD = 60.0   # below this → "Uncertain"


# --- Model Loading ---
@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception:
            print("Error loading model. Check model file integrity.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}.")


@app.post("/predict")
@limiter.limit("20/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not available. Please contact support."}
        )

    try:
        # --- 1. Enforce file size limit (read in chunks) ---
        contents = b""
        chunk_size = 1024 * 64  # 64 KB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            contents += chunk
            if len(contents) > MAX_FILE_SIZE_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."}
                )

        # --- 2. Validate file type via magic bytes (not just extension/Content-Type) ---
        detected_type = imghdr.what(None, h=contents)
        if detected_type not in ALLOWED_MIME_MAGIC:
            return JSONResponse(
                status_code=415,
                content={"error": "Invalid file type. Only image files are accepted."}
            )

        # --- 3. Open image and check dimensions (decompression bomb protection) ---
        try:
            Image.MAX_IMAGE_PIXELS = MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(
                status_code=415,
                content={"error": "File could not be read as an image."}
            )

        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return JSONResponse(
                status_code=400,
                content={"error": f"Image dimensions too large. Maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels."}
            )

        # --- 4. Preprocess and predict ---
        image = image.resize((50, 50))
        img_array = np.array(image).astype("float32") / 255.0

        # --- OOD Heuristic Filter ---
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
        
        # Check if it's too dark (screenshots) or lacks the pink/purple H&E signature
        if (mean_r + mean_g + mean_b) / 3.0 < 0.2:
            return JSONResponse(status_code=400, content={"error": "Invalid Image: Does not appear to be an H&E stained slide (too dark)."})
        if mean_g > mean_r + 0.05 and mean_g > mean_b + 0.05:
            return JSONResponse(status_code=400, content={"error": "Invalid Image: Does not appear to be an H&E stained slide (wrong color profile)."})

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        score = float(prediction[0][0])

        is_malignant = score > 0.5
        label = "Malignant" if is_malignant else "Benign"
        confidence = score if is_malignant else (1.0 - score)
        confidence_pct = round(confidence * 100, 2)

        # --- 5. Flag low-confidence predictions as uncertain ---
        if confidence_pct < CONFIDENCE_UNCERTAIN_THRESHOLD:
            label = "Uncertain"

        # Return bounded response — raw score NOT returned to prevent model extraction
        return {
            "prediction": label,
            "confidence": confidence_pct,
        }

    except Exception:
        # Generic error — no stack trace, no file paths, no library names
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred during analysis. Please try again."}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
