"""
Ocular Emotion Detection — FastAPI Backend
Run with: uvicorn app:app --reload
"""

import warnings
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager
from typing import Dict, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.tasks import python  # noqa: F401  (registers submodules)
from mediapipe.tasks.python import vision
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor as skimage_gabor
from sklearn.pipeline import Pipeline

# ================================================================
#  CONFIGURATION
# ================================================================

CONFIG = {
    "IMAGE_W": 128,
    "IMAGE_H": 64,
    "HOG_ORIENTATIONS": 9,
    "HOG_PIXELS_PER_CELL": (8, 8),
    "HOG_CELLS_PER_BLOCK": (2, 2),
    "LBP_CONFIGS": [
        {"points": 8,  "radius": 1},
        {"points": 16, "radius": 2},
        {"points": 24, "radius": 3},
    ],
    "GABOR_FREQUENCIES": [0.10, 0.25, 0.40],
    "GABOR_THETAS": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    "SPATIAL_GRID_ROWS": 4,
    "SPATIAL_GRID_COLS": 8,
    "LABELS": ["anger", "happy", "sad", "surprise"],
    "MODEL_PATH": "emotion_model.pkl",
    "LANDMARKER_PATH": "face_landmarker.task",
}

# ================================================================
#  APP STATE  (globals loaded once at startup)
# ================================================================

class AppState:
    model: Pipeline = None
    landmarker: vision.FaceLandmarker = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server starts."""
    # --- Emotion model ---
    state.model = joblib.load(CONFIG["MODEL_PATH"])

    # --- MediaPipe FaceLandmarker ---
    base_options = mp.tasks.BaseOptions(
        model_asset_path=CONFIG["LANDMARKER_PATH"]
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.75,
        running_mode=vision.RunningMode.IMAGE,
    )
    state.landmarker = vision.FaceLandmarker.create_from_options(options)

    yield  # ← server is running

    # Cleanup (if needed)
    state.landmarker.close()


app = FastAPI(
    title="Ocular Emotion Detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tighten this in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # Force FastAPI to use cdnjs instead of jsdelivr
        swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.9.0/swagger-ui.css",
    )

# ================================================================
#  MEDIAPIPE — OCULAR REGION CROP  (in-memory, no disk I/O)
# ================================================================

OCULAR_LANDMARK_INDICES = [
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,
    33, 133, 159, 145, 263, 362, 386, 374,
    69, 299,
    118, 347,
    143, 372,
]


def crop_ocular_region(bgr_image: np.ndarray) -> np.ndarray:
    """
    Detect face landmarks and return the cropped ocular region.
    Raises ValueError if no face is detected.
    """
    h, w = bgr_image.shape[:2]

    # MediaPipe expects RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    result = state.landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise ValueError("No face detected in the uploaded image.")

    landmarks = result.face_landmarks[0]
    coords_x = [int(landmarks[i].x * w) for i in OCULAR_LANDMARK_INDICES]
    coords_y = [int(landmarks[i].y * h) for i in OCULAR_LANDMARK_INDICES]

    x_min = max(0, min(coords_x))
    x_max = min(w, max(coords_x))
    y_min = max(0, min(coords_y))
    y_max = min(h, max(coords_y))

    cropped = bgr_image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        raise ValueError("Cropped ocular region is empty — check landmark indices.")

    return cropped


# ================================================================
#  PREPROCESSING
# ================================================================

def _pad_to_aspect_ratio(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape
    target_ratio = target_w / target_h
    current_ratio = w / (h + 1e-9)

    if current_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        pad_top = (new_h - h) // 2
        pad_bot = new_h - h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, cv2.BORDER_REFLECT)
    elif current_ratio < target_ratio:
        new_w = int(round(h * target_ratio))
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT)

    return img


def preprocess_image(img: np.ndarray) -> np.ndarray:
    W, H = CONFIG["IMAGE_W"], CONFIG["IMAGE_H"]

    img = _pad_to_aspect_ratio(img, W, H)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)
    return img


# ================================================================
#  FEATURE EXTRACTION
# ================================================================

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    return hog(
        img,
        orientations=CONFIG["HOG_ORIENTATIONS"],
        pixels_per_cell=CONFIG["HOG_PIXELS_PER_CELL"],
        cells_per_block=CONFIG["HOG_CELLS_PER_BLOCK"],
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    features = []
    for cfg in CONFIG["LBP_CONFIGS"]:
        P, R = cfg["points"], cfg["radius"]
        lbp = local_binary_pattern(img, P, R, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, P + 3),
            range=(0, P + 2),
        )
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-9
        features.append(hist)
    return np.concatenate(features)


def extract_gabor_features(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    features = []
    for freq in CONFIG["GABOR_FREQUENCIES"]:
        for theta in CONFIG["GABOR_THETAS"]:
            real, imag = skimage_gabor(img_f, frequency=freq, theta=theta)
            mag = np.hypot(real, imag)
            features.extend([float(mag.mean()), float(mag.std())])
    return np.array(features, dtype=np.float64)


def extract_spatial_features(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    rows, cols = CONFIG["SPATIAL_GRID_ROWS"], CONFIG["SPATIAL_GRID_COLS"]
    cell_h = H // rows
    cell_w = W // cols

    grid_features = []
    for r in range(rows):
        for c in range(cols):
            cell = img[
                r * cell_h : (r + 1) * cell_h,
                c * cell_w : (c + 1) * cell_w,
            ].astype(np.float64)
            grid_features.extend([cell.mean(), cell.std()])

    img_f = img.astype(np.float64)
    top_mean  = img_f[: H // 2, :].mean()
    bot_mean  = img_f[H // 2 :, :].mean()
    left_mean = img_f[:, : W // 2].mean()
    rgt_mean  = img_f[:, W // 2 :].mean()

    sobel_x  = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobel_x, sobel_y)

    structural = [
        top_mean,
        bot_mean,
        top_mean / (bot_mean + 1e-9),
        left_mean,
        rgt_mean,
        abs(left_mean - rgt_mean),
        np.abs(sobel_x).mean(),
        np.abs(sobel_y).mean(),
        grad_mag.mean(),
    ]

    return np.concatenate([grid_features, structural])


def extract_features(img: np.ndarray) -> np.ndarray:
    return np.concatenate([
        extract_hog_features(img),
        extract_lbp_features(img),
        extract_gabor_features(img),
        extract_spatial_features(img),
    ])


# ================================================================
#  INFERENCE  (accepts in-memory grayscale image array)
# ================================================================

def predict(gray_img: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """
    Run the full preprocessing → feature extraction → SVM prediction pipeline.

    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image array (H × W, uint8).

    Returns
    -------
    label      : predicted emotion string
    confidence : max class probability
    all_probs  : {class_name: probability} for all classes
    """
    img = preprocess_image(gray_img)
    features = extract_features(img).reshape(1, -1)

    label = state.model.predict(features)[0]
    proba = state.model.predict_proba(features)[0]
    all_probs = dict(zip(state.model.classes_, proba.tolist()))
    confidence = float(proba.max())

    return label, confidence, all_probs


# ================================================================
#  HELPERS
# ================================================================

def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR NumPy array."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded file as an image.")
    return img


# ================================================================
#  ENDPOINT
# ================================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Image file to analyze"),
    image_type: str = Form(
        ...,
        description='Either "full_face" or "cropped_ocular"',
    ),
):
    """
    Detect the ocular emotion in the uploaded image.

    - **full_face**: MediaPipe crops the ocular region first, then runs inference.
    - **cropped_ocular**: Skips cropping and runs inference directly.
    """
    # ── Validate image_type ──────────────────────────────────────
    if image_type not in ("full_face", "cropped_ocular"):
        raise HTTPException(
            status_code=422,
            detail='image_type must be "full_face" or "cropped_ocular".',
        )

    # ── Read uploaded bytes ──────────────────────────────────────
    raw_bytes = await file.read()
    try:
        bgr_image = decode_image_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Conditional cropping ─────────────────────────────────────
    if image_type == "full_face":
        try:
            bgr_image = crop_ocular_region(bgr_image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ── Convert to grayscale for the ML pipeline ─────────────────
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # ── Run inference ─────────────────────────────────────────────
    try:
        label, confidence, all_probs = predict(gray_image)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        )

    return {
        "predicted_emotion": label,
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
    }


# ================================================================
#  HEALTH CHECK
# ================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "landmarker_loaded": state.landmarker is not None,
    }