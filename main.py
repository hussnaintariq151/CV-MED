from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid
import logging
import torch
from torchvision import transforms
from PIL import Image

from appfile.video_loader import extract_frames, get_fps
from appfile.utils import load_model, predict_frame

app = FastAPI()

# Static files and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model once
MODEL_PATH = "tbut_resnet18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, device)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("intro.html", {"request": request})

@app.get("/upload-page", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload_page.html", {"request": request})


@app.post("/upload-video/")
async def upload_and_detect(file: UploadFile = File(...)):
    # Validate extension
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # Save uploaded video
    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logging.info(f"📁 Saved uploaded video to: {file_path}")

    try:
        frames_tensor = extract_frames(file_path)
        fps = get_fps(file_path)
        logging.info(f"📊 Video converted to {frames_tensor.shape[0]} frames at {fps} FPS")
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")

    # Optional cleanup
    os.remove(file_path)

    # TBUT Detection State Machine
    frame_id = 0
    state = "idle"
    blink_end_frame = None
    tbut_start_frame = None
    frame_predictions = []

    for i, frame_tensor in enumerate(frames_tensor):
        frame_id += 1
        image_pil = transforms.ToPILImage()(frame_tensor)
        label, conf = predict_frame(model, image_pil, device)

        frame_predictions.append({
            "frame": frame_id,
            "label": label,
            "confidence": round(conf, 4)
        })

        logging.info(f"🧠 Frame {frame_id}: {label} ({conf:.2%})")

        if state == "idle":
            if label == "blink":
                state = "blinking"
        elif state == "blinking":
            if label == "good":
                blink_end_frame = frame_id
                state = "waiting_tbut"
            elif label == "tbut":
                blink_end_frame = frame_id - 1  # last blink assumed
                tbut_start_frame = frame_id
                break
        elif state == "waiting_tbut":
            if label == "tbut":
                tbut_start_frame = frame_id
                break

    if blink_end_frame and tbut_start_frame:
        tbut_seconds = (tbut_start_frame - blink_end_frame) / fps
        logging.info(f"✅ TBUT Detected: {tbut_seconds:.2f} seconds")
        result = {
            "tbut_seconds": round(tbut_seconds, 2),
            "message": "✅ TBUT Detected",
            "total_frames": frame_id,
            "frame_predictions": frame_predictions
        }
    else:
        logging.warning("❌ TBUT could not be detected.")
        result = {
            "tbut_seconds": None,
            "message": "❌ TBUT could not be detected",
            "total_frames": frame_id,
            "frame_predictions": frame_predictions
        }

    return JSONResponse(content=result)
