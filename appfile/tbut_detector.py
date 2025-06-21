import cv2
import torch
from PIL import Image
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from appfile.utils import load_model, predict_frame

# Setup
model_path = "tbut_resnet18.pth"
video_path = r"D:\Job\CV-MED\synthetic_videos\synthetic_tbut_eye.avi"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎥 FPS: {fps}")

frame_id = 0
last_blink_frame = None
tbut_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    label, conf = predict_frame(model, image_pil, device)
    print(f"🧠 Frame {frame_id}: {label} ({conf:.2%})")

    if label == "blink":
        last_blink_frame = frame_id

    elif label == "tbut" and last_blink_frame is not None:
        tbut_frame = frame_id
        break  # First tbut after blink

cap.release()

if last_blink_frame and tbut_frame:
    tbut_seconds = (tbut_frame - last_blink_frame) / fps
    print(f"✅ TBUT Detected: {tbut_seconds:.2f} seconds")
else:
    print("❌ TBUT could not be detected in this video.")
