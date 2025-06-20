import cv2
import os

# === Configuration ===
video_path = "synthetic_videos/synthetic_tbut_eye.avi"
output_dir = "tbut_dataset"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_idx = 0

# Read frames one by one
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Label assignment based on frame index
    if 10 <= frame_idx <= 15:
        label = "blink"  # Optional; you can skip this if you want binary
    elif frame_idx >= 20:
        label = "tbut"
    else:
        label = "good"

    # Create label directory if not exists
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Save frame image
    filename = f"frame_{frame_idx:03d}.jpg"
    filepath = os.path.join(label_dir, filename)
    cv2.imwrite(filepath, frame)

    frame_idx += 1

cap.release()
print(f"✅ Extracted and labeled {frame_idx} frames into: {output_dir}")
