import cv2
import torch
from torchvision import transforms
from PIL import Image

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def extract_frames(video_path: str):
    """Extract frames from a video file and return them as a list of tensors."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        tensor = transform(image_pil)
        frames.append(tensor)
        frame_id += 1

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from the video.")

    return torch.stack(frames)

def get_fps(video_path: str):
    """Return frames per second (FPS) of a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# Test run
if __name__ == "__main__":
    test_path = r"D:\Job\CV-MED\synthetic_videos\synthetic_tbut_eye.avi"  # Replace with real test file
    try:
        tensor_frames = extract_frames(test_path)
        print(f"✅ Extracted tensor shape: {tensor_frames.shape}")  # For CLI confirmation
    except Exception as e:
        print(str(e))
