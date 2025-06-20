import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Optional: Load image path from CLI
image_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Job\CV-MED\test_images\frame_000.jpg"

# Check if image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
class_names = ['blink', 'good', 'tbut']  # 🔁 Update with your actual classes

# Define transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load and preprocess image
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Update num classes
model.load_state_dict(torch.load("tbut_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)
    predicted_class = class_names[pred.item()]

print(f"✅ Prediction: {predicted_class}")
