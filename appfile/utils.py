import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Constants
class_names = ['blink', 'good', 'tbut']

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load model only once
def load_model(model_path: str, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict from PIL.Image (no need to save to disk)
def predict_frame(model, image_pil: Image.Image, device):
    image = transform(image_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        _, pred = torch.max(probs, 1)
        return class_names[pred.item()], probs[0][pred.item()].item()
