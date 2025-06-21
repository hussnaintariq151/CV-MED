import torch
from torchvision import models, transforms
from PIL import Image
import os

# Define the model architecture
model = models.resnet18(weights=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # change '2' to your actual number of classes

# Load the trained weights
model.load_state_dict(torch.load("tbut_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

print("✅ Model loaded successfully!")
