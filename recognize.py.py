import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Setup (Must match the training setup)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['eating', 'hunting', 'sleeping', 'walking'] # Matches your folders

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("wildlife_ai_model.pth"))
model.to(DEVICE)
model.eval()

def auto_recognize(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(img_t)
        _, pred = torch.max(out, 1)
        
    print(f"AUTOMATED DETECTION: The animal is {CLASSES[pred.item()]}")

# To use:
# auto_recognize('test_photo_from_forest.jpg')