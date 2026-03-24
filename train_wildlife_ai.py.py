import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import kagglehub

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_and_train():
    # 1. DOWNLOAD DATASET
    print("Step 1: Downloading dataset from Kaggle...")
    raw_path = kagglehub.dataset_download("travisdaws/spatiotemporal-wildlife-dataset")
    
    # 2. AUTOMATE DATA ORGANIZATION
    # We create a local 'data' folder in your VS Code project
    base_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(base_dir, exist_ok=True)
    
    print("Step 2: Automatically organizing images by behavior...")
    # This searches the downloaded files for behavior keywords
    # If your dataset has a CSV in 'obs_and_meta', we could use that too.
    image_source = os.path.join(raw_path, 'images')
    
    behaviors = ['hunting', 'sleeping', 'walking', 'eating']
    for b in behaviors:
        os.makedirs(os.path.join(base_dir, b), exist_ok=True)

    # Move files based on filename keywords (Automation)
    for root, _, files in os.walk(image_source):
        for file in files:
            fname = file.lower()
            target = None
            if any(k in fname for k in ['eat', 'feed', 'browse']): target = 'eating'
            elif any(k in fname for k in ['sleep', 'rest', 'bed']): target = 'sleeping'
            elif any(k in fname for k in ['walk', 'run', 'move']): target = 'walking'
            elif any(k in fname for k in ['hunt', 'kill', 'attack']): target = 'hunting'
            
            if target:
                shutil.copy(os.path.join(root, file), os.path.join(base_dir, target, file))

    # 3. PREPARE DATA LOADERS
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=base_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 4. INITIALIZE MODEL (ResNet50)
    print("Step 3: Initializing AI Model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. TRAINING LOOP
    print("Step 4: Training Started...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # 6. SAVE THE AUTOMATED BRAIN
    torch.save(model.state_dict(), "wildlife_ai_model.pth")
    print("Automation Ready! Model saved as 'wildlife_ai_model.pth'")

if __name__ == "__main__":
    setup_and_train()