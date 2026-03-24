import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# --- 1. CONFIGURATION ---
DATA_PATH = r"C:\Users\Anusha\OneDrive\Desktop\Animal_AI\animals pics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Keep small for standard laptops
EPOCHS = 15     # 15 rounds of learning

def train_behavior_ai():
    # --- 2. THE "UNSTRUCTURED" TRANSFORMER ---
    # This makes the AI ignore shadows, grass, and weather
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4), # Handles forest shadows
        transforms.RandomRotation(15), # Handles animals on slopes/hills
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 3. LOAD YOUR DATA ---
    try:
        full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        
        print(f"✅ Success! Training on {len(full_dataset)} images.")
        print(f"Behaviors identified: {full_dataset.classes}")
    except Exception as e:
        print(f"❌ Error: {e}. Ensure you have sub-folders for each behavior.")
        return

    # --- 4. BUILD THE BRAIN (ResNet50) ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Customize the final layer for your 4 behaviors
    model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # --- 5. TRAINING LOOP ---
    print("Step 2: Training Automated AI...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Understanding Level: {1 - (total_loss/len(train_loader)):.2%}")

    # --- 6. SAVE THE AUTOMATED SYSTEM ---
    model_name = "wildlife_behavior_model.pth"
    torch.save(model.state_dict(), os.path.join(DATA_PATH, model_name))
    print(f"🚀 SUCCESS! Automated AI saved at: {os.path.join(DATA_PATH, model_name)}")

if __name__ == "__main__":
    train_behavior_ai()