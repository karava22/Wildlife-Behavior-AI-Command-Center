import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# --- 1. CONFIGURATION (Must match your training) ---
MODEL_PATH = r"C:\Users\Anusha\OneDrive\Desktop\Animal_AI\animals pics\wildlife_behavior_model.pth"
# Ensure these are in alphabetical order (the way PyTorch loads them)
CLASSES = ['Eating', 'Hunting', 'Resting', 'Sleeping', 'Walking'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD THE TRAINED BRAIN ---
def load_trained_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. PREDICT ON A SINGLE IMAGE ---
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred = torch.max(probabilities, 0)
    
    result = CLASSES[pred.item()]
    print(f"📸 Image Result: {result} ({confidence.item()*100:.2f}%)")
    return result

# --- 4. PREDICT ON A VIDEO (Automated Recognition) ---
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f"🎥 Analyzing Video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Prepare frame for AI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_t)
            _, pred = torch.max(outputs, 1)
            label = CLASSES[pred.item()]

        # Draw the behavior on the screen
        cv2.putText(frame, f"BEHAVIOR: {label}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Wildlife AI Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to stop
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 5. CHOOSE WHAT TO TEST ---
if __name__ == "__main__":
    # To test an image:
    test_image_path = r"animals pics\Sleeping\sleeping1.jpg" 

    print("--- Wildlife AI Recognition System ---")
    
    # Check if the image exists
    if os.path.exists(test_image_path):
        print(f"Checking image: {test_image_path}")
        
        # ACTIVATE the prediction
        result = predict_image(test_image_path)
        
        print(f"\nFinal Determination: The AI thinks this animal is {result}.")
    else:
        print(f"❌ ERROR: I could not find the file at {test_image_path}")
        print("Please check the folder path and the filename again.")