import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import cv2
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# --- 1. CONFIGURATION ---
MODEL_PATH = r"C:\Users\Anusha\OneDrive\Desktop\Animal_AI\animals pics\wildlife_behavior_model.pth"
CLASSES = ['Eating', 'Hunting', 'Resting', 'Sleeping', 'Walking'] 
HISTORY_FILE = "detection_history.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD THE AI BRAIN ---
def load_ai_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()
    return model

ai_brain = load_ai_model()

# Standard Processing for the AI
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. THE APP INTERFACE ---
class WildlifeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Wildlife Behavior Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a1a") # Dark Theme

        # Header Title
        self.header = tk.Label(root, text="WILDLIFE BEHAVIOR AI", font=("Verdana", 24, "bold"), bg="#1a1a1a", fg="#00FF00")
        self.header.pack(pady=20)

        # Control Buttons
        self.btn_frame = tk.Frame(root, bg="#1a1a1a")
        self.btn_frame.pack(pady=10)

        self.img_btn = tk.Button(self.btn_frame, text="📸 Analyze Image", command=self.upload_image, width=20, bg="#27ae60", fg="white", font=("Arial", 12, "bold"))
        self.img_btn.grid(row=0, column=0, padx=10)

        self.vid_btn = tk.Button(self.btn_frame, text="🎥 Analyze Video", command=self.upload_video, width=20, bg="#2980b9", fg="white", font=("Arial", 12, "bold"))
        self.vid_btn.grid(row=0, column=1, padx=10)

        self.hist_btn = tk.Button(self.btn_frame, text="📜 View History", command=self.show_history, width=20, bg="#8e44ad", fg="white", font=("Arial", 12, "bold"))
        self.hist_btn.grid(row=0, column=2, padx=10)

        # Main Display Panel (Shows the image)
        self.panel = tk.Label(root, bg="#333333", width=500, height=400)
        self.panel.pack(pady=20)

        # Prediction Result Label
        self.result_label = tk.Label(root, text="Upload an Image or Video to Start...", font=("Arial", 18), bg="#1a1a1a", fg="#ecf0f1")
        self.result_label.pack()

    # --- 4. DATA LOGGING LOGIC ---
    def save_to_history(self, filename, behavior, confidence):
        data = {
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "File": [filename],
            "Type": ["Video" if "[VIDEO]" in filename else "Image"],
            "Detected Behavior": [behavior],
            "Score/Coverage": [f"{confidence:.2f}%"]
        }
        df = pd.DataFrame(data)
        # Create file if it doesn't exist, otherwise append
        if not os.path.isfile(HISTORY_FILE):
            df.to_csv(HISTORY_FILE, index=False)
        else:
            df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

    # --- 5. IMAGE DETECTION LOGIC ---
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not file_path: return

        # AI Prediction
        img = Image.open(file_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = ai_brain(img_t)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)
        
        behavior = CLASSES[pred.item()]
        conf_pct = conf.item() * 100

        # Update App Display
        self.result_label.config(text=f"IMAGE RESULT: {behavior} ({conf_pct:.2f}%)", fg="#00FF00")
        
        # Resize image to fit the window
        img.thumbnail((500, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.panel.configure(image=img_tk)
        self.panel.image = img_tk

        # Save to History
        self.save_to_history(os.path.basename(file_path), behavior, conf_pct)

    # --- 6. VIDEO DETECTION LOGIC ---
    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if not file_path: return

        cap = cv2.VideoCapture(file_path)
        video_stats = {cls: 0 for cls in CLASSES} 
        total_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # AI Logic
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = ai_brain(img_t)
                _, pred = torch.max(outputs, 1)
                label = CLASSES[pred.item()]
            
            video_stats[label] += 1
            total_frames += 1

            # Draw on screen
            cv2.putText(frame, f"AI RECOGNITION: {label}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Automated Video Analysis (Press 'q' to stop)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

        # Save Summary to History
        if total_frames > 0:
            dominant_behavior = max(video_stats, key=video_stats.get)
            coverage = (video_stats[dominant_behavior] / total_frames) * 100
            
            self.save_to_history(f"[VIDEO] {os.path.basename(file_path)}", dominant_behavior, coverage)
            self.result_label.config(text=f"VIDEO SUMMARY: {dominant_behavior} ({coverage:.1f}%)", fg="#3498db")
            messagebox.showinfo("Analysis Complete", f"Main Behavior: {dominant_behavior}\nCoverage: {coverage:.1f}%")

    # --- 7. HISTORY VIEWER ---
    def show_history(self):
        if os.path.exists(HISTORY_FILE):
            # Opens the CSV in your default app (Excel, Notepad, etc.)
            os.startfile(HISTORY_FILE) 
        else:
            messagebox.showwarning("History", "The history file is empty. Analyze some wildlife first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = WildlifeApp(root)
    root.mainloop()