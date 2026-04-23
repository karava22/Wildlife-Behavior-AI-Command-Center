import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Wildlife AI Dashboard", layout="wide")

# Ensure required libraries are present
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ModuleNotFoundError:
    st.error("Missing dependencies. Please ensure 'requirements.txt' is in your GitHub repo.")
    st.stop()

# --- 1. SETTINGS & PATHS (Cloud Optimized) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logic to find the model file regardless of folder structure on GitHub
if os.path.exists("wildlife_behavior_model.pth"):
    MODEL_PATH = "wildlife_behavior_model.pth"
elif os.path.exists(os.path.join(BASE_DIR, "animals pics", "wildlife_behavior_model.pth")):
    MODEL_PATH = os.path.join(BASE_DIR, "animals pics", "wildlife_behavior_model.pth")
else:
    MODEL_PATH = "wildlife_behavior_model.pth" # Default fallback

HISTORY_FILE = "detection_history.csv"
CLASSES = ['Eating', 'Hunting', 'Resting', 'Sleeping', 'Walking']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}. Please check your GitHub file list.")
        st.stop()
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- 3. SMART HISTORY LOGIC ---
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["Timestamp", "File", "Detected Behavior", "Confidence"])
    df = pd.read_csv(HISTORY_FILE)
    rename_map = {'Behavior': 'Detected Behavior', 'File Name': 'File', 'Score/Coverage': 'Confidence'}
    return df.rename(columns=rename_map)

def save_detection(filename, behavior, confidence):
    df = get_history()
    new_row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "File": filename,
        "Detected Behavior": behavior,
        "Confidence": f"{confidence:.2f}%"
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

# --- 4. STREAMLIT UI ---
st.title("🐾 Wildlife Behavior AI Command Center")

tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "📜 History"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Media")
        # Added accept_multiple_files=False to prevent the PIL Error
        uploaded_file = st.file_uploader("Upload ONE Image or Video", type=["jpg", "png", "jpeg", "jfif", "mp4"], accept_multiple_files=False)

    with col2:
        if uploaded_file:
            file_type = uploaded_file.type.split('/')[0]
            
            if file_type == 'image':
                try:
                    # --- IMAGE PREDICTION ---
                    img = Image.open(uploaded_file).convert('RGB')
                    st.image(img, width=400, caption=f"Uploaded: {uploaded_file.name}")
                    
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(img_t)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        conf, pred = torch.max(probs, 0)
                    
                    res = CLASSES[pred.item()]
                    conf_pct = conf.item() * 100
                    
                    if conf_pct > 70:
                        st.success(f"### Result: {res}")
                    else:
                        st.warning(f"### Result: {res} (Uncertain)")
                        
                    st.write(f"**AI Confidence Score:** {conf_pct:.2f}%")
                    save_detection(uploaded_file.name, res, conf_pct)
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}. Try a standard .jpg or .png file.")

            elif file_type == 'video':
                st.video(uploaded_file)
                save_detection(f"[VIDEO] {uploaded_file.name}", "Video Logged", 0.0)
                st.info("Video added to history.")

with tab2:
    st.subheader("📈 Behavioral Trends")
    df_hist = get_history()
    if not df_hist.empty and "Detected Behavior" in df_hist.columns:
        col_left, col_right = st.columns(2)
        with col_left:
            fig, ax = plt.subplots()
            df_hist["Detected Behavior"].value_counts().plot(kind='bar', color='#3498db', ax=ax)
            plt.title("Activities Captured")
            st.pyplot(fig)
        with col_right:
            st.metric("Total Detections", len(df_hist))
            st.metric("Top Activity", df_hist["Detected Behavior"].mode()[0])
    else:
        st.write("No data found in history yet.")

with tab3:
    st.subheader("Full History (Stored in CSV)")
    df_display = get_history()
    st.dataframe(df_display.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    st.download_button("📥 Download History CSV", df_display.to_csv(index=False), "wildlife_history.csv", "text/csv")
