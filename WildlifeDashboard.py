import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Wildlife AI Dashboard", layout="wide")

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ModuleNotFoundError as exc:
    st.error(
        "Missing a required dependency for the dashboard. "
        "Make sure the app is deployed with the packages from requirements.txt."
    )
    st.exception(exc)
    st.stop()

# --- 1. SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "animals pics", "wildlife_behavior_model.pth")
HISTORY_FILE = os.path.join(BASE_DIR, "detection_history.csv")
CLASSES = ['Eating', 'Hunting', 'Resting', 'Sleeping', 'Walking']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

try:
    model = load_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that the .pth file is included in the repo "
        "and that the path is correct on Streamlit Cloud."
    )
    st.exception(exc)
    st.stop()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- 3. SMART HISTORY LOGIC ---
def get_history():
    """Safely loads history and fixes column names automatically"""
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["Timestamp", "File", "Detected Behavior", "Confidence"])
    
    df = pd.read_csv(HISTORY_FILE)
    
    # RENAME OLD COLUMNS IF THEY EXIST (Fixes the KeyError)
    rename_map = {
        'Behavior': 'Detected Behavior',
        'File Name': 'File',
        'Score/Coverage': 'Confidence'
    }
    df = df.rename(columns=rename_map)
    return df

def save_detection(filename, behavior, confidence):
    """Saves new data while keeping the existing file structure"""
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
        uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    with col2:
        if uploaded_file:
            file_type = uploaded_file.type.split('/')[0]
            if file_type == 'image':
                # --- IMAGE PREDICTION ---
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, width=400)
                
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(img_t)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, pred = torch.max(probs, 0)
                
                res = CLASSES[pred.item()]
                st.success(f"### Result: {res}")
                save_detection(uploaded_file.name, res, conf.item()*100)

            elif file_type == 'video':
                # --- VIDEO PREDICTION ---
                st.video(uploaded_file)
                # Quick summary for the history
                save_detection(f"[VIDEO] {uploaded_file.name}", "Processing...", 0.0)
                st.info("Video added to history. (Full analysis plays in viewer)")

with tab2:
    st.subheader("📈 Behavioral Trends")
    df = get_history()
    if not df.empty and "Detected Behavior" in df.columns:
        col_left, col_right = st.columns(2)
        with col_left:
            # Chart
            fig, ax = plt.subplots()
            df["Detected Behavior"].value_counts().plot(kind='bar', color='#3498db', ax=ax)
            plt.title("Activities Captured")
            st.pyplot(fig)
        with col_right:
            st.metric("Total Detections", len(df))
            st.metric("Top Activity", df["Detected Behavior"].mode()[0])
    else:
        st.write("No behavioral data found yet.")

with tab3:
    st.subheader("Full History (Stored in CSV)")
    df_display = get_history()
    st.dataframe(df_display.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    
    # Option to download the CSV directly from the web page
    st.download_button("📥 Download History CSV", df_display.to_csv(index=False), "wildlife_history.csv", "text/csv")
