import os
import json
import hashlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import time
from PIL import Image

# 1. PAGE SETUP
st.set_page_config(page_title="Wildlife AI Command Center", layout="wide")

# Ensure required libraries are present for the Cloud
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ModuleNotFoundError:
    st.error("Missing dependencies. Please ensure 'requirements.txt' is in your GitHub repo.")
    st.stop()

# --- 2. SETTINGS & PATHS (Optimized for Cloud & Local) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the model file automatically
if os.path.exists("wildlife_behavior_model.pth"):
    MODEL_PATH = "wildlife_behavior_model.pth"
elif os.path.exists(os.path.join(BASE_DIR, "animals pics", "wildlife_behavior_model.pth")):
    MODEL_PATH = os.path.join(BASE_DIR, "animals pics", "wildlife_behavior_model.pth")
else:
    MODEL_PATH = "wildlife_behavior_model.pth"

HISTORY_FILE = "detection_history.csv"
USERS_FILE = os.path.join(BASE_DIR, "users.json")
CLASSES = ['Eating', 'Hunting', 'Resting', 'Sleeping', 'Walking']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 3. AUTHENTICATION HELPERS (compact) ---
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120000).hex()
    return salt, h


def verify_password(password, salt, expected_hash):
    _, h = hash_password(password, salt)
    return h == expected_hash


def set_authenticated_user(user):
    st.session_state.authenticated = True
    st.session_state.current_user = {"full_name": user.get("full_name", ""), "email": user.get("email", ""), "role": user.get("role", "")}


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None


# Safe rerun helper compatible with multiple Streamlit versions
def safe_rerun():
    try:
        # Preferred if available
        st.experimental_rerun()
    except Exception:
        # Fallback: changing query params triggers a rerun in most Streamlit versions
        try:
            st.experimental_set_query_params(_rerun=str(int(time.time())))
        except Exception:
            # Last resort: raise a Streamlit exception to request a rerun if available
            try:
                from streamlit.runtime.scriptrunner.script_runner import RerunException

                raise RerunException()
            except Exception:
                # If none of the above work, just return and rely on state changes
                return


def render_auth_page():
    # center the form and keep it compact (half-width)
    c1, c2, c3 = st.columns([1, 2, 1])
    users = load_users()
    with c2:
        st.title("🐾 Wildlife AI Access")
        st.caption("Create account or log in to use the AI tools")

        tab_login, tab_register = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form(key="login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Login")
            if submit:
                key = email.strip().lower()
                user = users.get(key)
                if not user:
                    st.error("No account for that email")
                elif not verify_password(password, user.get("salt"), user.get("password_hash")):
                    st.error("Incorrect password")
                else:
                    set_authenticated_user(user)
                    st.success(f"Welcome back, {user.get('full_name','')}")
                    safe_rerun()

        with tab_register:
            with st.form(key="register_form"):
                full_name = st.text_input("Full name", key="reg_fullname")
                email = st.text_input("Email", key="reg_email")
                password = st.text_input("Password", type="password", key="reg_password")
                confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                role = st.selectbox("Who are you?", ["Student", "Wildlife Conservationist", "Ecological Researcher"], key="reg_role")
                submit = st.form_submit_button("Create account")
            if submit:
                k = email.strip().lower()
                if not full_name or not k or not password:
                    st.error("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif k in users:
                    st.error("Account already exists for that email")
                else:
                    salt, ph = hash_password(password)
                    users[k] = {"full_name": full_name, "email": k, "password_hash": ph, "salt": salt, "role": role, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    save_users(users)
                    set_authenticated_user(users[k])
                    st.success("Account created")
                    safe_rerun()


# Auth gate
if not st.session_state.authenticated:
    render_auth_page()
    st.stop()


# --- 4. AI MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found. Ensure 'wildlife_behavior_model.pth' is uploaded to GitHub.")
        st.stop()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model


model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- 5. SMART HISTORY LOGIC ---
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["Timestamp", "File", "Detected Behavior", "Confidence"])
    df = pd.read_csv(HISTORY_FILE)
    rename_map = {'Behavior': 'Detected Behavior', 'File Name': 'File', 'Score/Coverage': 'Confidence'}
    return df.rename(columns=rename_map)


def save_detection(filename, behavior, confidence):
    df = get_history()
    new_row = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File": filename, "Detected Behavior": behavior, "Confidence": f"{confidence:.2f}%"}])
    df = pd.concat([df, new_row], ignore_index=True)
    try:
        df.to_csv(HISTORY_FILE, index=False)
    except PermissionError:
        st.error("❌ Permission Denied: Close the CSV file if it's open in another program.")


# --- 6. STREAMLIT UI ---
st.title("🐾 Wildlife Behavior AI Command Center")

# Sidebar with logout (unique key)
with st.sidebar:
    st.subheader("Account")
    if st.session_state.current_user:
        st.write(f"**Name:** {st.session_state.current_user.get('full_name','')}")
        st.write(f"**Email:** {st.session_state.current_user.get('email','')}")
        st.write(f"**Role:** {st.session_state.current_user.get('role','')}")
        st.divider()
        if st.button("Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            safe_rerun()


tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics ", "📜 History Log"])

# --- TAB 1: DETECTION ---
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Media")
        uploaded_file = st.file_uploader("Upload ONE Image or Video", type=["jpg", "png", "jpeg", "jfif", "mp4"], accept_multiple_files=False)

    with col2:
        if uploaded_file:
            file_type = uploaded_file.type.split('/')[0]
            if file_type == 'image':
                try:
                    img = Image.open(uploaded_file).convert('RGB')
                    st.image(img, width=350, caption=f"Processing: {uploaded_file.name}")
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(img_t)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        conf, pred = torch.max(probs, 0)
                    res = CLASSES[pred.item()]
                    conf_pct = conf.item() * 100
                    st.success(f"{res}: {conf_pct:.2f}%")
                    save_detection(uploaded_file.name, res, conf_pct)
                except Exception as e:
                    st.error(f"Error: {e}")
            elif file_type == 'video':
                st.video(uploaded_file)
                save_detection(f"[VIDEO] {uploaded_file.name}", "Video Logged", 0.0)
                st.info("Video behavior summary added to history.")


# --- TAB 2: ANALYTICS (SMALLER PLOT FOR PPT) ---
with tab2:
    st.subheader("📈 Behavioral Trends for Presentation")
    df_hist = get_history()
    if not df_hist.empty and "Detected Behavior" in df_hist.columns:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df_hist["Detected Behavior"].value_counts()
            counts.plot(kind='bar', color='#3498db', ax=ax, edgecolor='black', linewidth=0.8)
            plt.title("Wildlife Activities Captured", fontsize=12, fontweight='bold')
            plt.ylabel("Number of Detections", fontsize=10)
            plt.xlabel("Detected Behavior", fontsize=10)
            plt.xticks(rotation=45, fontsize=9)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Detections", len(df_hist))
        m2.metric("Most Frequent", df_hist["Detected Behavior"].mode()[0])
        m3.metric("System Status", "Online / Cloud")
    else:
        st.info("No data found yet. Detect some behaviors first!")


# --- TAB 3: HISTORY ---
with tab3:
    st.subheader("📜 Complete Activity Log")
    df_display = get_history()
    if not df_display.empty:
        st.dataframe(df_display.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        st.download_button("📥 Download Excel/CSV Report", df_display.to_csv(index=False), "wildlife_report.csv", "text/csv")
    else:
        st.write("History is empty.")
