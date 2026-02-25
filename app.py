import os
import sys

# 1. CRITICAL: MUST BE AT THE VERY TOP TO BYPASS KERAS 3 ERRORS
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Emotion Analytics Dashboard", layout="wide")
st.title("😊 Real-Time Emotion Detection & Analytics")

# --- Session State ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# --- WebRTC Configuration (Fixes the Connection Error) ---
# We use multiple Google STUN servers to ensure the connection bypasses firewalls.
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]}
    ]}
)

# --- Emoji Loading ---
@st.cache_resource
def load_emojis():
    emoji_map = {
        'happy': 'emojis/happy.png', 'sad': 'emojis/sad.png', 
        'angry': 'emojis/angry.png', 'surprise': 'emojis/surprise.png',
        'neutral': 'emojis/neutral.png', 'disgust': 'emojis/disgust.png',
        'fear': 'emojis/fear.png'
    }
    imgs = {}
    for k, v in emoji_map.items():
        try:
            imgs[k] = Image.open(v).convert("RGBA").resize((64, 64))
        except: continue
    return imgs

emojis_dict = load_emojis()

# --- Video Processing Logic ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every 15th frame to keep RAM usage under 1GB
        if self.frame_count % 15 == 0:
            try:
                # Resize frame for faster processing
                small_img = cv2.resize(img, (300, 300))
                # detector_backend='opencv' is the only one that bypasses RetinaFace check errors
                results = DeepFace.analyze(
                    small_img, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv'
                )
                self.current_emotion = results[0]['dominant_emotion']
                
                # Log data to session state
                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            except:
                pass

        # Visual Overlay on video stream
        cv2.putText(img, f"Emotion: {self.current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("Live Stream")
    webrtc_streamer(
        key="emotion-analysis",
        mode=WebRtcMode.SENDRECV, # Explicit mode prevents NoneType errors
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            },
            "audio": False
        },
        async_processing=True,
    )

with col_info:
    st.subheader("How to use")
    st.markdown("""
    1. Click **Start** on the video player.
    2. Grant permission to your camera.
    3. The AI will analyze your face every few seconds.
    4. Charts below will update automatically.
    """)
    if not st.session_state.data.empty:
        dominant = st.session_state.data['Emotion'].mode()[0]
        st.metric("Dominant Emotion", dominant.upper())

# --- Analytics Dashboard ---
if not st.session_state.data.empty:
    st.divider()
    st.header("📊 Emotion Dashboard")
    
    c1, c2 = st.columns(2)
    counts = st.session_state.data['Emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    
    with c1:
        st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion', template="plotly_dark"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(counts, names='Emotion', values='Count', hole=0.4), use_container_width=True)

    # Export Section
    st.subheader("Export Data")
    csv = st.session_state.data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Session CSV", csv, "emotion_history.csv", "text/csv")

st.caption("Developed by Pulak Saha")
