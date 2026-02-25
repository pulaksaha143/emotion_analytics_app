import os
# FORCE LEGACY KERAS BEFORE ANY OTHER IMPORTS
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Emotion Analytics", layout="wide")
st.title("😊 Real-Time Emotion Detection & Analytics")

# --- Session State ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# --- Emoji Handling ---
emoji_map = {
    'happy': 'emojis/happy.png', 'sad': 'emojis/sad.png', 
    'angry': 'emojis/angry.png', 'surprise': 'emojis/surprise.png',
    'neutral': 'emojis/neutral.png', 'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png'
}

@st.cache_resource
def load_emojis():
    imgs = {}
    for k, v in emoji_map.items():
        try:
            imgs[k] = Image.open(v).convert("RGBA").resize((64, 64))
        except: continue
    return imgs

emojis_dict = load_emojis()

# --- WebRTC Video Processor ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Use 'opencv' for speed on the free tier
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            self.current_emotion = results[0]['dominant_emotion']
            
            # Log data to session state
            now = datetime.now().strftime("%H:%M:%S")
            new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        except:
            pass

        # Overlay text on stream
        cv2.putText(img, f"Emotion: {self.current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Setup ---
webrtc_streamer(
    key="emotion-stream", 
    video_processor_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Dashboard ---
if not st.session_state.data.empty:
    st.divider()
    st.subheader("📊 Session Analytics")
    col1, col2 = st.columns(2)
    
    counts = st.session_state.data['Emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    
    with col1:
        st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion'), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(counts, names='Emotion', values='Count'), use_container_width=True)

    # Download
    csv = st.session_state.data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export Data (CSV)", csv, "session_emotions.csv", "text/csv")

st.caption("Developed by Pulak Saha")
