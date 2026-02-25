import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("😊 Real-Time Emotion Detection & Analytics")

# Ensure your 'emojis' folder is in your GitHub repo!
emoji_map = {
    'happy': 'emojis/happy.png',
    'sad': 'emojis/sad.png',
    'angry': 'emojis/angry.png',
    'surprise': 'emojis/surprise.png',
    'neutral': 'emojis/neutral.png',
    'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png'
}

# Pre-load emojis to save memory
@st.cache_resource
def load_emojis():
    return {k: Image.open(v).convert("RGBA").resize((64, 64)) for k, v in emoji_map.items()}

try:
    emoji_images = load_emojis()
except:
    st.error("Emoji folder or files missing in GitHub!")
    st.stop()

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.current_emotion = "neutral"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every few frames to prevent server lag
        try:
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            self.current_emotion = results[0]['dominant_emotion']
            
            # Save to session state
            now = datetime.now().strftime("%H:%M:%S")
            new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        except:
            pass

        cv2.putText(img, f"Status: {self.current_emotion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# Start the WebRTC streamer
webrtc_streamer(key="emotion-analysis", video_transformer_factory=EmotionTransformer)

# Analytics Dashboard
if not st.session_state.data.empty:
    st.subheader("📊 Emotion Analytics Dashboard")
    # (Include your Plotly and PDF generation code here)
