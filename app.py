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

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# Load Emojis safely
emoji_map = {
    'happy': 'emojis/happy.png', 'sad': 'emojis/sad.png', 
    'angry': 'emojis/angry.png', 'surprise': 'emojis/surprise.png',
    'neutral': 'emojis/neutral.png', 'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png'
}

@st.cache_resource
def load_emoji_images():
    images = {}
    for k, v in emoji_map.items():
        try:
            images[k] = Image.open(v).convert("RGBA").resize((64, 64))
        except FileNotFoundError:
            continue
    return images

emoji_images = load_emoji_images()

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every few frames to save CPU memory
        try:
            # Using 'opencv' backend as it is the most lightweight for free tier
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            self.current_emotion = results[0]['dominant_emotion']
            
            # Store data (Streamlit allows session_state access from within the processor)
            now = datetime.now().strftime("%H:%M:%S")
            new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        except Exception:
            pass

        # Visual feedback on the stream
        cv2.putText(img, f"Emotion: {self.current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC Streamer
webrtc_streamer(key="emotion-detection", video_processor_factory=EmotionProcessor)

# --- Analytics Section ---
if not st.session_state.data.empty:
    st.divider()
    st.subheader("📊 Emotion Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        counts = st.session_state.data['Emotion'].value_counts().reset_index()
        counts.columns = ['Emotion', 'Count']
        st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion'), use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(counts, names='Emotion', values='Count'), use_container_width=True)

    # Download Buttons
    csv = st.session_state.data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "emotions.csv", "text/csv")
    
    if st.button("📝 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Emotion Detection Report", ln=True, align='C')
        for i, row in st.session_state.data.tail(20).iterrows(): # Last 20 rows
            pdf.cell(200, 10, txt=f"{row['Time']}: {row['Emotion']}", ln=True)
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button("⬇️ Download PDF", pdf_output, "report.pdf", "application/pdf")

st.caption("Developed by Pulak Saha")
