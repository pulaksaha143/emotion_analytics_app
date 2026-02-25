import os
# Force Legacy Keras to prevent the ValueError we saw earlier
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
from fpdf import FPDF
import io

st.set_page_config(page_title="Emotion AI", layout="wide")
st.title("😊 Real-Time Emotion Detection & Analytics")

# --- Session State ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# --- Emoji Loading Logic ---
@st.cache_resource
def load_emojis():
    # Based on your repo structure (emojis in root)
    emoji_map = {
        'happy': 'happy.png', 'sad': 'sad.png', 
        'angry': 'angry.png', 'surprise': 'surprise.png',
        'neutral': 'neutral.png', 'disgust': 'disgust.png',
        'fear': 'fear.png'
    }
    imgs = {}
    for k, v in emoji_map.items():
        try:
            # Load and convert to RGBA for transparent overlay
            img = Image.open(v).convert("RGBA").resize((80, 80))
            imgs[k] = np.array(img)
        except Exception as e:
            st.warning(f"Could not load {v}: {e}")
    return imgs

emojis = load_emojis()

def overlay_emoji(frame, emotion):
    if emotion in emojis:
        emoji = emojis[emotion]
        h, w, _ = frame.shape
        # Place in top right corner
        roi = frame[10:90, w-90:w-10]
        # Standard Alpha Blending
        alpha_s = emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            roi[:, :, c] = (alpha_s * emoji[:, :, c] + alpha_l * roi[:, :, c])
        frame[10:90, w-90:w-10] = roi
    return frame

# --- Video Processor ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze every 20th frame to prevent Segmentation Fault (RAM limit)
        if self.frame_count % 20 == 0:
            try:
                small_img = cv2.resize(img, (200, 200))
                results = DeepFace.analyze(small_img, actions=['emotion'], 
                                         enforce_detection=False, 
                                         detector_backend='opencv')
                self.current_emotion = results[0]['dominant_emotion']
                
                # Log to session state
                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            except:
                pass

        # Apply Emoji Overlay
        img = overlay_emoji(img, self.current_emotion)
        
        cv2.putText(img, f"Status: {self.current_emotion.upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
webrtc_ctx = webrtc_streamer(
    key="emotion-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Reporting Logic (Triggers when Camera Stops) ---
if not webrtc_ctx.state.playing and not st.session_state.data.empty:
    st.divider()
    st.header("📋 Post-Session Emotion Report")
    
    col1, col2 = st.columns(2)
    df = st.session_state.data
    counts = df['Emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']

    with col1:
        st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion', title="Total Emotion Counts"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(counts, names='Emotion', values='Count', hole=0.3, title="Emotion Distribution"), use_container_width=True)

    # PDF Generation
    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Emotion Analytics Session Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(200, 10, txt=f"Total Samples Captured: {len(df)}", ln=True)
        pdf.ln(5)

        # List last 20 emotions
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(100, 10, "Time", border=1)
        pdf.cell(100, 10, "Emotion", border=1, ln=True)
        
        pdf.set_font("Arial", size=10)
        for i, row in df.tail(20).iterrows():
            pdf.cell(100, 10, row['Time'], border=1)
            pdf.cell(100, 10, row['Emotion'], border=1, ln=True)
        
        # Output PDF to memory buffer
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="⬇️ Download PDF Report", data=pdf_output, file_name="emotion_report.pdf", mime="application/pdf")

st.caption("Developed by Pulak Saha")
