import os
import sys

# 1. FORCE LEGACY KERAS - MUST BE FIRST
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
st.set_page_config(page_title="Emotion Analytics", layout="wide")
st.title("😊 Real-Time Emotion Detection")

# Initialize data storage
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# --- WebRTC Settings ---
# These STUN servers help bypass firewalls
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Only analyze every 25th frame (roughly once every 2 seconds)
        # This is the ONLY way to stay under the 1GB RAM limit
        if self.frame_count % 25 == 0:
            try:
                # Resize to a very small image for the AI to process quickly
                small_img = cv2.resize(img, (224, 224))
                
                # Using 'opencv' backend because it is the lightest
                results = DeepFace.analyze(
                    small_img, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv'
                )
                self.current_emotion = results[0]['dominant_emotion']
                
                # Update Session Data
                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            except Exception as e:
                print(f"AI Error: {e}")

        # Draw the emotion on the video
        cv2.putText(img, f"Emotion: {self.current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Sidebar/Instructions ---
st.sidebar.header("How it works")
st.sidebar.write("1. Start the camera.")
st.sidebar.write("2. The AI processes one frame every 2 seconds to save memory.")
st.sidebar.write("3. View charts below.")

# --- The Streamer ---
webrtc_streamer(
    key="emotion-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Analytics ---
if not st.session_state.data.empty:
    st.divider()
    st.subheader("📊 Live Analytics")
    df = st.session_state.data
    counts = df['Emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion'), use_container_width=True)
    with col2:
        st.plotly_chart(px.line(df, x='Time', y='Emotion', title="Emotion Timeline"), use_container_width=True)

st.caption("Developed by Pulak Saha")
