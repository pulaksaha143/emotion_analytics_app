import os
import sys

# 1. PREVENT SEGMENTATION FAULT & KERAS ERRORS
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import plotly.express as px

st.set_page_config(page_title="Emotion AI", layout="wide")
st.title("😊 Emotion Analytics")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Analyze only once every 30 frames (about 3 seconds) 
        # to prevent the Segmentation Fault/Crash
        if self.frame_count % 30 == 0:
            try:
                # Resize to very small 150x150 for stability
                small_img = cv2.resize(img, (150, 150))
                results = DeepFace.analyze(
                    small_img, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv'
                )
                self.current_emotion = results[0]['dominant_emotion']
                
                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            except:
                pass

        cv2.putText(img, f"EMOTION: {self.current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotion-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if not st.session_state.data.empty:
    st.divider()
    df = st.session_state.data
    st.plotly_chart(px.bar(df['Emotion'].value_counts()), use_container_width=True)

st.caption("Developed by Pulak Saha")
