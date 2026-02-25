import os
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

st.set_page_config(page_title="Emotion Analytics", layout="wide")
st.title("😊 Real-Time Emotion Detection")

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Emotion'])

# --- RE-ENGINEERED CONNECTION SETTINGS ---
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]}
    ]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Lower frequency to prevent server timeouts
        if self.frame_count % 20 == 0:
            try:
                # Resize heavily to reduce computation time
                small_img = cv2.resize(img, (224, 224))
                results = DeepFace.analyze(small_img, actions=['emotion'], 
                                         enforce_detection=False, 
                                         detector_backend='opencv')
                self.current_emotion = results[0]['dominant_emotion']
                
                now = datetime.now().strftime("%H:%M:%S")
                new_row = pd.DataFrame({'Time': [now], 'Emotion': [self.current_emotion]})
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            except:
                pass

        cv2.putText(img, f"EMOTION: {self.current_emotion.upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- THE STREAMER ---
webrtc_streamer(
    key="emotion-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    # REMOVED async_processing to stabilize connection
)

if not st.session_state.data.empty:
    st.divider()
    counts = st.session_state.data['Emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    st.plotly_chart(px.bar(counts, x='Emotion', y='Count', color='Emotion'), use_container_width=True)
