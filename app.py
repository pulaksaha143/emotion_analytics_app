import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html

st.set_page_config(page_title="Ultra-Light Emotion AI", layout="wide")
st.title("😊 Fast Emotion Detection (Browser-Based)")

# This HTML/JS code runs the AI in your browser, NOT on the server.
# It uses the face-api.js library which is very fast.
remote_html = """
<div id="container" style="position: relative;">
    <video id="video" width="640" height="480" autoplay muted style="border-radius: 10px;"></video>
    <canvas id="overlay" style="position: absolute; top: 0; left: 0;"></canvas>
</div>
<div id="status">Loading AI Models...</div>

<script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>
<script>
    const video = document.getElementById('video');
    const status = document.getElementById('status');

    async function setup() {
        // Load models from a public CDN
        const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        
        status.innerText = "Camera Starting...";
        
        navigator.mediaDevices.getUserMedia({ video: {} })
            .then(stream => {
                video.srcObject = stream;
                status.innerText = "AI Active!";
            });
    }

    video.addEventListener('play', () => {
        const canvas = faceapi.createCanvasFromMedia(video);
        document.getElementById('container').append(canvas);
        const displaySize = { width: video.width, height: video.height };
        faceapi.matchDimensions(canvas, displaySize);

        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceExpressions();
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
            
            if(detections.length > 0) {
                const emotion = detections[0].expressions.asSortedList()[0].expression;
                window.parent.postMessage({type: 'emotion_detected', value: emotion}, "*");
            }
        }, 200);
    });

    setup();
</script>
"""

# Render the AI component
html(remote_html, height=550)

st.sidebar.info("This app uses your browser's power to run AI, making it much faster and more stable than server-side processing.")
st.caption("Developed by Pulak Saha")
