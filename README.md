# 😊 Real-Time Emotion Detection & Analytics App

An interactive Streamlit app that:
- 📸 Detects your emotion live from webcam video
- 📊 Shows a real-time analytics dashboard: bar chart, pie chart, timeline
- 😎 Adds a fun emoji overlay matching the detected emotion
- 📝 Lets you download a CSV of all detected emotions and a PDF session report
- ⚡ Optimized for higher FPS by analyzing every 5th frame

Built with:
- Python
- Streamlit
- OpenCV
- DeepFace
- Pandas, Plotly
- Pillow & fpdf2

---

## 🚀 Features

✅ Real-time emotion detection from webcam  
✅ Live analytics dashboard with dynamic charts  
✅ Emoji overlay matching current emotion  
✅ Download CSV & PDF report of session  
✅ Optimized for smooth video (even on CPU)  
✅ Completely free & deployable on Streamlit Cloud

---

## 📦 Project Structure
emotion_analytics_app/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md # Project info
└── emojis/ # Emoji PNGs (happy.png, sad.png, etc.)

---

## 🧑‍💻 Run Locally

1️⃣ Clone this repository:
```bash
git clone https://github.com/pulaksaha143/emotion_analytics_app.git
cd emotion_analytics_app

pip install -r requirements.txt

streamlit run app.py
