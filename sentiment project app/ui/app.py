# app.py
import gradio as gr
import requests

# ----------------------------
# 1️⃣ FastAPI endpoint
# ----------------------------
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# ----------------------------
# 2️⃣ Function to send text to API
# ----------------------------
def get_sentiment(text):
    try:
        response = requests.post(FASTAPI_URL, json={"text": text})
        if response.status_code == 200:
            result = response.json()
            return result.get("sentiment", "Error")
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Connection Error: {str(e)}"

# ----------------------------
# 3️⃣ Gradio interface
# ----------------------------
ui = gr.Interface(
    fn=get_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence here..."),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Multilingual Sentiment Analyzer",
    description="Type any sentence and get sentiment prediction from your FastAPI model."
)

# ----------------------------
# 4️⃣ Launch UI
# ----------------------------
ui.launch()