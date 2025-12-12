# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import os

# ----------------------------
# 1️⃣ Define the request model
# ----------------------------
class TextRequest(BaseModel):
    text: str

# ----------------------------
# 2️⃣ Create FastAPI app
# ----------------------------
app = FastAPI(title="Multilingual Sentiment API")

# ----------------------------
# 3️⃣ Load model and tokenizer
# ----------------------------
# Use relative path since saved_model is in the same folder as main.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")

# Load tokenizer & model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# ----------------------------
# 4️⃣ Helper function for inference
# ----------------------------
def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class_id = torch.argmax(logits, dim=1).item()

    # Map numeric label to sentiment (adjust according to your labels)
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    return {"sentiment": label_map.get(pred_class_id, "Unknown")}

# ----------------------------
# 5️⃣ API endpoint
# ----------------------------
@app.post("/predict")
def predict(request: TextRequest):
    result = predict_sentiment(request.text)
    return result

# ----------------------------
# Optional test endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "Multilingual Sentiment API is running ✅"}



# main.py
# ... all your code above

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)