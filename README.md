# README — Multilingual Sentiment Model (EN | Sinhala | Singlish)

This project is a **multilingual sentiment classifier** trained on:

- **English** (SST-2)  
- **Sinhala** (DGurgurov/sinhala_sa)  
- **Singlish** (custom CSV)  

The model is based on **XLM-Roberta-Base** and fine-tuned for **3-class sentiment**:

- `0` = Negative  
- `1` = Positive  
- `2` = Neutral  

---

## 📌 Project Structure

| File/Folder             | Description                                         |
|-------------------------|-----------------------------------------------------|
| `data_prep.py`          | Loads datasets, merges, splits                      |
| `train.py`              | Fine-tuning XLM-R                                   |
| `eval_util.py`          | Accuracy, F1, classification report                |
| `inference.py`          | Run predictions on your own text                   |
| `app.py`                | Gradio UI interface for interacting with the model |
| `singlish_dataset.csv`  | Custom dataset                                     |
| `model/`                | Final trained model files                           |
| `outputs/`              | Training logs + evaluation metrics                 |

---

## 🔧 Training

`train.py` fine-tunes XLM-R using:

- Optimizer: **AdamW** (`lr = 2e-5`)  
- **Batch size:** 16  
- **Epochs:** 3  
- **Warmup scheduler**  
- **Max length:** 128 tokens  

Training logs are saved in:  
`outputs/training_logs.txt`

---

## 📊 Evaluation / Metrics

Run:

```bash
python eval_util.py

Metrics are saved in:
outputs/metrics.txt

Model performance (example results):

Dataset	Accuracy	F1 Score	Precision	Recall

English	91%	0.90	0.91	0.90
Sinhala	88%	0.87	0.88	0.87
Singlish	85%	0.84	0.85	0.84


(Replace these numbers with your actual results.)


---

🚀 Inference

Using Python script

Example:

python inference.py

Inside inference.py:

predict("Meka hari hodai machan 👌")
predict("Umbalage service eka naraka")

Using Gradio UI

Run the Gradio interface:

python app.py

A local URL will appear (e.g., http://127.0.0.1:7860)

Enter text in English, Sinhala, or Singlish

Click Submit

The model predicts the sentiment instantly



---

📷 Screenshots

Here are some screenshots from the Gradio UI:
## UI Screenshots

 
![Screenshot 1](screenshots/1.png)
 
![Screenshot 2](screenshots/2.png)


![Screenshot 3](screenshots/3.png)


![Screenshot 4](screenshots/4.png)


![Screenshot 5](screenshots/5.png)
---

📦 Model Folder

The model/ directory contains:

config.json

pytorch_model.bin → fine-tuned weights

tokenizer.json

tokenizer_config.json

vocab.json


You can load the model like this:

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

model = XLMRobertaForSequenceClassification.from_pretrained("./model")
tokenizer = XLMRobertaTokenizer.from_pretrained("./model")


---

🧩 Notes

This project is built end-to-end by me (Joel)

Kept everything simple, realistic, and reproducible

Anyone can run training again using the same scripts

Gradio UI provides an easy way to test the model without coding

Screenshots + YouTube video showcase the project for portfolio purposes


✔️ End of README
