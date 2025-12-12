# Multilingual Sentiment Analysis (English | Sinhala | Singlish)

![1](https://raw.githubusercontent.com/JanithDoesCode/Multilingual-Sentiment-Analysis-for-Sinhala-and-English/main/Screenshots/1.png)
![5](https://raw.githubusercontent.com/JanithDoesCode/Multilingual-Sentiment-Analysis-for-Sinhala-and-English/main/Screenshots/5.png)
This project is a complete **end-to-end multilingual sentiment analysis system** built using Transformer-based Natural Language Processing.

The model supports:
- English
- Sinhala
- Singlish

It is fine-tuned using **XLM-Roberta-Base** for **3-class sentiment classification**.

Sentiment Labels:
- 0 → Negative
- 1 → Positive
- 2 → Neutral

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|------------|
| data_prep.py | Dataset loading, cleaning, merging |
| train.py | Model training and fine-tuning |
| eval_util.py | Accuracy, Precision, Recall, F1 |
| inference.py | Prediction script |
| app.py | Gradio UI |
| singlish_dataset.csv | Custom Singlish dataset |
| model/ | Trained model files |
| outputs/ | Logs and metrics |
| Screenshots/ | UI screenshots |

---

## 🔧 Training Details

Model: XLM-Roberta-Base  
Optimizer: AdamW  
Learning Rate: 2e-5  
Batch Size: 16  
Epochs: 3  
Max Token Length: 128  
Scheduler: Warmup + Linear Decay  

Training logs are saved in:
outputs/training_logs.txt

---

## 📊 Model Performance (Actual Results)

Evaluation file:
outputs/metrics.txt

| Dataset | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| English | 91% | 0.90 | 0.91 | 0.90 |
| Sinhala | 88% | 0.87 | 0.88 | 0.87 |
| Singlish | 85% | 0.84 | 0.85 | 0.84 |

---

## 🚀 Inference

### Using Python Script

```python
from inference import predict

predict("Meka hari hodai machan 👌")
predict("Umbalage service eka naraka")


```

## Gradio UI

Run the application:
```
python app.py
```
Steps:

1. Open the local URL shown in terminal (e.g. http://127.0.0.1:7860)


2. Enter text in English, Sinhala, or Singlish


3. Click Submit


4. Sentiment is predicted instantly




---

## Screenshots (Gradio UI)

![2](https://raw.githubusercontent.com/JanithDoesCode/Multilingual-Sentiment-Analysis-for-Sinhala-and-English/main/Screenshots/2.png)
![3](https://raw.githubusercontent.com/JanithDoesCode/Multilingual-Sentiment-Analysis-for-Sinhala-and-English/main/Screenshots/3.png)
![4](https://raw.githubusercontent.com/JanithDoesCode/Multilingual-Sentiment-Analysis-for-Sinhala-and-English/main/Screenshots/4.png)

## Model Files

The model/ directory contains:

- config.json

- model.safetensors

- tokenizer_config.json

- sentencepiece.bpe.model

- special_tokens_map.json


## Load the model:
```
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

model = XLMRobertaForSequenceClassification.from_pretrained("./model")
tokenizer = XLMRobertaTokenizer.from_pretrained("./model")
```

---

## What I Learned

- Multilingual NLP processing and working with modern NLP pipelines

- Fine-tuning Transformer models

- Handling low-resource languages

- Model evaluation using F1, Precision, Recall

- Gradio UI deployment

- Structuring production-ready ML projects



---

👤 Author(This project is built end-to-end by me)

Janith Perera
