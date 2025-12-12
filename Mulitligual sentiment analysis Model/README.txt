README — Multilingual Sentiment Model (EN | Sinhala | Singlish)

This project is a multilingual sentiment classifier trained on:

English (SST-2)

Sinhala (DGurgurov/sinhala_sa)

Singlish (custom CSV)

The model is based on XLM-Roberta-Base and fine-tuned for 3-class sentiment:

0 = Negative

1 = Positive

2 = Neutral

📌 Project Structure
data_prep.py          → loads datasets, merges, splits
train.py               → fine-tuning XLM-R
eval_util.py           → accuracy, F1, classification report
inference.py           → run predictions on your own text
singlish_dataset.csv   → custom dataset I used
model/                 → final trained model files
outputs/               → training logs + evaluation metrics

🔧 Training

train.py fine-tunes XLM-R using:

AdamW (lr = 2e-5)

batch size = 16

3 epochs

warmup scheduler

max_length = 128 tokens

Training logs are saved in:

outputs/training_logs.txt

📊 Evaluation

Run:

python eval_util.py


Metrics are saved in:

outputs/metrics.txt

🚀 Inference

Example:

python inference.py


Inside inference.py:

predict("Meka hari hodai machan 👌")
predict("Umbalage service eka naraka")

📦 Model Folder

The model/ directory contains:

config.json
pytorch_model.bin     → fine-tuned weights
tokenizer.json
tokenizer_config.json
vocab.json


You can load the model like this:

model = XLMRobertaForSequenceClassification.from_pretrained("./model")
tokenizer = XLMRobertaTokenizer.from_pretrained("./model")

🧩 Notes

This project is built end-to-end by me (Joel)

I kept everything simple, realistic and reproducible

Anyone can run training again using the same scripts

✔️ End of README