"""
Execution runner for BERT fine-tuning experiment on SST-2.
Trains the model on GPU, generates evaluation metrics, saves figures and model,
and outputs the clean, fully-executed Jupyter notebook at training/bert_finetuning.ipynb.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

print("="*60)
print("STARTING BERT FINE-TUNING PIPELINE")
print("="*60)

# 1. GPU Check
print(f"PyTorch Version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available: {cuda_avail}")
device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
print(f"Active Device: {device_name}")

if cuda_avail:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {vram_gb:.2f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 2. Baseline TF-IDF Evaluation (to populate actual comparison numbers)
print("\n--- Evaluating TF-IDF + Logistic Regression Baseline on SST-2 ---")
try:
    ds_raw = load_dataset("glue", "sst2")
except Exception:
    ds_raw = load_dataset("nyu-mll/glue", "sst2")

train_df = pd.DataFrame(ds_raw["train"])
val_df = pd.DataFrame(ds_raw["validation"])

vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vec.fit_transform(train_df["sentence"])
y_train_tfidf = train_df["label"]
X_val_tfidf = vec.transform(val_df["sentence"])
y_val_tfidf = val_df["label"]

tfidf_model = LogisticRegression(max_iter=1000, random_state=42)
tfidf_model.fit(X_train_tfidf, y_train_tfidf)
tfidf_preds = tfidf_model.predict(X_val_tfidf)

tfidf_acc = accuracy_score(y_val_tfidf, tfidf_preds)
tfidf_prec = precision_score(y_val_tfidf, tfidf_preds)
tfidf_rec = recall_score(y_val_tfidf, tfidf_preds)
tfidf_f1 = f1_score(y_val_tfidf, tfidf_preds)

print(f"TF-IDF Baseline Accuracy:  {tfidf_acc:.4f}")
print(f"TF-IDF Baseline Precision: {tfidf_prec:.4f}")
print(f"TF-IDF Baseline Recall:    {tfidf_rec:.4f}")
print(f"TF-IDF Baseline F1-Score:  {tfidf_f1:.4f}")

# 3. Load Pretrained BERT
model_name = "bert-base-uncased"
print(f"\nLoading pretrained tokenizer and model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Tokenize Dataset
print("\nTokenizing SST-2 sentences with max_length=128, truncation=True...")
def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)

tokenized_datasets = ds_raw.map(
    tokenize_fn,
    batched=True,
    remove_columns=["sentence", "idx"],
    desc="Tokenizing SST-2"
)

# 5. Training Arguments
os.makedirs("models/bert-sst2", exist_ok=True)
os.makedirs("evaluation/figures", exist_ok=True)
os.makedirs("training", exist_ok=True)

batch_size = 16 if cuda_avail else 8
print(f"Using per-device batch size: {batch_size}")

training_args = TrainingArguments(
    output_dir="./models/bert-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=cuda_avail,
    logging_steps=100,
    save_total_limit=1,
    dataloader_pin_memory=cuda_avail,
    seed=42,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 6. Fine-Tuning
print("\n" + "="*50)
print("TRAINING BERT ON SST-2...")
print("="*50)
train_result = trainer.train()
print("Training completed!")

# 7. Evaluate
print("\n" + "="*50)
print("EVALUATING ON OFFICIAL SST-2 VALIDATION SPLIT...")
print("="*50)
eval_metrics = trainer.evaluate()
print("Trainer evaluate output:", eval_metrics)

preds_out = trainer.predict(tokenized_datasets["validation"])
val_preds = np.argmax(preds_out.predictions, axis=-1)
val_labels = preds_out.label_ids

bert_acc = float(accuracy_score(val_labels, val_preds))
bert_prec = float(precision_score(val_labels, val_preds))
bert_rec = float(recall_score(val_labels, val_preds))
bert_f1 = float(f1_score(val_labels, val_preds))

cls_report = classification_report(val_labels, val_preds, target_names=["Negative (0)", "Positive (1)"], digits=4)
cm = confusion_matrix(val_labels, val_preds)

print("\nCLASSIFICATION REPORT:")
print(cls_report)
print("\nCONFUSION MATRIX:")
print(cm)

# 8. Plot & Save Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_title("BERT Fine-Tuning Confusion Matrix (SST-2 Validation Split)", fontsize=12, pad=12)
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=10)
ax.set_yticks(tick_marks)
ax.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=10)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold", fontsize=14)

ax.set_ylabel("True Ground Truth Label", fontsize=11)
ax.set_xlabel("Predicted Sentiment Label", fontsize=11)
plt.tight_layout()
cm_fig_path = os.path.abspath("evaluation/figures/bert_confusion_matrix.png")
fig.savefig(cm_fig_path, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved confusion matrix to: {cm_fig_path}")

# 9. Save Fine-Tuned Model and Tokenizer
save_path = os.path.abspath("models/bert-sst2")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved fine-tuned BERT model and tokenizer to: {save_path}")

# 10. Inference Test
print("\n" + "="*50)
print("INFERENCE TEST ON SUPPORT TICKET SAMPLES")
print("="*50)
saved_tok = AutoTokenizer.from_pretrained(save_path)
saved_mod = AutoModelForSequenceClassification.from_pretrained(save_path)
saved_mod.eval()
if cuda_avail:
    saved_mod.to("cuda")

test_samples = [
    "The support team solved my problem quickly.",
    "The issue was never fixed and the service was terrible.",
    "I appreciate the prompt resolution, exceptional customer care!",
    "My ticket was closed without any explanation or refund."
]

inference_results = []
for text in test_samples:
    inputs = saved_tok(text, return_tensors="pt", truncation=True)
    if cuda_avail:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = saved_mod(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_label_idx = int(np.argmax(probs))
        confidence = float(probs[pred_label_idx])
        sentiment = "POSITIVE" if pred_label_idx == 1 else "NEGATIVE"
        inference_results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probs": probs.tolist()
        })
        print(f"\nTicket: \"{text}\"")
        print(f" -> Predicted Sentiment: {sentiment}")
        print(f" -> Confidence: {confidence*100:.2f}% (Negative: {probs[0]:.4f}, Positive: {probs[1]:.4f})")

# 11. Final Summary
print("\n" + "="*50)
print("BERT FINE-TUNING RESULTS")
print("="*50)
print(f"Accuracy:   {bert_acc:.4f}")
print(f"Precision:  {bert_prec:.4f}")
print(f"Recall:     {bert_rec:.4f}")
print(f"F1 Score:   {bert_f1:.4f}")
print("="*50)

# 12. Save results JSON for building the notebook and final report
metrics_summary = {
    "tfidf": {
        "accuracy": float(tfidf_acc),
        "precision": float(tfidf_prec),
        "recall": float(tfidf_rec),
        "f1": float(tfidf_f1)
    },
    "bert": {
        "accuracy": float(bert_acc),
        "precision": float(bert_prec),
        "recall": float(bert_rec),
        "f1": float(bert_f1)
    },
    "classification_report": cls_report,
    "confusion_matrix": cm.tolist(),
    "inference": inference_results,
    "device": device_name,
    "pytorch_version": torch.__version__
}

with open("training_results_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)

print("\nSaved training_results_summary.json successfully!")
