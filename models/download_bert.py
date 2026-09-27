"""
Script to download and cache pretrained BERT model and tokenizer for future fine-tuning.
No fine-tuning, training, or evaluation is performed.
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_bert():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bert")
    os.makedirs(target_dir, exist_ok=True)
    
    model_name = "bert-base-uncased"
    print(f"Downloading pretrained tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(target_dir)
    print(f"Tokenizer saved to {target_dir}")
    
    print(f"Downloading pretrained BERT model weights for {model_name}...")
    # num_labels=2 corresponds to binary sentiment classification (positive/negative) for SST-2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.save_pretrained(target_dir)
    print(f"Pretrained BERT model saved to {target_dir}")

if __name__ == "__main__":
    download_bert()
