"""
Script to download and store the official SST-2 dataset from GLUE benchmark.
Downloads both official splits: train and validation.
No custom splits, no modifications, no preprocessing, no model training.
"""
import os
from datasets import load_dataset

def download_sst2():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sst2")
    os.makedirs(target_dir, exist_ok=True)
    
    print("Downloading official SST-2 dataset from GLUE benchmark...")
    try:
        dataset = load_dataset("glue", "sst2")
    except Exception as e:
        print(f"Direct 'glue' reference redirected to official GLUE benchmark repo 'nyu-mll/glue' (Reason: {e})")
        dataset = load_dataset("nyu-mll/glue", "sst2")
    
    print(f"Dataset loaded: {dataset}")
    print(f"Train split size: {len(dataset['train'])}")
    print(f"Validation split size: {len(dataset['validation'])}")
    
    print(f"Saving dataset to {target_dir}...")
    dataset.save_to_disk(target_dir)
    print("SST-2 dataset successfully saved under data/sst2/")

if __name__ == "__main__":
    download_sst2()
