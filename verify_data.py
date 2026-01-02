import numpy as np
import os
from src.config import DATA_DIR, PROCESSED_DATA_FILE

def verify_dataset():
    file_path = os.path.join(DATA_DIR, PROCESSED_DATA_FILE)
    if not os.path.exists(file_path):
        print(f"FAILED: {file_path} does not exist.")
        return

    try:
        data = np.load(file_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"FAILED: Error loading dataset: {e}")
        return

    required_keys = ['train_X', 'train_y', 'val_X', 'val_y']
    for key in required_keys:
        if key not in data:
            print(f"FAILED: Missing key {key}")
            return
        
    train_X = data['train_X']
    train_y = data['train_y']
    val_X = data['val_X']
    val_y = data['val_y']

    print(f"Train X shape: {train_X.shape}")
    print(f"Train y shape: {train_y.shape}")
    print(f"Val X shape: {val_X.shape}")
    print(f"Val y shape: {val_y.shape}")

    # Check shapes
    if len(train_X.shape) != 4 or train_X.shape[1:] != (8, 8, 13):
        print(f"FAILED: Train X shape invalid. Expected (N, 8, 8, 13), got {train_X.shape}")
    
    # Check label range
    if np.max(train_y) > 1.0 or np.min(train_y) < -1.0:
        print(f"FAILED: Labels out of range [-1, 1]. Max: {np.max(train_y)}, Min: {np.min(train_y)}")

    # Check for NaN
    if np.isnan(train_X).any() or np.isnan(train_y).any():
        print("FAILED: NaNs found in training data.")

    print("VERIFICATION PASSED: Dataset structure and values look correct.")

if __name__ == "__main__":
    verify_dataset()
