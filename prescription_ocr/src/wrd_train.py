"""Train OCR on Tesseract-labeled data and synthetic data.
Fixed for CTC loss compatibility.
"""
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from word_model import build_word_model
from preprocessing import preprocess

# Paths
SYNTHETIC_CSV = 'data/prescription_data.csv'
TESSERACT_CSV = 'data/tesseract_labels/prescription_labels.csv'
COMBINED_CSV = 'data/combined_training_data.csv'
CHARFILE = 'data/charList.txt'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# load char list
with open(CHARFILE, 'r', encoding='utf-8') as f:
    alphabets = f.read()

max_str_len = 25  # Reduced from 32 to prevent CTC issues
num_timestamps = 31  # Should match model output timesteps

def check_and_generate_synthetic_data():
    """Check if synthetic data exists, generate if not"""
    if not os.path.exists(SYNTHETIC_CSV):
        print("Synthetic data not found. Generating synthetic data...")
        from generate_synthetic_data import main as generate_synth
        generate_synth(n=1000)  # Reduced for testing
    
    if not os.path.exists(SYNTHETIC_CSV):
        raise FileNotFoundError(f"Synthetic data CSV not found at {SYNTHETIC_CSV}")
    
    return pd.read_csv(SYNTHETIC_CSV)

def check_and_generate_tesseract_labels():
    """Check if Tesseract labels exist, generate if not"""
    if not os.path.exists(TESSERACT_CSV):
        print("Tesseract labels not found. Generating labels...")
        from tesseract_labeling import main as generate_tesseract
        generate_tesseract()
    
    if os.path.exists(TESSERACT_CSV):
        return pd.read_csv(TESSERACT_CSV)
    else:
        print("Warning: Tesseract labeling failed or no prescription images found")
        return pd.DataFrame(columns=['image_path', 'tesseract_label', 'confidence_estimate'])

def combine_training_data():
    """Combine synthetic data and Tesseract-labeled data"""
    synthetic_data = check_and_generate_synthetic_data()
    tesseract_data = check_and_generate_tesseract_labels()
    
    # Filter Tesseract data with some confidence
    if not tesseract_data.empty:
        tesseract_data = tesseract_data[tesseract_data['confidence_estimate'] > 30]
        tesseract_data = tesseract_data.rename(columns={'tesseract_label': 'label'})
        tesseract_data = tesseract_data[['label', 'image_path']]
    
    # Combine datasets
    if not tesseract_data.empty:
        combined = pd.concat([
            synthetic_data[['label', 'image_path']],
            tesseract_data[['label', 'image_path']]
        ], ignore_index=True)
        print(f"Combined training data: {len(combined)} samples")
        print(f"  - Synthetic: {len(synthetic_data)}")
        print(f"  - Tesseract-labeled: {len(tesseract_data)}")
    else:
        combined = synthetic_data[['label', 'image_path']]
        print(f"Using only synthetic data: {len(combined)} samples")
    
    # Remove empty labels and filter long labels
    combined = combined[combined['label'].str.len() > 0]
    combined = combined[combined['label'].str.len() <= max_str_len]
    
    combined.to_csv(COMBINED_CSV, index=False)
    return combined

def load_and_preprocess_images(data):
    """Load and preprocess images with error handling"""
    X = []
    valid_indices = []
    valid_image_paths = []
    
    for idx, row in data.iterrows():
        try:
            import cv2
            img_path = row['image_path']
            
            # Check if file exists
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = preprocess(img, imgSize=(128,32))
                X.append(img.flatten())
                valid_indices.append(idx)
                valid_image_paths.append(img_path)
            else:
                print(f"Warning: Could not read image: {img_path}")
        except Exception as e:
            print(f"Error processing {row['image_path']}: {str(e)}")
            continue
    
    if len(X) == 0:
        return np.array([]), data.iloc[0:0]  # Return empty arrays
    
    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1,128,32,1)
    
    return X, data.iloc[valid_indices]

def prepare_labels(filtered_data, char_to_idx):
    """Prepare labels for training with CTC compatibility"""
    # Pad labels
    Y = np.ones((len(filtered_data), max_str_len)) * -1
    label_len = np.zeros((len(filtered_data),1), dtype=np.int32)
    
    labels_list = filtered_data['label'].tolist()
    valid_label_indices = []
    
    for i, label in enumerate(labels_list):
        nums = label_to_num(label, char_to_idx)
        if len(nums) > 0 and len(nums) <= max_str_len:
            label_len[i,0] = len(nums)
            Y[i,0:len(nums)] = nums
            valid_label_indices.append(i)
        else:
            print(f"Warning: Label too long or empty: '{label}' (length: {len(nums)})")
    
    # Filter out invalid labels
    if len(valid_label_indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return Y[valid_label_indices], label_len[valid_label_indices], valid_label_indices

def label_to_num(label, char_to_idx):
    """Convert label string to numerical representation"""
    arr = []
    for ch in str(label):
        if ch in char_to_idx:
            arr.append(char_to_idx[ch])
    return np.array(arr, dtype=np.int32)

def main():
    print("Starting OCR model training...")
    
    # Combine training data
    data = combine_training_data()
    
    if len(data) == 0:
        print("No training data available!")
        return
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    X, filtered_data = load_and_preprocess_images(data)
    
    if len(X) == 0:
        print("No valid images found after preprocessing!")
        return
    
    print(f"Successfully loaded {len(X)} images")
    
    # Prepare character mapping
    char_to_idx = {c:i for i,c in enumerate(alphabets)}
    print(f"Character set size: {len(alphabets)}")
    
    # Prepare labels
    Y, label_len, valid_indices = prepare_labels(filtered_data, char_to_idx)
    
    if len(Y) == 0:
        print("No valid labels found!")
        return
    
    # Apply valid indices to X
    X = X[valid_indices]
    
    # Input length should be <= number of timesteps (31)
    input_len = np.ones((len(X),1), dtype=np.int64) * (num_timestamps - 2)
    
    print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")
    print(f"Input lengths: {input_len.shape}, Label lengths: {label_len.shape}")
    
    # Split data
    if len(X) > 1:
        X_train, X_valid, Y_train, Y_valid, L_train, L_valid, I_train, I_valid = train_test_split(
            X, Y, label_len, input_len, test_size=0.15, random_state=42
        )
    else:
        # If only one sample, use it for both train and validation
        X_train, Y_train, L_train, I_train = X, Y, label_len, input_len
        X_valid, Y_valid, L_valid, I_valid = X, Y, label_len, input_len
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    
    if len(X_train) == 0:
        print("No training samples available after split!")
        return
    
    # Build model
    print("Building model...")
    word_model, word_model_CTC = build_word_model(
        alphabets=alphabets, max_str_len=max_str_len
    )
    
    # Use a lower learning rate for stability
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    word_model_CTC.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred}, 
        optimizer=optimizer
    )
    
    # Prepare dummy outputs (CTC loss doesn't need actual target values)
    train_output = np.zeros((len(X_train),))
    valid_output = np.zeros((len(X_valid),))
    
    print("Model compiled successfully")
    
    # Train with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, min_delta=0.01
    )
    
    # Reduce batch size for small datasets
    batch_size = min(8, len(X_train))
    
    print("Starting training...")
    try:
        history = word_model_CTC.fit(
            x=[X_train, Y_train, I_train, L_train], 
            y=train_output,
            validation_data=([X_valid, Y_valid, I_valid, L_valid], valid_output),
            epochs=30, 
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(MODEL_DIR, 'prescription_ocr.h5')
        word_model.save(model_path)
        print(f'Saved model to {model_path}')
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
            plt.close()
            print("Training history plot saved.")
        except Exception as e:
            print(f"Could not save training plot: {e}")
            
    except Exception as e:
        print(f"Training failed: {e}")
        print("Trying with simpler configuration...")
        
        # Fallback: try with even smaller batch size
        try:
            history = word_model_CTC.fit(
                x=[X_train, Y_train, I_train, L_train], 
                y=train_output,
                epochs=10, 
                batch_size=2,
                verbose=1
            )
            
            model_path = os.path.join(MODEL_DIR, 'prescription_ocr.h5')
            word_model.save(model_path)
            print(f'Saved model to {model_path}')
        except Exception as e2:
            print(f"Fallback training also failed: {e2}")

if __name__ == '__main__':
    main()