import os
from word_model import build_word_model
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint

# Paths
real_data_dir = "data/prescriptions"  # your Kaggle images + optional labels
model_save_path = "model/prescription_ocr_finetuned.h5"

# Parameters
alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
max_str_len = 32
batch_size = 16
epochs = 5  # can increase if needed

# Build model
word_model, _ = build_word_model(alphabets=alphabets, max_str_len=max_str_len)

# Data generator
train_gen = DataGenerator(real_data_dir, batch_size=batch_size, img_w=256, img_h=64)

# Checkpoint
checkpoint = ModelCheckpoint(model_save_path, monitor='loss', save_best_only=True, verbose=1)

# Fine-tune
word_model.fit(train_gen, epochs=epochs, callbacks=[checkpoint])
print(f"Fine-tuned model saved to {model_save_path}")
