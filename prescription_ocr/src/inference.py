"""Run OCR on a prescription image using our trained model.
"""
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from preprocessing import preprocess
from segmentation import segment_lines_and_words
from word_model import build_word_model

def load_model(model_path, alphabets):
    """Load the trained OCR model"""
    word_model, _ = build_word_model(alphabets=alphabets, max_str_len=32)
    word_model.load_weights(model_path)
    return word_model

def num_to_label(num, alphabets):
    """Convert numerical predictions to text"""
    ret = ''
    for ch in num:
        if ch == -1:
            break
        ret += alphabets[ch]
    return ret

def run_ocr_on_image(image_path, model_path):
    """Run OCR on a single image and return text"""
    # Load alphabet
    with open('data/charList.txt','r',encoding='utf-8') as f:
        alphabets = f.read()
    
    # Load model
    model = load_model(model_path, alphabets)
    
    # Read and process image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Segment words
    words = segment_lines_and_words(img)
    
    pred_text = []
    for (box, wimg) in words:
        try:
            p = preprocess(wimg, imgSize=(128,32))
            p = np.array(p).reshape(-1,128,32,1).astype(np.float32)
            preds = model.predict(p, verbose=0)
            decoded = tf.keras.backend.get_value(
                tf.keras.backend.ctc_decode(
                    preds, 
                    input_length=np.ones(preds.shape[0])*preds.shape[1],
                    greedy=True
                )[0][0]
            )
            pred = num_to_label(decoded[0], alphabets=alphabets)
            if pred.strip():
                pred_text.append(pred)
        except Exception as e:
            continue
    
    return ' '.join(pred_text)

def main():
    parser = argparse.ArgumentParser(description='Run OCR on prescription image')
    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--model_path', default='model/prescription_ocr.h5', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    try:
        predicted_text = run_ocr_on_image(args.image_path, args.model_path)
        
        print('PREDICTED TEXT:')
        print(predicted_text)
        
        # Save results
        os.makedirs('results/predictions', exist_ok=True)
        base = os.path.basename(args.image_path)
        name = os.path.splitext(base)[0]
        
        with open(f'results/predictions/{name}.txt', 'w', encoding='utf-8') as f:
            f.write(predicted_text)
            
        print(f'\nSaved to results/predictions/{name}.txt')
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()