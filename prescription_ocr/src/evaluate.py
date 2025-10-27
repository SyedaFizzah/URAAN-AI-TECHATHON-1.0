"""Evaluate the OCR model accuracy on test images.
Compares model predictions with Tesseract results and calculates metrics.
"""
import os
import argparse
import pandas as pd
import numpy as np
from jiwer import wer, cer
import cv2
from inference import run_ocr_on_image
from tesseract_labeling import extract_text_with_tesseract

def calculate_accuracy_metrics(true_texts, pred_texts):
    """Calculate Word Error Rate (WER) and Character Error Rate (CER)"""
    wer_score = wer(true_texts, pred_texts)
    cer_score = cer(true_texts, pred_texts)
    
    # Simple exact match accuracy
    exact_matches = sum(1 for true, pred in zip(true_texts, pred_texts) 
                       if true.lower() == pred.lower())
    exact_accuracy = exact_matches / len(true_texts) if true_texts else 0
    
    return {
        'word_error_rate': wer_score,
        'character_error_rate': cer_score,
        'exact_match_accuracy': exact_accuracy,
        'accuracy_percentage': (1 - wer_score) * 100
    }

def evaluate_model(test_images, model_path):
    """Evaluate model performance on test images"""
    model_results = []
    tesseract_results = []
    
    for img_path in test_images:
        # Model prediction
        model_text = run_ocr_on_image(img_path, model_path)
        model_results.append(model_text)
        
        # Tesseract prediction
        tesseract_text = extract_text_with_tesseract(img_path)
        tesseract_results.append(tesseract_text)
        
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  Model: {model_text}")
        print(f"  Tesseract: {tesseract_text}")
        print()
    
    # Since we don't have ground truth, compare with Tesseract as baseline
    # In real scenario, you'd have human-validated labels here
    metrics = calculate_accuracy_metrics(tesseract_results, model_results)
    
    return metrics, model_results, tesseract_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='data/prescriptions')
    parser.add_argument('--model_path', default='model/prescription_ocr.h5')
    parser.add_argument('--num_test', type=int, default=50)
    args = parser.parse_args()
    
    # Get test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = []
    
    for file in os.listdir(args.test_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            all_images.append(os.path.join(args.test_dir, file))
    
    # Use subset for testing
    test_images = all_images[:min(args.num_test, len(all_images))]
    
    print(f"Evaluating on {len(test_images)} images...")
    
    # Evaluate
    metrics, model_preds, tesseract_preds = evaluate_model(test_images, args.model_path)
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        'image_path': test_images,
        'model_prediction': model_preds,
        'tesseract_prediction': tesseract_preds
    })
    results_df.to_csv('results/evaluation/detailed_predictions.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('results/evaluation/accuracy_metrics.csv', index=False)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Word Error Rate: {metrics['word_error_rate']:.4f}")
    print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"Accuracy Percentage: {metrics['accuracy_percentage']:.2f}%")
    print("\nResults saved to results/evaluation/")

if __name__ == '__main__':
    main()