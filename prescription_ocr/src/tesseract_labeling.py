"""Use Tesseract OCR to generate initial labels for unlabeled prescription images.
This creates a labeled dataset for training our custom OCR model.
"""
import os
import csv
import pytesseract
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# Tesseract path (update this for your system)
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/Mac: usually automatically found

INPUT_DIR = 'data/prescriptions'
OUTPUT_CSV = 'data/tesseract_labels/prescription_labels.csv'
os.makedirs('data/tesseract_labels', exist_ok=True)

def preprocess_for_tesseract(img):
    """Preprocess image to improve Tesseract accuracy"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    # 1. Noise removal
    img = cv2.medianBlur(img, 3)
    
    # 2. Thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological operations to clean up image
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return img

def extract_text_with_tesseract(img_path):
    """Extract text from image using Tesseract with multiple configurations"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return ""
        
        # Preprocess image
        processed_img = preprocess_for_tesseract(img)
        
        # Try different Tesseract configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 7',  # Single text line
            '--psm 8',  # Single word
            '--psm 11'  # Sparse text
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Get OCR data with confidence
                data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=config)
                
                # Calculate average confidence for non-empty words
                confidences = [int(data['conf'][i]) for i in range(len(data['text'])) 
                             if data['text'][i].strip() and int(data['conf'][i]) > 0]
                
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = ' '.join([data['text'][i] for i in range(len(data['text'])) 
                                   if data['text'][i].strip()])
                    
                    if avg_confidence > best_confidence and text.strip():
                        best_confidence = avg_confidence
                        best_text = text
            except Exception as e:
                continue
        
        return best_text.strip()
    
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return ""

def filter_valid_prescription_text(text):
    """Filter and clean extracted text to keep only plausible prescription content"""
    if not text:
        return ""
    
    # Remove obviously wrong OCR results
    if len(text) < 3:
        return ""
    
    # Keep only lines that contain plausible prescription content
    words = text.split()
    valid_words = []
    
    for word in words:
        # Filter based on common prescription patterns
        if (any(c.isalpha() for c in word) or  # Contains letters
            any(c.isdigit() for c in word) or  # Contains numbers
            any(c in '.,-/%():mg' for c in word)):  # Contains common prescription symbols
            valid_words.append(word)
    
    return ' '.join(valid_words)

def main():
    """Main function to label all prescription images using Tesseract"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    # Collect all image files
    for file in os.listdir(INPUT_DIR):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(INPUT_DIR, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Create CSV with Tesseract labels
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'tesseract_label', 'confidence_estimate'])
        
        for img_path in tqdm(image_files, desc="Processing images with Tesseract"):
            text = extract_text_with_tesseract(img_path)
            filtered_text = filter_valid_prescription_text(text)
            
            # Simple confidence estimate based on text characteristics
            confidence = min(90, len(filtered_text) * 2) if filtered_text else 0
            
            writer.writerow([img_path, filtered_text, confidence])
    
    print(f"Tesseract labeling complete. Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()