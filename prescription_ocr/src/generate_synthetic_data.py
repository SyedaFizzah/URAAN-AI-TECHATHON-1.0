"""Generate synthetic labeled word images for training.
Fixed to generate shorter labels for CTC compatibility.
"""
import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUT_DIR = 'data/synthetic_data'
CSV_PATH = 'data/prescription_data.csv'

# Shorter labels for CTC compatibility
COMMON_MEDICINES = [
    'Paracetamol', 'Amoxicillin', 'Aspirin', 'Ibuprofen',
    'Metformin', 'Omeprazole', 'Atorvastatin', 'Cetirizine',
    'Vitamin C', 'Calcium', 'Azithromycin', 'Ciprofloxacin'
]

DOSAGE_INSTRUCTIONS = [
    '1 tab daily', '2 tabs bid', '1 cap', 'With food',
    'Before meal', 'At bedtime', '6h prn', 'Apply thin'
]

os.makedirs(OUT_DIR, exist_ok=True)

# Try to load fonts
FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    'C:/Windows/Fonts/arial.ttf',
    'C:/Windows/Fonts/times.ttf',
]

font = None
for p in FONT_PATHS:
    if os.path.exists(p):
        try:
            font = ImageFont.truetype(p, 24)  # Slightly smaller font
            print(f"Using font: {p}")
            break
        except Exception:
            continue

if font is None:
    try:
        font = ImageFont.load_default()
        print("Using default font")
    except:
        print("Warning: Could not load any font")

from PIL import ImageFilter

def render_text(text, out_path):
    if font is None:
        return False
        
    try:
        # Smaller image for shorter text
        img = Image.new('L', (256, 32), color=255)
        draw = ImageDraw.Draw(img)

        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (256 - text_width) // 2
        y = (32 - text_height) // 2

        draw.text((x, y), text, font=font, fill=0)

        # Mild augmentations
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.8)))

        # Resize to model input size
        img = img.resize((128, 32))
        img.save(out_path)
        return True
    except Exception as e:
        print(f"Error rendering text '{text}': {e}")
        return False

def main(n=500):  # Reduced number for testing
    # create csv
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'image_path'])

        count = 0
        successful_renders = 0
        
        for i in range(n):
            # Generate shorter labels
            if random.random() < 0.7:
                text = random.choice(COMMON_MEDICINES)
                if random.random() < 0.4 and len(text) < 15:
                    text += ' ' + random.choice(DOSAGE_INSTRUCTIONS)
            else:
                text = random.choice(['Take 1 tab', 'Before meal', 'After meal', 
                                    '1 tab bid', 'Apply', 'With water'])

            # Ensure label isn't too long
            if len(text) > 25:
                text = text[:25]
                
            filename = f'synth_{i:05d}.png'
            out_path = os.path.join(OUT_DIR, filename)
            
            if render_text(text, out_path):
                writer.writerow([text, out_path])
                successful_renders += 1
            
            count += 1
            
            if count % 100 == 0:
                print(f"Generated {count}/{n} samples...")
    
    print(f'Successfully generated {successful_renders} synthetic samples to {OUT_DIR}')
    print(f'CSV saved at {CSV_PATH}')

if __name__ == '__main__':
    main(n=500)