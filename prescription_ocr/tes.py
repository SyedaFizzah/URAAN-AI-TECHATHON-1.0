import pytesseract
from PIL import Image
import os

input_folder = "data/prescriptions/"
output_file = "data/prescriptions_labels.csv"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("image_path,label\n")
    for img_name in os.listdir(input_folder):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, img_name)
            text = pytesseract.image_to_string(Image.open(img_path))
            text = text.replace("\n"," ").strip()
            f.write(f"{img_path},{text}\n")
