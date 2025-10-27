

## README.md

````markdown
# Prescription OCR — Complete Pipeline

## Overview
This repo trains a CNN+BiLSTM+CTC model for handwritten prescription OCR. It uses unlabeled Kaggle prescription images for fine-tuning (denoising/augmentation) and synthetic labeled data to train the OCR head.

## Quick start
1. Create python venv and install requirements:

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Download Kaggle dataset (put images into `data/prescriptions/`).

3. Generate synthetic labeled data (creates `data/synthetic_data/` and `data/prescription_data.csv`):

```bash
python src/generate_synthetic_data.py
```

4. Train model:

```bash
python src/wrd_train.py
```

5. Run inference on a test image:

```bash
python src/inference.py --image_path "data/prescriptions/some_image.jpg"
```

Results stored in `results/predictions/` as `.txt` files.

## Notes
- The Kaggle dataset you provided (`mehaksingal/illegible-medical-prescription-images-dataset`) is unlabeled: we use it for augmentation/denoising and optional fine-tuning.
- Synthetic data provides supervised labels. If you can create even a small labeled set (200–1000 line-level samples) from real prescriptions, fine-tune the saved model with them — accuracy will improve significantly.

````


### Final notes

* This repo is a functional, end-to-end pipeline. It's intentionally simple and uses synthetic labeled data to bootstrap training because the Kaggle dataset is unlabeled.
* For competition: gather ~500–2k labeled real samples (line- or word-level) if possible and fine-tune the saved model — this will push you well above 80%.
