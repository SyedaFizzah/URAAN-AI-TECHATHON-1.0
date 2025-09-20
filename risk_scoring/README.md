# Heart Attack Risk Scoring Using Machine Learning

## Project Overview
This project aims to predict the risk of heart attack (`cardio`) in patients using a combination of demographic, lifestyle, and medical features. We applied multiple machine learning models to determine the most accurate predictor for cardiovascular risk.

**Dataset:**  
The dataset consists of 70,000 entries with 12 features including age, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol consumption, physical activity, and gender.

---

## Features

| Feature       | Description                               |
|---------------|-------------------------------------------|
| age           | Age of the patient (in days)             |
| height        | Height (cm)                               |
| weight        | Weight (kg)                               |
| ap_hi         | Systolic blood pressure                   |
| ap_lo         | Diastolic blood pressure                  |
| cholesterol   | Cholesterol level (1=normal, 2=above, 3=high) |
| gluc          | Glucose level (1=normal, 2=above, 3=high)     |
| smoke         | Smoking status (0=No, 1=Yes)             |
| alco          | Alcohol consumption (0=No, 1=Yes)        |
| active        | Physical activity (0=No, 1=Yes)          |
| female        | Gender encoded (boolean)                  |
| male          | Gender encoded (boolean)                  |
| cardio        | Target variable: Heart disease risk (0=No, 1=Yes) |

---

## Models Implemented
We trained and compared the following machine learning models:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**
4. **XGBoost Classifier**
5. **Feedforward Neural Network (MLP)**

**Performance Metric:**  
We primarily used **ROC-AUC** along with **precision, recall, and F1-score** to evaluate model performance.  

**Best Model:**  
The **Random Forest Classifier** performed the best with **ROC-AUC = 0.981** using 1000 trees. This model is saved for future predictions.

---

## Workflow

1. **Data Preprocessing**
   - Removed irrelevant columns (`index`, `id`)
   - One-hot encoded categorical variables (`gender`)
   - Scaled numeric features using `StandardScaler`
   - (Optional) PCA for dimensionality reduction

2. **Model Training**
   - Split data into `train` and `test` sets
   - Trained each model
   - Used `class_weight='balanced'` for handling class imbalance
   - Neural network used **dropout** and **early stopping** to prevent overfitting

3. **Model Evaluation**
   - Calculated **classification report** for each model
   - Plotted **ROC curves** to compare models visually
   - Selected **Random Forest** as the final model

4. **Saving Artifacts**
   - Random Forest model saved as `random_forest_model.pkl` in Google Drive
   - Notebook saved to the same folder for reproducibility

---drive.mount('/content/drive')

