# DS.v2.5.3.2.5

# Stroke Prediction Model

## Overview
This project focuses on predicting stroke risk using machine learning models. The dataset was preprocessed, key features were selected, and various models were evaluated to achieve the best predictive performance. The study also explored different data balancing techniques, including oversampling and undersampling.

## Features and Methodology

### **Data Pre-processing and Transformation**
- **Feature Encoding & Scaling:** Categorical features were encoded using OrdinalEncoder, while numerical features were standardized using StandardScaler.
- **Data Splitting:** The dataset was divided into training and testing sets using stratified sampling.
- **Class Imbalance Handling:** Implemented oversampling using SMOTE-NC and undersampling using RandomUnderSampler.

### **Model Selection and Evaluation**
- **Baseline Models:** Initial models were trained using default parameters to establish baseline performance.
- **Hyperparameter Tuning:** GridSearchCV was used to optimize model performance.
- **Model Performance Metrics:** Accuracy, precision, recall, and F1-score were used for evaluation.

### **Deployment Strategies**
Three different approaches were tested:
1. **Without data imbalance handling**
2. **With oversampling (SMOTE-NC)**
3. **With undersampling (RandomUnderSampler)**

## Key Findings
- **CatBoost with selected features performed best**, achieving a recall of 0.85, making it highly effective in identifying stroke cases.
- **Age was the most significant predictor**, followed by glucose level, BMI, hypertension, and smoking status.
- **Handling class imbalance impacted performance**, with SMOTE-NC improving recall but introducing synthetic data, while undersampling preserved data integrity but removed information.
- **Feature selection improved efficiency**, as mutual information helped streamline the model while maintaining strong predictive power.

## Future Improvements
- Exploring additional ensemble learning methods like XGBoost.
- Enhancing feature engineering to uncover hidden patterns.
- Testing alternative data balancing techniques to optimize performance.

## Installation & Usage
### **Requirements**
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```

## Authors
Developed by Arantxa Ortega.


