# ⚡ Quick-Witted Fraud Detection System

A real-time machine learning system designed to detect fraudulent financial transactions with low-latency and high precision.

## 📌 Project Overview
Fraud detection is a high-stakes problem where decisions must be both fast and accurate.  
This project focuses on identifying fraudulent transactions by learning behavioral patterns from historical data and making instant risk-based decisions.

## 🚀 Key Features
- Real-time fraud detection pipeline
- Intelligent feature engineering (transaction velocity, amount deviation, time-based risk)
- Imbalanced data handling using SMOTE and class-weighted models
- Threshold tuning to reduce false positives
- Precision–Recall focused evaluation

## 🧠 Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn
- FastAPI (optional)
- Streamlit (optional)

## 📊 Machine Learning Workflow
1. Data collection and inspection  
2. Exploratory Data Analysis (EDA)  
3. Data preprocessing and scaling  
4. Feature engineering  
5. Imbalanced data handling  
6. Model training and comparison  
7. Threshold optimization  
8. Real-time inference simulation  

## 📈 Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall Curve

## 🏗️ Project Structure
Refer to the folder structure in the repository for modular implementation.

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
