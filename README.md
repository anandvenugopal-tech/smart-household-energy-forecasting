# Smart Household Energy Consumption Forecasting

## Overview
This project focuses on predicting household energy consumption using historical time-series data. It compares traditional machine learning models with a deep learning approach (LSTM) and builds a simple pipeline for training and evaluation.

---

## Problem Statement
Given past electricity usage data, predict future energy consumption. This can help in understanding usage patterns and planning energy usage more efficiently.

---

## Approach

### 1. Data Processing
- Cleaned the dataset by handling missing values
- Converted relevant columns to numeric format
- Prepared time-series data for modeling

---

### 2. Machine Learning Models
- Linear Regression (baseline)
- Decision Tree
- Random Forest

Steps performed:
- Model comparison using MAE
- Hyperparameter tuning using GridSearchCV
- Selected best-performing model

---

### 3. Deep Learning Model (LSTM)
- Built using PyTorch
- Created sequences using sliding window approach
- Applied MinMax scaling
- Trained model to learn temporal patterns

---

### 4. Evaluation
- Used Mean Absolute Error (MAE)
- Compared ML models vs LSTM
- Converted predictions back to original scale before evaluation

---

### 5. Workflow Automation
- Used Apache Airflow to organize steps:
  - Data ingestion
  - Cleaning
  - Model training
  - Evaluation

---

### 6. Deployment Setup
- Built a simple API using FastAPI for predictions
- Used Docker to containerize the application

---

## Results
- Random Forest performed well among traditional models
- LSTM captured time dependencies but required more tuning
- Final model selected based on lowest MAE and stability

---

## Conclusion
This project demonstrates how different modeling approaches perform on time-series data and highlights the trade-off between simplicity and performance. It also shows how to structure a basic ML pipeline with training, evaluation, and deployment components.

