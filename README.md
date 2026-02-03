# Smart Household Energy Consumption Forecasting 

This project implements an end-to-end, production-grade system for forecasting household electricity consumption using machine learning, deep learning, and modern MLOps practices. It demonstrates the full lifecycle of a real-world forecasting pipeline: data handling, feature engineering, modeling, deployment, and automation.

The solution combines traditional regression models with LSTM-based deep learning to capture both short-term and long-term consumption patterns. The system is modular, scalable, and designed to operate in a continuous retraining environment using Apache Airflow for workflow orchestration and FastAPI for model serving.

---

## Key Features

- Complete data pipeline for ingestion, cleaning, and transformation  
- Comprehensive exploratory and advanced visual analysis  
- Feature engineering for ML and deep learning models  
- Machine learning forecasting using multiple regression algorithms  
- Deep learning sequence modeling using LSTM  
- Behavioral load pattern clustering for deeper insights  
- Modular Python codebase designed for production  
- REST API for real-time inference using FastAPI  
- Automated retraining and pipeline management using Apache Airflow  
- Docker-enabled reproducible environment for development and deployment  

---

## Modeling Overview

### Machine Learning Models  
The project includes multiple regression models to benchmark classical approaches against deep learning, including:  
- Linear Regression  
- Random Forest  
- Gradient Boosting (XGBoost, LightGBM)  
- KNN Regression  

### Deep Learning  
A sequence modeling approach using LSTM captures temporal dependencies and improves forecasting accuracy in multi-step and minute-level prediction tasks.

### Clustering  
Consumption patterns are analyzed using unsupervised clustering techniques, such as:  
- K-Means  
- DBSCAN  
These methods help uncover daily usage behaviors and consumer segments.

---

## MLOps Workflow

The project uses Apache Airflow to orchestrate all core pipeline tasks, enabling continuous, automated learning.  
Automated stages include:  
- Scheduled data ingestion  
- Cleaning and feature updates  
- Model retraining  
- Saving updated model artifacts  
- Refreshing the prediction API with the latest model  

FastAPI exposes the forecasting model through an endpoint for real-time consumption predictions, suitable for integration into dashboards or energy management systems.

---

## Technical Highlights

- Demonstrates end-to-end ML, DL, and MLOps in a single project  
- Separation of experimentation (notebooks) and production-ready code (src/)  
- Fully modular pipeline allowing rapid iteration and scalable deployment  
- Supports both traditional ML workflows and modern deep learning pipelines  
- Designed for real-world deployment with automated monitoring and updates  

---

## Future Enhancements

- Integration of transformer-based forecasting models  
- Model experiment tracking using MLflow  
- Real-time data ingestion through streaming platforms  
- Cloud deployment on AWS/GCP/Azure  
- Monitoring dashboards using Prometheus and Grafana  

---

