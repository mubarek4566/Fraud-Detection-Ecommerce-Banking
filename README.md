# Fraud-Detection-Ecommerce-Banking
This repository contains data analysis, feature engineering, machine learning models, and explainability techniques developed to improve the detection of fraudulent activities in e-commerce and bank credit transactions.

# Fraud Detection for E-commerce and Banking Transactions

## Overview
This project aims to develop accurate fraud detection models for e-commerce and bank credit transactions using advanced machine learning techniques. We address challenges such as class imbalance, real-time detection needs, and explainability to enhance trust and reduce financial losses.

## Business Context
Fraudulent transactions cause significant financial and reputational damage. Balancing fraud prevention with user experience is critical to avoid alienating legitimate customers. This project supports Adey Innovations Inc. in improving transaction security by combining data analysis, feature engineering, and explainable AI.

## Data Description
- **Fraud_Data.csv**: E-commerce transaction data with user, device, and transaction details.
- **IpAddress_to_Country.csv**: IP address ranges mapped to countries for geolocation analysis.
- **creditcard.csv**: Bank credit card transaction data with anonymized PCA features.

## Project Components
1. Data cleaning and preprocessing  
2. Feature engineering including time, geolocation, and device-based features  
3. Handling class imbalance with techniques such as SMOTE, undersampling, or class weighting  
4. Machine learning model training and evaluation using metrics suitable for imbalanced datasets (AUC-PR, F1-Score)  
5. Model interpretability using SHAP to understand feature impacts  
6. Performance comparison and final model selection  

## Installation
Describe required packages and environment setup. Example:

```bash
pip install -r requirements.txt
