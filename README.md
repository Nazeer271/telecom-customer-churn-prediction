# 📉 Telecom Customer Churn Prediction

A machine learning classification model to predict whether a customer will churn.

## 📌 Overview

Customer churn is one of the biggest challenges for subscription-based businesses. This project uses demographic and service-related data to build a classification model that identifies customers at high risk of churning, enabling proactive retention strategies.

## 🎯 Problem Statement

Given customer data including gender, senior citizen status, tenure, internet service type, contract type, and charges, predict whether a customer will **churn (1)** or **stay (0)**.

## 📂 Project Structure

```
telecom-customer-churn-prediction/
│
├── customer_churn_prediction.ipynb   # Main notebook
├── data/
│   └── Data_file.csv                 # Dataset
├── README.md
└── requirements.txt
```

## 🔧 Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## 📊 Workflow

1. **Exploratory Data Analysis (EDA)** — Churn distribution (≈38.9% churn rate), feature correlations, visualizations
2. **Data Preprocessing** — Missing value handling, label encoding, feature scaling with StandardScaler
3. **Model Development** — Four classifiers trained and compared
4. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, AUC-ROC
5. **Predictions on New Data** — Final model applied to unseen records

## ✅ Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6125 | 0.5012 | 0.7336 | 0.5956 | 0.6711 |
| Decision Tree | 0.6182 | 0.5064 | 0.7190 | 0.5943 | 0.6685 |
| Random Forest | 0.6317 | 0.5298 | 0.4708 | 0.4986 | 0.6442 |
| **Gradient Boosting** ✅ | **0.6515** | **0.5578** | **0.5018** | **0.5283** | **0.6873** |

> **Selected Model: Gradient Boosting** — Highest overall accuracy (65.15%) and AUC-ROC (0.6873), with the best balance across all metrics.

### Key Insights
- Month-to-month contract customers are significantly more likely to churn
- Fiber optic internet service users show higher churn rates
- Longer tenure strongly correlates with lower churn probability
- Senior citizens exhibit a higher churn tendency

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/telecom-customer-churn-prediction.git
cd telecom-customer-churn-prediction
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
```

## 💡 Business Implications

- Target month-to-month contract customers with loyalty incentives
- Investigate service quality issues among Fiber optic users
- Design early engagement programs for new customers (low tenure)

## 📋 Guidelines

- Data split: 80% training / 20% testing with stratified sampling
- Class weights balanced to handle churn imbalance
- All steps thoroughly documented in the notebook
