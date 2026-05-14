# Customer Churn Prediction

Predict telecom customer churn using machine learning to enable proactive retention strategies.

## 🎯 Objective
Identify at-risk customers before they churn, reducing customer acquisition costs through targeted retention efforts.

## 📊 Dataset
- **Source:** Telco Customer Churn Dataset
- **Size:** 7,043 records, 21 features
- **Class Imbalance:** 26.5% churn rate

## 🛠️ Tech Stack
- Python | Pandas | NumPy | Scikit-learn | XGBoost
- SMOTE | StandardScaler | Streamlit | Joblib

## 🔑 Key Results

| Model | Baseline Recall | Improved Recall |
|-------|----------------|-----------------|
| Logistic Regression | 56.7% | 97.6% |
| Random Forest | 49.2% | 85.3% |
| **XGBoost (Selected)** | 53.5% | **85.3%** |

**Final Model Metrics (Threshold = 0.30):**
| Metric | Value |
|--------|-------|
| Recall | 85.3% |
| Precision | 39% |
| F1-Score | 54% |

## 🚀 Approach

1. **Data Cleaning:** Converted TotalCharges to numeric; filled missing values with tenure × MonthlyCharges
2. **EDA:** Analyzed churn distribution, tenure patterns, and categorical feature relationships
3. **Preprocessing:** One-hot encoding with drop_first; standardized features
4. **Class Imbalance:** Applied SMOTE on training data after train/test split
5. **Modeling:** Trained and compared Logistic Regression, Random Forest, and XGBoost
6. **Threshold Tuning:** Set threshold to 0.30 to optimize recall for churn detection
7. **Deployment:** Saved pipeline with joblib; built Streamlit app for real-time predictions

## 🖥️ Live Demo

```bash
# Clone repo
git clone https://github.com/Poorna321288/Customer-Churn-Prediction.git

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

📁 Project Structure

├── data/
│   └── raw/
│       └── Telecom-Customer-Churn.csv
├── notebooks/
│   └── Churn_Prediction.ipynb
├── models/
│   ├── churn_pipeline.joblib
│   └── feature_columns.joblib
├── app.py
├── requirements.txt
└── README.md

📝 What I Learned
 
Handling class imbalance with SMOTE for better minority class detection
 
Threshold tuning to prioritize recall in business-critical scenarios
 
Building end-to-end ML pipelines with preprocessing and model persistence
 
Deploying interactive ML applications with Streamlit
