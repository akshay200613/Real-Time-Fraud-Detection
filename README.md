#  Real-Time Fraud Detection using Apache Spark MLlib

An end-to-end **Real-Time Fraud Detection System** built using **Apache Spark**, **PySpark MLlib**, **FastAPI**, and **Streamlit**. This project demonstrates the complete machine learning lifecycle, including ETL, feature engineering, model training, hyperparameter tuning, model evaluation, REST API deployment, and an interactive web dashboard for fraud prediction.

---

##  Features

- End-to-End ETL Pipeline using Apache Spark
- Data Cleaning & Feature Engineering
- Multiple Machine Learning Models using Spark MLlib
- Hyperparameter Tuning with Cross Validation
- Model Evaluation using Multiple Metrics
- Batch Fraud Prediction
- Real-Time Prediction API using FastAPI
- Interactive Streamlit Dashboard
- Download Prediction Results
- Modular and Scalable Project Structure

---

## 📂 Dataset

**Dataset:** IEEE-CIS Fraud Detection Dataset

🔗 https://www.kaggle.com/competitions/ieee-fraud-detection

Files Used:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

---

##  Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.11 |
| Big Data Framework | Apache Spark 4.x |
| Machine Learning | PySpark MLlib |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Visualization | Plotly |
| Version Control | Git & GitHub |

---

# 📁 Project Structure

```text
Real-Time-Fraud-Detection/
│
├── data/
│   └── raw/
│
├── logs/
│
├── models/
│
├── reports/
│
├── src/
│   ├── etl/
│   │   ├── extract.py
│   │   ├── cleaning.py
│   │   ├── feature_engineering.py
│   │   └── pipeline.py
│   │
│   ├── ml/
│   │   ├── prepare_data.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── hyperparameter.py
│   │   └── predict.py
│   │
│   ├── utils/
│   │   └── logger.py
│   │
│   ├── config.py
│   └── spark_session.py
│
├── api.py
├── infer.py
├── inspect_pipeline.py
├── main.py
├── streamlit_app.py
│
├── requirements.txt
├── requirements_web.txt
├── README.md
└── .gitignore
```

---

#  ETL Pipeline

## 1️⃣ Extract

- Read Transaction Dataset
- Read Identity Dataset
- Merge datasets using `TransactionID`

---

## 2️⃣ Transform

### Data Cleaning

- Handle Missing Values
- Remove Columns with High Missing Percentage
- Median Imputation
- Fill Missing Categorical Values

### Feature Engineering

Created Features:

- TransactionHour
- TransactionDay
- HighAmount
- EmailMatch

---

## 3️⃣ Load

- Prepare Data for Machine Learning
- Create Feature Vectors
- Save Pipeline
- Save Trained Models

---

#  Machine Learning Models

The following Spark MLlib classification algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Trees (GBT)

---

#  Hyperparameter Tuning

Random Forest was optimized using:

- Grid Search
- 3-Fold Cross Validation
- ROC-AUC Evaluation

### Tuned Parameters

- Number of Trees
- Maximum Tree Depth
- Maximum Bins

---

#  Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

#  Prediction Pipeline

The prediction workflow performs:

1. Load Saved Pipeline
2. Load Trained Random Forest Model
3. Read Unseen Transaction Data
4. Apply ETL Pipeline
5. Generate Fraud Predictions
6. Display Fraud Probability
7. Export Prediction Results

---

#  FastAPI Backend

Start the API server:

```bash
pip install -r requirements_web.txt

python -m uvicorn api:app --reload
```

API URL

```
http://localhost:8000
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

#  Streamlit Dashboard

Start the dashboard:

```bash
streamlit run streamlit_app.py
```

### Dashboard Features

### 🔹 Single Transaction Prediction

- Autofill Transaction Templates
- Fraud Probability Prediction
- Risk Gauge Visualization
- Instant Fraud Classification

### 🔹 Batch CSV Prediction

- Upload CSV Files
- Batch Fraud Prediction
- Download Prediction Results
- Summary Analytics

### 🔹 Interactive Visualizations

- Fraud Distribution
- Fraud Probability Chart
- Transaction Amount vs Fraud Probability
- Summary Statistics

### 🔹 Model Insights

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

#  How to Run

## Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Models

```bash
python main.py
```

---

## Run Inference

```bash
python infer.py
```

---

## Start FastAPI Backend

```bash
pip install -r requirements_web.txt

python -m uvicorn api:app --reload
```

---

## Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

#  Complete Workflow

```text
IEEE-CIS Dataset
        │
        ▼
Extract
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Feature Preparation
        │
        ▼
Spark ML Models
        │
        ▼
Model Evaluation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Save Best Model
        │
        ▼
FastAPI Backend
        │
        ▼
Streamlit Dashboard
        │
        ▼
Real-Time Fraud Prediction
```

---

#  Key Features

- Apache Spark ETL Pipeline
- Scalable Data Processing
- Feature Engineering
- Spark MLlib Models
- Hyperparameter Optimization
- Model Persistence
- REST API Deployment
- Interactive Dashboard
- Batch Prediction
- Downloadable Prediction Results

---

#  Learning Outcomes

This project provided practical experience in:

- Apache Spark
- Distributed Data Processing
- ETL Pipeline Development
- Feature Engineering
- Spark MLlib
- Hyperparameter Tuning
- Model Evaluation
- FastAPI Development
- Streamlit Dashboard Development
- End-to-End Machine Learning Pipeline Design

---

#  Future Improvements

- Real-Time Streaming using Spark Structured Streaming
- Kafka Integration
- Docker Containerization
- Kubernetes Deployment
- AWS Cloud Deployment
- MLflow Experiment Tracking
- Model Monitoring and Drift Detection

---

#  Author

**Akshay**

AI & Machine Learning Student

---

## ⭐ If you found this project useful, consider giving it a Star!
