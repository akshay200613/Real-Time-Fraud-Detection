# Web Dashboard and API Instructions

This package contains the **FastAPI Backend Service** (`api.py`) and the **Streamlit Frontend Dashboard** (`streamlit_app.py`) for the Real-Time Fraud Detection application.

## 🚀 Setup Instructions

### 1. Install Dependencies
Install the required packages for running the web backend and frontend. You can use your active Python environment:

```bash
pip install -r requirements_web.txt
```

### 2. Start the FastAPI Backend
The backend utilizes PySpark to load the machine learning model stages. It keeps a single persistent Spark session in memory to provide high-speed inference.

Run the following command from the project root:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** The backend might take 5–10 seconds to spin up, configure the PySpark JVM context, load the Random Forest model and Pipeline stages, and load sample template rows. Once online, it will show a message in the console.

### 3. Start the Streamlit Frontend
The frontend dashboard connects to the FastAPI backend to run predictions and render metrics.

From a separate terminal window in the project root, run:

```bash
streamlit run streamlit_app.py
```

Streamlit will automatically open a browser window at `http://localhost:8501`.

---

## 📈 Dashboard Features

### 🔍 Single Transaction Predictor
- **Autofill Templates**: Choose a sample transaction from the drop-down menu. This dynamically pulls real transactions from the test dataset to pre-populate all 300+ parameters in the background.
- **Interactive Form**: Adjust main factors (e.g. Transaction Amount, Product Type, Card Network, Email domains, Card types).
- **Risk Evaluation**: Runs instant prediction, displaying a high-contrast glowing card (Green for Legitimate, Red for Fraud) and a detailed Plotly gauge chart mapping the exact fraud probability.

### 📁 Batch CSV Predictor
- **CSV Uploader**: Drop a CSV file containing transactions.
- **Template Generator**: If you don't have a batch file ready, click the "Generate Sample CSV File" button to download a 10-row dataset template.
- **Analytics Dashboard**: Generates key figures (Fraud Rates, Averages) and interactive charts (distribution pie chart, scatter plot of amount vs. fraud probability).
- **Exporting predictions**: Download the entire predicted dataset containing predicted labels and risk scores.

### 📊 Model Insights
- View high-fidelity metrics computed during evaluation (Accuracy, Area Under ROC (AUC), Precision, F1-Score).
- Visual flowchart showcasing the Spark ETL and Machine Learning pipeline architecture.
