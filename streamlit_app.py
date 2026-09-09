import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import io

# Setup Page Configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

# Backend API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Inject Custom CSS for premium glassmorphic dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
        color: #f3f4f6 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    .metric-header {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-value-fraud {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ef4444, #f87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value-success {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-fraud {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 25px;
        color: #fca5a5;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
        text-align: center;
    }
    
    .status-legit {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 25px;
        color: #a7f3d0;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
        text-align: center;
    }
    
    .stAlert {
        border-radius: 12px !important;
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    .feature-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 0.82rem;
        color: #7dd3fc;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# API helpers
# --------------------------------------------------

@st.cache_data(ttl=5)
def check_backend_status():
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("pyspark_version", "Active")
    except Exception as e:
        print(f"Backend status check failed: {e}")
    return False, None


def fetch_samples():
    try:
        response = requests.get(f"{API_URL}/samples", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=60)
def fetch_metrics():
    """Fetches live model metrics from the API (cached for 60 seconds)."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# --------------------------------------------------
# App Header
# --------------------------------------------------

st.title("🛡️ IEEE-CIS Real-Time Fraud Detection")
st.write("An enterprise-grade Spark ETL & Machine Learning prediction platform designed to intercept fraudulent transactions in sub-seconds.")

backend_online, spark_ver = check_backend_status()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/shield.png", width=90)
    st.markdown("### System Diagnostics")

    if backend_online:
        st.markdown(f"""
        <div class="metric-card" style="padding: 10px 15px; margin-bottom: 10px;">
            <div style="color: #10b981; font-weight: bold; font-size: 1rem;">● Backend Online</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Spark version: {spark_ver}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card" style="padding: 10px 15px; margin-bottom: 10px;">
            <div style="color: #ef4444; font-weight: bold; font-size: 1rem;">○ Backend Offline</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Run: <code>uvicorn api:app</code></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Architecture")
    st.markdown("""
    - **Classification Model**: Random Forest (class-weighted)
    - **Feature Preprocessing**: PySpark Pipeline
    - **Imputation**: Median (Numeric — fitted on train only)
    - **Class Imbalance**: Handled via `classWeight`
    - **Engineered Features**: 
      * `TransactionHour`, `TransactionDay`, `IsWeekend`
      * `HighAmount`, `AmtLog`, `AmtToCard1Ratio`
      * `EmailMatch`, `PEmailHasDigit`, `PEmailUnknown`
      * `CardTypeUnknown`, `CSum`, `DMax`
    """)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Transaction Predictor",
    "📁 Batch CSV Predictor",
    "📈 Model Evaluation Insights"
])


# --------------------------------------------------
# TAB 1: Single Prediction
# --------------------------------------------------

with tab1:
    if not backend_online:
        st.warning("⚠️ The FastAPI backend is currently offline. Please start the backend service to run predictions.")
    else:
        st.markdown("### Predict Individual Transaction")
        st.write("Select a real sample transaction from the test dataset to auto-populate the form, or enter details manually.")

        samples = fetch_samples()
        sample_options = ["Manual Entry"] + [
            f"Sample Transaction ID: {s.get('TransactionID', i)}"
            for i, s in enumerate(samples)
        ]
        selected_sample = st.selectbox("Autofill Template", sample_options)

        default_data = {}
        if selected_sample != "Manual Entry":
            idx = sample_options.index(selected_sample) - 1
            default_data = samples[idx]

        st.markdown("#### Transaction Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            tx_amt = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=100000.0,
                value=float(default_data.get("TransactionAmt", 50.0))
            )
            product_cd = st.selectbox(
                "Product Code (ProductCD)",
                ["W", "H", "C", "S", "R"],
                index=["W", "H", "C", "S", "R"].index(default_data.get("ProductCD", "W"))
                if default_data.get("ProductCD") in ["W", "H", "C", "S", "R"] else 0
            )
            card4 = st.selectbox(
                "Card Network (card4)",
                ["visa", "mastercard", "american express", "discover", "Unknown"],
                index=["visa", "mastercard", "american express", "discover", "Unknown"].index(
                    default_data.get("card4", "visa")
                ) if default_data.get("card4") in ["visa", "mastercard", "american express", "discover", "Unknown"] else 0
            )
            card6 = st.selectbox(
                "Card Type (card6)",
                ["debit", "credit", "Unknown"],
                index=["debit", "credit", "Unknown"].index(default_data.get("card6", "debit"))
                if default_data.get("card6") in ["debit", "credit", "Unknown"] else 0
            )

        with col2:
            p_email = st.text_input(
                "Purchaser Email Domain (P_emaildomain)",
                value=str(default_data.get("P_emaildomain", "gmail.com"))
            )
            r_email = st.text_input(
                "Recipient Email Domain (R_emaildomain)",
                value=str(default_data.get("R_emaildomain", "gmail.com"))
            )
            dev_type = st.selectbox(
                "Device Type",
                ["desktop", "mobile", "Unknown"],
                index=["desktop", "mobile", "Unknown"].index(default_data.get("DeviceType", "Unknown"))
                if default_data.get("DeviceType") in ["desktop", "mobile", "Unknown"] else 2
            )
            tx_dt = st.number_input(
                "Transaction Delta Time (TransactionDT)",
                min_value=0,
                value=int(default_data.get("TransactionDT", 18403200))
            )

        with col3:
            card1 = st.number_input("Card Parameter 1 (card1)", value=int(default_data.get("card1", 10000)))
            card2 = st.number_input("Card Parameter 2 (card2)", value=float(default_data.get("card2", 321.0)) if default_data.get("card2") else 321.0)
            card3 = st.number_input("Card Parameter 3 (card3)", value=float(default_data.get("card3", 150.0)) if default_data.get("card3") else 150.0)
            card5 = st.number_input("Card Parameter 5 (card5)", value=float(default_data.get("card5", 226.0)) if default_data.get("card5") else 226.0)

        # Build payload: start from template, override with form inputs
        transaction_payload = {**default_data}
        transaction_payload.update({
            "TransactionAmt": tx_amt,
            "ProductCD":      product_cd,
            "card1":          card1,
            "card2":          card2,
            "card3":          card3,
            "card4":          card4,
            "card5":          card5,
            "card6":          card6,
            "P_emaildomain":  p_email if p_email else "Unknown",
            "R_emaildomain":  r_email if r_email else "Unknown",
            "DeviceType":     dev_type,
            "TransactionDT":  tx_dt
        })

        st.markdown("---")

        if st.button("🚀 Analyze Transaction", use_container_width=True):
            with st.spinner("Executing fraud check via PySpark model..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=[transaction_payload])
                    if response.status_code == 200:
                        res  = response.json().get("results")[0]
                        prob = res.get("FraudProbability")
                        label = res.get("PredictionLabel")

                        col_res1, col_res2 = st.columns([1, 1])

                        with col_res1:
                            if label == "Fraud":
                                st.markdown(f"""
                                <div class="status-fraud">
                                    <h2 style="margin: 0; font-size: 2rem;">🚨 FRAUD DETECTED</h2>
                                    <p style="font-size: 1.1rem; margin-top: 10px;">This transaction flags high risk of fraudulent activities.</p>
                                    <div style="font-size: 3rem; font-weight: 800; margin: 15px 0;">{prob*100:.2f}%</div>
                                    <span style="font-weight: 500; font-size: 0.9rem; text-transform: uppercase;">Fraud Confidence Score</span>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="status-legit">
                                    <h2 style="margin: 0; font-size: 2rem;">✅ LEGITIMATE</h2>
                                    <p style="font-size: 1.1rem; margin-top: 10px;">This transaction is verified clean and safe.</p>
                                    <div style="font-size: 3rem; font-weight: 800; margin: 15px 0;">{prob*100:.2f}%</div>
                                    <span style="font-weight: 500; font-size: 0.9rem; text-transform: uppercase;">Fraud Probability</span>
                                </div>
                                """, unsafe_allow_html=True)

                        with col_res2:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Fraud Risk Gauge", 'font': {'size': 20, 'color': '#ffffff'}},
                                number={'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
                                gauge={
                                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                                    'bar': {'color': "#ef4444" if label == "Fraud" else "#10b981"},
                                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                                    'borderwidth': 2,
                                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                                    'steps': [
                                        {'range': [0, 25],  'color': 'rgba(16, 185, 129, 0.1)'},
                                        {'range': [25, 75], 'color': 'rgba(245, 158, 11, 0.1)'},
                                        {'range': [75, 100],'color': 'rgba(239, 68, 68, 0.1)'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': "#ffffff"},
                                height=280,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # ------ Engineered Feature Breakdown ------
                        st.markdown("#### Engineered Feature Analysis")
                        st.write("These are the derived signals that feed directly into the model:")

                        tx_hour = int((tx_dt / 3600) % 24)
                        tx_day  = int(tx_dt / 86400)
                        is_weekend = "Yes" if tx_day % 7 >= 5 else "No"
                        amt_log = round(__import__("math").log1p(tx_amt), 3)
                        amt_card_ratio = round(tx_amt / (card1 + 1), 4) if card1 > 0 else 0.0

                        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                        with f_col1:
                            st.metric("Transaction Hour (UTC)", f"{tx_hour}:00")
                            st.metric("Transaction Day", f"Day {tx_day}")
                            st.metric("Is Weekend", is_weekend)
                        with f_col2:
                            st.metric("High Amount (>$500)", "Yes ✓" if tx_amt > 500 else "No")
                            st.metric("Log(1 + Amount)", f"{amt_log}")
                            st.metric("Amt / Card1 Ratio", f"{amt_card_ratio}")
                        with f_col3:
                            st.metric("Email Domains Match", "Yes ✓" if p_email == r_email else "No ✗")
                            st.metric("P-Email Has Digit", "Yes" if any(c.isdigit() for c in p_email) else "No")
                            st.metric("P-Email Unknown", "Yes" if not p_email or p_email == "Unknown" else "No")
                        with f_col4:
                            st.metric("Card Type Unknown", "Yes" if card6 == "Unknown" else "No")
                            st.metric("Device Type", dev_type)

                    else:
                        st.error(f"Error from API: {response.text}")
                except Exception as ex:
                    st.error(f"Network error: Can't connect to API server. {ex}")


# --------------------------------------------------
# TAB 2: Batch Prediction
# --------------------------------------------------

with tab2:
    st.markdown("### Process Transactions in Batch")
    st.write("Upload a raw CSV file containing transaction records. The backend will parse, impute, feature engineer, and run inference in Spark.")

    st.markdown("Don't have a file? Generate a template of rows using the dataset.")
    if st.button("📥 Generate Sample CSV File"):
        try:
            samples_data = fetch_samples()
            if samples_data:
                sample_df = pd.DataFrame(samples_data)
                if "isFraud" in sample_df.columns:
                    sample_df = sample_df.drop(columns=["isFraud"])
                csv_buffer = io.StringIO()
                sample_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Generated CSV Template",
                    data=csv_buffer.getvalue(),
                    file_name="sample_test_transactions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("API offline or sample rows could not be retrieved.")
        except Exception as e:
            st.error(f"Could not construct template: {e}")

    st.markdown("---")

    uploaded_file = st.file_uploader("Choose CSV File to Upload", type=["csv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        raw_df = pd.read_csv(uploaded_file)
        st.markdown(f"**Previewing Raw Upload ({len(raw_df)} Rows):**")
        st.dataframe(raw_df.head(5), use_container_width=True)

        uploaded_file.seek(0)

        if not backend_online:
            st.warning("⚠️ Start the FastAPI backend server to run batch inference.")
        else:
            if st.button("⚡ Run PySpark Batch Prediction", use_container_width=True):
                with st.spinner("Processing in PySpark Engine..."):
                    try:
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                        res_file = requests.post(f"{API_URL}/predict_file", files=files)

                        if res_file.status_code == 200:
                            pred_df = pd.read_csv(io.BytesIO(res_file.content))
                            st.success("Batch Prediction Complete!")

                            st.markdown("### 📊 Inference Analytics")

                            total_tx   = len(pred_df)
                            fraud_cnt  = len(pred_df[pred_df["PredictionLabel"] == "Fraud"])
                            fraud_rate = (fraud_cnt / total_tx) * 100
                            avg_amt    = pred_df["TransactionAmt"].mean()

                            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                            with col_s1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-header">Total Transactions</div>
                                    <div class="metric-value">{total_tx:,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_s2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-header">Fraud Cases</div>
                                    <div class="metric-value-fraud">{fraud_cnt:,}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_s3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-header">Fraud Rate</div>
                                    <div class="metric-value">{fraud_rate:.2f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_s4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-header">Average Value</div>
                                    <div class="metric-value">${avg_amt:.2f}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("#### Visual Insights")
                            p_col1, p_col2 = st.columns(2)

                            with p_col1:
                                pie_data = pred_df["PredictionLabel"].value_counts().reset_index()
                                pie_data.columns = ["Status", "Count"]
                                fig_pie = px.pie(
                                    pie_data,
                                    values="Count",
                                    names="Status",
                                    color="Status",
                                    color_discrete_map={"Legitimate": "#10b981", "Fraud": "#ef4444"},
                                    title="Distribution of Predictions"
                                )
                                fig_pie.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': "#ffffff"}
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                            with p_col2:
                                fig_scatter = px.scatter(
                                    pred_df,
                                    x="TransactionAmt",
                                    y="FraudProbability",
                                    color="PredictionLabel",
                                    color_discrete_map={"Legitimate": "#10b981", "Fraud": "#ef4444"},
                                    title="Transaction Amount vs Fraud Probability",
                                    labels={"TransactionAmt": "Amount ($)", "FraudProbability": "Probability"}
                                )
                                fig_scatter.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': "#ffffff"}
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)

                            st.markdown("#### Prediction Output Details")
                            st.dataframe(pred_df.head(100), use_container_width=True)

                            csv_out_buffer = io.StringIO()
                            pred_df.to_csv(csv_out_buffer, index=False)
                            st.download_button(
                                label="💾 Download Complete Predicted CSV",
                                data=csv_out_buffer.getvalue(),
                                file_name="predictions_output.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error(f"Error from API: {res_file.text}")
                    except KeyError as ke:
                        st.error(f"⚠️ CSV Schema Mismatch: Missing column {ke}.")
                    except Exception as ex:
                        st.error(f"Failed connection to backend API: {ex}")


# --------------------------------------------------
# TAB 3: Model Evaluation Insights (Live from API)
# --------------------------------------------------

with tab3:
    st.markdown("### Model Training & Performance Metrics")

    metrics_data = fetch_metrics()

    if metrics_data is None:
        st.info("📭 No metrics found yet. Run `python main.py` to train the model and generate metrics.")

        st.markdown("""
        Once training is complete, this tab will automatically display:
        - Per-model accuracy, PR-AUC, ROC-AUC
        - **Fraud-class** precision, recall, and F1 (the real signal for imbalanced fraud data)
        - Confusion matrix heatmap per model
        - Training timestamp
        """)
    else:
        trained_at = metrics_data.get("trained_at", "Unknown")
        models_data = metrics_data.get("models", [])

        st.caption(f"📅 Last trained: **{trained_at}** UTC")

        if not models_data:
            st.warning("Metrics file found but no model data inside.")
        else:
            # Find best RF model stats for headline cards
            rf_data = next(
                (m for m in models_data if "Random Forest" in m.get("Model", "")),
                models_data[0]
            )

            # Headline metric cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-header">RF Accuracy</div>
                    <div class="metric-value">{rf_data.get('Accuracy', 0)*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">On Test Split</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-header">PR-AUC (Fraud)</div>
                    <div class="metric-value">{rf_data.get('PR_AUC', 0)*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Precision-Recall AUC</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-header">Fraud Recall</div>
                    <div class="metric-value-fraud">{rf_data.get('FraudRecall', 0)*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">% of Real Fraud Caught</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m4:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-header">Fraud Precision</div>
                    <div class="metric-value-success">{rf_data.get('FraudPrecision', 0)*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">% of Alerts That Are Real</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Full comparison table
            st.markdown("#### All Models — Full Metric Comparison")

            table_rows = []
            for m in models_data:
                table_rows.append({
                    "Model":             m.get("Model"),
                    "Accuracy":          f"{m.get('Accuracy', 0)*100:.2f}%",
                    "ROC-AUC":           f"{m.get('ROC_AUC', 0)*100:.2f}%",
                    "PR-AUC":            f"{m.get('PR_AUC', 0)*100:.2f}%",
                    "Fraud Recall":      f"{m.get('FraudRecall', 0)*100:.2f}%",
                    "Fraud Precision":   f"{m.get('FraudPrecision', 0)*100:.2f}%",
                    "Fraud F1":          f"{m.get('FraudF1', 0)*100:.2f}%",
                    "TP": m.get("TP", 0),
                    "FP": m.get("FP", 0),
                    "FN": m.get("FN", 0),
                    "TN": m.get("TN", 0),
                })

            st.dataframe(
                pd.DataFrame(table_rows).set_index("Model"),
                use_container_width=True
            )

            # Confusion Matrix heatmaps
            st.markdown("#### Confusion Matrix Heatmaps")
            cm_cols = st.columns(min(len(models_data), 2))

            for i, m in enumerate(models_data):
                with cm_cols[i % 2]:
                    tp = m.get("TP", 0)
                    fp = m.get("FP", 0)
                    fn = m.get("FN", 0)
                    tn = m.get("TN", 0)

                    z = [[tn, fp], [fn, tp]]
                    labels = [["TN", "FP"], ["FN", "TP"]]
                    text = [
                        [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
                        [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]
                    ]

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=z,
                        text=text,
                        texttemplate="%{text}",
                        colorscale=[[0, "#1e293b"], [0.5, "#1d4ed8"], [1, "#10b981"]],
                        showscale=False,
                        hoverinfo="text"
                    ))
                    fig_cm.update_layout(
                        title=f"{m.get('Model')}",
                        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Pred: Legit", "Pred: Fraud"], color="#94a3b8"),
                        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["Actual: Legit", "Actual: Fraud"], color="#94a3b8"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30,41,59,0.4)',
                        font={'color': "#ffffff", 'size': 13},
                        height=300,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Pipeline Architecture — Plotly Sankey (replaces broken Mermaid)
    st.markdown("#### ETL & Machine Learning Pipeline Architecture")

    node_labels = [
        "train_transaction.csv",   # 0
        "train_identity.csv",      # 1
        "Merged DataFrame",        # 2
        "Drop High-Null Cols",     # 3
        "Fill Categoricals",       # 4
        "Train/Test Split",        # 5
        "Fit Imputer (Train Only)",# 6
        "Test Set Transform",      # 7
        "Feature Engineering",     # 8
        "Class Weights",           # 9
        "StringIndexer",           # 10
        "VectorAssembler",         # 11
        "Random Forest (CV)",      # 12
        "models/rf_model",         # 13
        "models/tree_pipeline",    # 14
        "models/metrics.json",     # 15
    ]

    # (source, target, value, color)
    links = [
        (0, 2, 4, "rgba(56,189,248,0.4)"),
        (1, 2, 2, "rgba(56,189,248,0.4)"),
        (2, 3, 4, "rgba(129,140,248,0.4)"),
        (3, 4, 3, "rgba(129,140,248,0.4)"),
        (4, 5, 3, "rgba(129,140,248,0.4)"),
        (5, 6, 2, "rgba(251,191,36,0.4)"),
        (5, 7, 1, "rgba(251,191,36,0.4)"),
        (6, 8, 2, "rgba(52,211,153,0.4)"),
        (7, 8, 1, "rgba(52,211,153,0.4)"),
        (8, 9, 2, "rgba(52,211,153,0.4)"),
        (9, 10, 2, "rgba(167,139,250,0.4)"),
        (10, 11, 2, "rgba(167,139,250,0.4)"),
        (11, 12, 2, "rgba(239,68,68,0.4)"),
        (12, 13, 2, "rgba(239,68,68,0.4)"),
        (11, 14, 1, "rgba(148,163,184,0.4)"),
        (12, 15, 1, "rgba(148,163,184,0.4)"),
    ]

    sources = [l[0] for l in links]
    targets = [l[1] for l in links]
    values  = [l[2] for l in links]
    colors  = [l[3] for l in links]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            label=node_labels,
            color=[
                "#38bdf8", "#38bdf8",          # CSVs
                "#818cf8",                       # Merged
                "#a78bfa", "#a78bfa",            # Cleaning steps
                "#fbbf24",                       # Split
                "#34d399", "#34d399",            # Impute
                "#10b981",                       # Features
                "#f59e0b",                       # Weights
                "#8b5cf6", "#8b5cf6",            # Indexer + Assembler
                "#ef4444",                       # RF
                "#f97316", "#94a3b8", "#94a3b8"  # Outputs
            ]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    ))

    fig_sankey.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ffffff", size=11),
        height=500,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig_sankey, use_container_width=True)
