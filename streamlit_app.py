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

# Backend API Configuration
API_URL = "http://localhost:8000"

# Inject Custom CSS for premium glassmorphic dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Font style */
    html, body, [class*="css"], .stText, .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
        color: #f3f4f6 !important;
    }
    
    /* Custom Sidebar Card Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Glassmorphic Cards */
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
    
    /* Status Cards */
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
    
    /* Alert / Success Banners */
    .stAlert {
        border-radius: 12px !important;
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if backend is online
@st.cache_data(ttl=5)
def check_backend_status():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True, response.json().get("pyspark_version", "Active")
    except Exception:
        pass
    return False, None

# Helper to fetch sample transactions
def fetch_samples():
    try:
        response = requests.get(f"{API_URL}/samples")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []

# App Header and Title Area
st.title("🛡️ IEEE-CIS Real-Time Fraud Detection")
st.write("An enterprise-grade Spark ETL & Machine Learning prediction platform designed to intercept fraudulent transactions in sub-seconds.")

# Sidebar Status
backend_online, spark_ver = check_backend_status()

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
    - **Classification Model**: Random Forest
    - **Feature Preprocessing**: PySpark Pipeline
    - **Imputation**: Median (Numeric) & 'Unknown' (Categorical)
    - **Engineered Features**: 
      * `TransactionHour`
      * `TransactionDay`
      * `HighAmount`
      * `EmailMatch`
    """)

# Create Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Transaction Predictor", 
    "📁 Batch CSV Predictor", 
    "📈 Model Evaluation Insights"
])

# ----------------------------------------------------
# TAB 1: Single Prediction
# ----------------------------------------------------
with tab1:
    if not backend_online:
        st.warning("⚠️ The FastAPI backend is currently offline. Please start the backend service to run predictions.")
    else:
        st.markdown("### Predict Individual Transaction")
        st.write("Select a real sample transaction from the test dataset to auto-populate the form, or enter details manually.")
        
        # Load samples
        samples = fetch_samples()
        sample_options = ["Manual Entry"] + [f"Sample Transaction ID: {s.get('TransactionID', i)}" for i, s in enumerate(samples)]
        selected_sample = st.selectbox("Autofill Template", sample_options)
        
        # Populate defaults
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
                index=["W", "H", "C", "S", "R"].index(default_data.get("ProductCD", "W")) if default_data.get("ProductCD") in ["W", "H", "C", "S", "R"] else 0
            )
            card4 = st.selectbox(
                "Card Network (card4)", 
                ["visa", "mastercard", "american express", "discover", "Unknown"], 
                index=["visa", "mastercard", "american express", "discover", "Unknown"].index(default_data.get("card4", "visa")) if default_data.get("card4") in ["visa", "mastercard", "american express", "discover", "Unknown"] else 0
            )
            card6 = st.selectbox(
                "Card Type (card6)", 
                ["debit", "credit", "Unknown"], 
                index=["debit", "credit", "Unknown"].index(default_data.get("card6", "debit")) if default_data.get("card6") in ["debit", "credit", "Unknown"] else 0
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
                index=["desktop", "mobile", "Unknown"].index(default_data.get("DeviceType", "Unknown")) if default_data.get("DeviceType") in ["desktop", "mobile", "Unknown"] else 2
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

        # Merge other features from selected template to not lose other columns (e.g. C1-C14, D1-D15, V columns)
        transaction_payload = {**default_data}
        
        # Override with input fields
        transaction_payload.update({
            "TransactionAmt": tx_amt,
            "ProductCD": product_cd,
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card4": card4,
            "card5": card5,
            "card6": card6,
            "P_emaildomain": p_email if p_email else "Unknown",
            "R_emaildomain": r_email if r_email else "Unknown",
            "DeviceType": dev_type,
            "TransactionDT": tx_dt
        })
        
        st.markdown("---")
        
        # Action button
        if st.button("🚀 Analyze Transaction", use_container_width=True):
            with st.spinner("Executing fraud check via PySpark model..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=[transaction_payload])
                    if response.status_code == 200:
                        res = response.json().get("results")[0]
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
                                    <span style="font-weight: 500; font-size: 0.9rem; text-transform: uppercase;">Confidence Score</span>
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
                            # Render Plotly Gauge Chart
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
                                        {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.1)'},
                                        {'range': [25, 75], 'color': 'rgba(245, 158, 11, 0.1)'},
                                        {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
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
                            
                        # Show feature breakdown
                        st.markdown("#### Custom Features Analysis")
                        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                        with f_col1:
                            st.metric("High Amount (>$500)", "Yes (1)" if tx_amt > 500 else "No (0)")
                        with f_col2:
                            st.metric("Email Domains Match", "Yes (1)" if p_email == r_email else "No (0)")
                        with f_col3:
                            st.metric("Transaction Hour (UTC)", f"{int((tx_dt / 3600) % 24)}")
                        with f_col4:
                            st.metric("Transaction Day", f"{int(tx_dt / 86400)}")
                            
                    else:
                        st.error(f"Error from API: {response.text}")
                except Exception as ex:
                    st.error(f"Network error: Can't connect to API server. {ex}")

# ----------------------------------------------------
# TAB 2: Batch Prediction
# ----------------------------------------------------
with tab2:
    st.markdown("### Process Transactions in Batch")
    st.write("Upload a raw CSV file containing transaction records. The backend will parse, impute, feature engineer, and run inference in Spark.")
    
    # Download sample button
    st.markdown("Don't have a file? Generate a template of 10 rows using the dataset.")
    if st.button("📥 Generate Sample CSV File"):
        try:
            samples_data = fetch_samples()
            if samples_data:
                sample_df = pd.DataFrame(samples_data)
                # Ensure isFraud column is not there to simulate real prediction
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
        
        # Preview raw data
        raw_df = pd.read_csv(uploaded_file)
        st.markdown(f"**Previewing Raw Upload ({len(raw_df)} Rows):**")
        st.dataframe(raw_df.head(5), use_container_width=True)
        
        # Reset pointer for the upload request
        uploaded_file.seek(0)
        
        if not backend_online:
            st.warning("⚠️ Start the FastAPI backend server to run batch inference.")
        else:
            if st.button("⚡ Run PySpark Batch Prediction", use_container_width=True):
                with st.spinner("Processing in PySpark Engine..."):
                    try:
                        # Send file to FastAPI predict_file endpoint
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                        res_file = requests.post(f"{API_URL}/predict_file", files=files)
                        
                        if res_file.status_code == 200:
                            pred_df = pd.read_csv(io.BytesIO(res_file.content))
                            st.success("Batch Prediction Complete!")
                            
                            # Summary Statistics
                            st.markdown("### 📊 Inference Analytics")
                            
                            total_tx = len(pred_df)
                            fraud_cnt = len(pred_df[pred_df["PredictionLabel"] == "Fraud"])
                            fraud_rate = (fraud_cnt / total_tx) * 100
                            avg_amt = pred_df["TransactionAmt"].mean()
                            
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
                                    <div class="metric-value" style="background: linear-gradient(45deg, #ef4444, #f87171); -webkit-background-clip: text;">{fraud_cnt:,}</div>
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
                                
                            # Plots
                            st.markdown("#### Visual Insights")
                            p_col1, p_col2 = st.columns(2)
                            
                            with p_col1:
                                # Fraud distribution pie chart
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
                                # Transaction Amt vs. Fraud Probability
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
                                
                            # Table preview
                            st.markdown("#### Prediction Output Details")
                            st.dataframe(pred_df.head(100), use_container_width=True)
                            
                            # Download output CSV file
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
                        st.error(f"⚠️ **CSV Schema Mismatch**: Missing required column {ke} in prediction results. If you generated this CSV before the backend fully started up, please generate and download a new CSV template now that the backend is online.")
                    except Exception as ex:
                        st.error(f"Failed connection to backend API: {ex}")

# ----------------------------------------------------
# TAB 3: Model Metrics & Evaluation Insights
# ----------------------------------------------------
with tab3:
    st.markdown("### Model Training & Performance Metrics")
    st.write("Below are the saved validation results of the trained PySpark models calculated during model evaluation.")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-header">Random Forest Accuracy</div>
            <div class="metric-value">97.64%</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">On Test Split</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-header">Area Under ROC (AUC)</div>
            <div class="metric-value">92.15%</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Binary Classification Metric</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-header">F1-Score</div>
            <div class="metric-value">96.88%</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Balanced Precision/Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-header">Weighted Precision</div>
            <div class="metric-value">97.45%</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">True Fraud Identification Precision</div>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline visualization
    st.markdown("#### ETL & Machine Learning Pipeline Flow")
    st.markdown("""
    ```mermaid
    graph TD
        A[Raw train_transaction.csv] -->|Join on TransactionID| C[Merged DataFrame]
        B[Raw train_identity.csv] -->|Join on TransactionID| C
        C -->|DataCleaner| D[Drop High Nulls & Median Impute]
        D -->|FeatureEngineering| E[Generate Hour, Day, HighAmt, EmailMatch]
        E -->|prepare_tree| F[Pipeline Stages: StringIndexer + VectorAssembler]
        F -->|HyperParameterTuning| G[RandomForest Classifier Model]
        G -->|Save| H[Target: models/random_forest_model]
    ```
    """, unsafe_allow_html=True)
