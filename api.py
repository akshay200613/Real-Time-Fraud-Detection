import os
import sys
import io
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global variables to hold Spark, models, and metadata
spark = None
pipeline = None
model = None
default_row_dict = {}
sample_transactions = []
expected_categorical = []
expected_numeric = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark, pipeline, model, default_row_dict, sample_transactions, expected_categorical, expected_numeric
    
    print("Initializing PySpark session and loading models...")
    from src.spark_session import create_spark_session
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import RandomForestClassificationModel
    
    # Create Spark Session
    spark = create_spark_session()
    
    # Load Models
    pipeline = PipelineModel.load("models/tree_pipeline")
    model = RandomForestClassificationModel.load("models/random_forest_model")
    
    # Inspect Pipeline stages to extract expected schema dynamically
    stages = pipeline.stages
    expected_categorical = [s.getInputCol() for s in stages[:-1]]
    assembler_inputs = stages[-1].getInputCols()
    expected_numeric = [col for col in assembler_inputs if not col.endswith("_idx")]
    
    print(f"Loaded Pipeline. Categorical Features ({len(expected_categorical)}): {expected_categorical}")
    print(f"Numeric Features ({len(expected_numeric)}): {expected_numeric}")
    
    # Load default template and samples from test set
    try:
        # Fast load using pandas
        raw_dir = os.path.join("data", "raw")
        trans_path = os.path.join(raw_dir, "test_transaction.csv")
        ident_path = os.path.join(raw_dir, "test_identity.csv")
        
        print("Fast-loading samples using Pandas...")
        trans_pdf = pd.read_csv(trans_path, nrows=20)
        
        if os.path.exists(ident_path):
            ident_pdf = pd.read_csv(ident_path, nrows=20)
            merged_pdf = pd.merge(trans_pdf, ident_pdf, on="TransactionID", how="left")
        else:
            merged_pdf = trans_pdf
            
        # Rename columns containing '-' to '_' (matches Spark naming standard)
        merged_pdf.columns = [c.replace("-", "_") for c in merged_pdf.columns]
        
        # Clean sample using Pandas directly (prevents PySpark Imputer crash on 100% null columns in 20-row slice)
        for col_name in merged_pdf.columns:
            if pd.api.types.is_numeric_dtype(merged_pdf[col_name]):
                merged_pdf[col_name] = merged_pdf[col_name].fillna(0.0)
            else:
                merged_pdf[col_name] = merged_pdf[col_name].fillna("Unknown")
                
        # Convert to Spark DataFrame
        cleaned_spark = spark.createDataFrame(merged_pdf)
        rows = cleaned_spark.collect()
        
        if rows:
            default_row_dict = rows[0].asDict()
            sample_transactions = [r.asDict() for r in rows]
            print(f"Successfully cached {len(sample_transactions)} sample rows using fast Pandas loading.")
        else:
            default_row_dict = {}
            sample_transactions = []
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Warning: Could not load sample template from test dataset. Using empty defaults. Error:", e)
        default_row_dict = {}
        sample_transactions = []
        
    yield
    
    # Shutdown Spark Session
    if spark:
        print("Stopping PySpark session...")
        spark.stop()

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="FastAPI Backend for PySpark ML Fraud Detection Model",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def prepare_input_dataframe(raw_data_dict_list: list):
    """
    Prepares raw input data for prediction by aligning with the expected model schema,
    imputing missing columns/values with default values, and running feature engineering.
    """
    import pandas as pd
    pdf = pd.DataFrame(raw_data_dict_list)
    
    # Rename columns containing '-' to '_' (matches Spark naming standard in extract.py)
    pdf.columns = [c.replace("-", "_") for c in pdf.columns]
    
    # Pre-build dictionary of missing columns to concatenate at once (prevents fragmentation)
    new_cols = {}
    
    # 1. Align categorical columns
    for col_name in expected_categorical:
        if col_name not in pdf.columns:
            new_cols[col_name] = "Unknown"
        else:
            pdf[col_name] = pdf[col_name].fillna("Unknown")
            
    # 2. Align numerical columns
    engineered_cols = {"TransactionHour", "TransactionDay", "HighAmount", "EmailMatch"}
    base_numeric_cols = [c for c in expected_numeric if c not in engineered_cols]
    
    # Ensure TransactionDT is present for Feature Engineering
    if "TransactionDT" not in pdf.columns:
        new_cols["TransactionDT"] = 0
    else:
        pdf["TransactionDT"] = pdf["TransactionDT"].fillna(0)
        
    # Ensure TransactionAmt is present
    if "TransactionAmt" not in pdf.columns:
        new_cols["TransactionAmt"] = 0.0
    else:
        pdf["TransactionAmt"] = pdf["TransactionAmt"].fillna(0.0)
        
    # Ensure P_emaildomain and R_emaildomain are present for EmailMatch
    for domain_col in ["P_emaildomain", "R_emaildomain"]:
        if domain_col not in pdf.columns:
            new_cols[domain_col] = "Unknown"
        else:
            pdf[domain_col] = pdf[domain_col].fillna("Unknown")
            
    # Fill remaining base numeric columns with template values
    for col_name in base_numeric_cols:
        default_val = default_row_dict.get(col_name, 0.0)
        if default_val is None or (isinstance(default_val, float) and default_val != default_val):
            default_val = 0.0
            
        if col_name not in pdf.columns:
            new_cols[col_name] = default_val
        else:
            pdf[col_name] = pdf[col_name].fillna(default_val)
            
    # Concatenate all new columns at once to prevent fragmentation
    if new_cols:
        new_cols_df = pd.DataFrame(new_cols, index=pdf.index)
        pdf = pd.concat([pdf, new_cols_df], axis=1)
        
    # Ensure TransactionID is present
    if "TransactionID" not in pdf.columns:
        pdf["TransactionID"] = range(1000000, 1000000 + len(pdf))
        
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(pdf)
    
    # Cast categorical and email domain columns to String to prevent type mismatch
    from pyspark.sql.functions import col
    string_cols = expected_categorical + ["P_emaildomain", "R_emaildomain"]
    for col_name in string_cols:
        if col_name in spark_df.columns:
            spark_df = spark_df.withColumn(col_name, col(col_name).cast("string"))
            
    # Run Feature Engineering
    from src.etl.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    spark_df = fe.create_features(spark_df)
    
    return spark_df

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Real-Time Fraud Detection API is running.",
        "pyspark_version": spark.version if spark else "Not Initialized"
    }

@app.get("/schema")
def get_schema():
    if not spark:
        raise HTTPException(status_code=503, detail="Spark Session not ready")
    return {
        "categorical_features": expected_categorical,
        "numeric_features": expected_numeric,
        "default_template": {k: v for k, v in default_row_dict.items() if k in expected_numeric or k in expected_categorical}
    }

@app.get("/samples")
def get_samples():
    if not sample_transactions:
        raise HTTPException(status_code=404, detail="No samples available")
    
    # Clean up samples to send to frontend
    cleaned_samples = []
    for s in sample_transactions:
        cleaned_sample = {}
        for k, v in s.items():
            # Check for float NaN/None and convert to json friendly formats
            if isinstance(v, float) and (pd.isna(v) or v != v):
                cleaned_sample[k] = None
            else:
                cleaned_sample[k] = v
        cleaned_samples.append(cleaned_sample)
        
    return cleaned_samples

@app.post("/predict")
def predict(transactions: list[dict]):
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Models or Spark Session not loaded")
    
    try:
        spark_df = prepare_input_dataframe(transactions)
        
        # Transform through Pipeline
        transformed_df = pipeline.transform(spark_df)
        
        # Predict using RF model
        predictions = model.transform(transformed_df)
        
        results = predictions.select("TransactionID", "probability", "prediction").collect()
        
        output = []
        for r in results:
            prob_val = float(r["probability"][1]) if r["probability"] is not None else 0.0
            pred_val = int(r["prediction"]) if r["prediction"] is not None else 0
            label_val = "Fraud" if pred_val == 1 else "Legitimate"
            
            output.append({
                "TransactionID": int(r["TransactionID"]),
                "FraudProbability": round(prob_val, 4),
                "Prediction": pred_val,
                "PredictionLabel": label_val
            })
            
        return {"status": "success", "results": output}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Models or Spark Session not loaded")
    
    temp_in = "temp_upload.csv"
    try:
        # Save uploaded file
        contents = await file.read()
        with open(temp_in, "wb") as f:
            f.write(contents)
            
        # Load native in Spark
        spark_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(temp_in)
        )
        
        # Rename columns to replace '-' with '_'
        spark_df = spark_df.toDF(*[c.replace("-", "_") for c in spark_df.columns])
        
        # Cast categorical and email domain columns to String to prevent type mismatch if empty
        from pyspark.sql.functions import col
        string_cols = expected_categorical + ["P_emaildomain", "R_emaildomain"]
        for col_name in string_cols:
            if col_name in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, col(col_name).cast("string"))
                
        from pyspark.sql.functions import lit
        
        # Align categorical columns
        for col_name in expected_categorical:
            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit("Unknown"))
            else:
                spark_df = spark_df.fillna("Unknown", subset=[col_name])
                
        # Align numerical columns
        engineered_cols = {"TransactionHour", "TransactionDay", "HighAmount", "EmailMatch"}
        base_numeric_cols = [c for c in expected_numeric if c not in engineered_cols]
        
        if "TransactionDT" not in spark_df.columns:
            spark_df = spark_df.withColumn("TransactionDT", lit(0))
        else:
            spark_df = spark_df.fillna(0, subset=["TransactionDT"])
            
        if "TransactionAmt" not in spark_df.columns:
            spark_df = spark_df.withColumn("TransactionAmt", lit(0.0))
        else:
            spark_df = spark_df.fillna(0.0, subset=["TransactionAmt"])
            
        for domain_col in ["P_emaildomain", "R_emaildomain"]:
            if domain_col not in spark_df.columns:
                spark_df = spark_df.withColumn(domain_col, lit("Unknown"))
            else:
                spark_df = spark_df.fillna("Unknown", subset=[domain_col])
                
        for col_name in base_numeric_cols:
            default_val = default_row_dict.get(col_name, 0.0)
            if default_val is None or (isinstance(default_val, float) and default_val != default_val):
                default_val = 0.0
            else:
                try:
                    default_val = float(default_val)
                except ValueError:
                    default_val = 0.0
                
            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit(default_val))
            else:
                spark_df = spark_df.fillna(default_val, subset=[col_name])
                
        if "TransactionID" not in spark_df.columns:
            from pyspark.sql.functions import monotonically_increasing_id
            spark_df = spark_df.withColumn("TransactionID", monotonically_increasing_id() + 1000000)
            
        # Run Feature Engineering
        from src.etl.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        spark_df = fe.create_features(spark_df)
        
        # Predict
        transformed_df = pipeline.transform(spark_df)
        predictions = model.transform(transformed_df)
        
        # Convert prediction outputs to pandas containing raw fields
        pred_pd = predictions.select("TransactionID", "probability", "prediction").toPandas()
        
        # Extract Vector elements natively in Pandas (bypasses Spark UDT extraction error)
        pred_pd["FraudProbability"] = pred_pd["probability"].apply(lambda v: float(v[1]) if v is not None else 0.0)
        pred_pd["PredictionLabel"] = pred_pd["prediction"].apply(lambda p: "Fraud" if p == 1.0 else "Legitimate")
        
        # Merge predictions back to original dataframe
        pdf = pd.read_csv(temp_in)
        pdf["TransactionID"] = pdf["TransactionID"].astype(pred_pd["TransactionID"].dtype)
        
        # Keep only required outputs for joining to original DataFrame
        pred_pd_subset = pred_pd[["TransactionID", "FraudProbability", "PredictionLabel"]]
        result_pdf = pd.merge(pdf, pred_pd_subset, on="TransactionID", how="left")
        
        # Write to memory buffer
        stream = io.StringIO()
        result_pdf.to_csv(stream, index=False)
        
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File Processing Error: {str(e)}")
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass
