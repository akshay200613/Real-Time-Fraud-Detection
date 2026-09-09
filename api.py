import os
import sys
import io
import json
import tempfile
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

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


# --------------------------------------------------
# Input Validation Schema
# --------------------------------------------------

class TransactionInput(BaseModel):
    """
    Pydantic model for single transaction prediction input.
    All fields have safe defaults so the API won't crash on partial input.
    """
    TransactionAmt: float = 50.0
    TransactionDT:  int   = 18403200
    ProductCD:      Optional[str]   = "W"
    card1:          Optional[float] = None
    card2:          Optional[float] = None
    card3:          Optional[float] = None
    card4:          Optional[str]   = "Unknown"
    card5:          Optional[float] = None
    card6:          Optional[str]   = "Unknown"
    P_emaildomain:  Optional[str]   = "Unknown"
    R_emaildomain:  Optional[str]   = "Unknown"
    DeviceType:     Optional[str]   = "Unknown"

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("TransactionAmt must be greater than 0")
        return v

    @field_validator("TransactionDT")
    @classmethod
    def dt_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("TransactionDT must be >= 0")
        return v

    model_config = {"extra": "allow"}   # Pass-through unknown fields for template rows


# --------------------------------------------------
# Lifespan: Initialize Spark + load models on startup
# --------------------------------------------------

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

    # Load default template and samples from test set (50 rows for variety)
    try:
        raw_dir = os.path.join("data", "raw")
        trans_path = os.path.join(raw_dir, "test_transaction.csv")
        ident_path = os.path.join(raw_dir, "test_identity.csv")

        print("Fast-loading samples using Pandas...")
        trans_pdf = pd.read_csv(trans_path, nrows=50)

        if os.path.exists(ident_path):
            ident_pdf = pd.read_csv(ident_path, nrows=50)
            merged_pdf = pd.merge(trans_pdf, ident_pdf, on="TransactionID", how="left")
        else:
            merged_pdf = trans_pdf

        # Rename columns containing '-' to '_'
        merged_pdf.columns = [c.replace("-", "_") for c in merged_pdf.columns]

        # Fill nulls
        for col_name in merged_pdf.columns:
            if pd.api.types.is_numeric_dtype(merged_pdf[col_name]):
                merged_pdf[col_name] = merged_pdf[col_name].fillna(0.0)
            else:
                merged_pdf[col_name] = merged_pdf[col_name].fillna("Unknown")

        # Convert to Spark DataFrame for type consistency
        cleaned_spark = spark.createDataFrame(merged_pdf)
        rows = cleaned_spark.collect()

        if rows:
            default_row_dict = rows[0].asDict()
            sample_transactions = [r.asDict() for r in rows]
            print(f"Successfully cached {len(sample_transactions)} sample rows.")
        else:
            default_row_dict = {}
            sample_transactions = []
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Could not load sample template. Using empty defaults. Error: {e}")
        default_row_dict = {}
        sample_transactions = []

    yield

    # Shutdown
    if spark:
        print("Stopping PySpark session...")
        spark.stop()


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="FastAPI Backend for PySpark ML Fraud Detection Model",
    version="2.0.0",
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


# --------------------------------------------------
# Input preparation helper
# --------------------------------------------------

def prepare_input_dataframe(raw_data_dict_list: list):
    """
    Prepares raw input data for prediction by aligning with the expected model schema,
    imputing missing columns/values with default values, and running feature engineering.
    """
    pdf = pd.DataFrame(raw_data_dict_list)

    # Rename columns containing '-' to '_'
    pdf.columns = [c.replace("-", "_") for c in pdf.columns]

    new_cols = {}

    # 1. Align categorical columns
    for col_name in expected_categorical:
        if col_name not in pdf.columns:
            new_cols[col_name] = "Unknown"
        else:
            pdf[col_name] = pdf[col_name].fillna("Unknown")

    # 2. Align numerical columns
    engineered_cols = {
        "TransactionHour", "TransactionDay", "HighAmount", "EmailMatch",
        "IsWeekend", "AmtLog", "AmtToCard1Ratio",
        "PEmailHasDigit", "PEmailUnknown", "CardTypeUnknown",
        "CSum", "DMax"
    }
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

    # Ensure email domain columns are present for feature engineering
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

    if new_cols:
        new_cols_df = pd.DataFrame(new_cols, index=pdf.index)
        pdf = pd.concat([pdf, new_cols_df], axis=1)

    if "TransactionID" not in pdf.columns:
        pdf["TransactionID"] = range(1000000, 1000000 + len(pdf))

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(pdf)

    # Cast categorical columns to String
    from pyspark.sql.functions import col as spark_col
    string_cols = expected_categorical + ["P_emaildomain", "R_emaildomain"]
    for col_name in string_cols:
        if col_name in spark_df.columns:
            spark_df = spark_df.withColumn(col_name, spark_col(col_name).cast("string"))

    # Run Feature Engineering
    from src.etl.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    spark_df = fe.create_features(spark_df)

    return spark_df


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Real-Time Fraud Detection API is running.",
        "pyspark_version": spark.version if spark else "Not Initialized",
        "api_version": "2.0.0"
    }


@app.get("/schema")
def get_schema():
    if not spark:
        raise HTTPException(status_code=503, detail="Spark Session not ready")
    return {
        "categorical_features": expected_categorical,
        "numeric_features": expected_numeric,
        "default_template": {
            k: v
            for k, v in default_row_dict.items()
            if k in expected_numeric or k in expected_categorical
        }
    }


@app.get("/samples")
def get_samples():
    if not sample_transactions:
        raise HTTPException(status_code=404, detail="No samples available")

    cleaned_samples = []
    for s in sample_transactions:
        cleaned_sample = {}
        for k, v in s.items():
            if isinstance(v, float) and (pd.isna(v) or v != v):
                cleaned_sample[k] = None
            else:
                cleaned_sample[k] = v
        cleaned_samples.append(cleaned_sample)

    return cleaned_samples


@app.get("/metrics")
def get_metrics():
    """
    Returns the evaluation metrics from the most recent training run.
    Metrics are saved to models/metrics.json by main.py after training.
    """
    metrics_path = os.path.join("models", "metrics.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="No metrics file found. Run main.py to train the model first."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predict(transactions: List[TransactionInput]):
    """
    Single or multi-transaction prediction endpoint.
    Input is validated via Pydantic before reaching Spark.
    """
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Models or Spark Session not loaded")

    try:
        # Convert Pydantic models to dicts (extra fields allowed via model_config)
        raw_dicts = [t.model_dump() for t in transactions]

        spark_df = prepare_input_dataframe(raw_dicts)

        transformed_df = pipeline.transform(spark_df)
        predictions = model.transform(transformed_df)

        results = predictions.select("TransactionID", "probability", "prediction").collect()

        output = []
        for r in results:
            prob_val = float(r["probability"][1]) if r["probability"] is not None else 0.0
            pred_val = int(r["prediction"]) if r["prediction"] is not None else 0
            label_val = "Fraud" if pred_val == 1 else "Legitimate"

            output.append({
                "TransactionID":   int(r["TransactionID"]),
                "FraudProbability": round(prob_val, 4),
                "Prediction":      pred_val,
                "PredictionLabel": label_val
            })

        return {"status": "success", "results": output}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Batch prediction endpoint that accepts a CSV file.
    Uses a unique temp file per request to prevent concurrent-request conflicts.
    """
    if not spark or not model:
        raise HTTPException(status_code=503, detail="Models or Spark Session not loaded")

    # Use NamedTemporaryFile to avoid race conditions on concurrent requests
    tmp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    temp_path = tmp_file.name

    try:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_file.close()

        # Load in Spark
        spark_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(temp_path)
        )

        # Rename columns
        spark_df = spark_df.toDF(*[c.replace("-", "_") for c in spark_df.columns])

        from pyspark.sql.functions import col as spark_col, lit
        from pyspark.sql.functions import monotonically_increasing_id

        # Cast string columns
        string_cols = expected_categorical + ["P_emaildomain", "R_emaildomain"]
        for col_name in string_cols:
            if col_name in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, spark_col(col_name).cast("string"))

        # Align categorical columns
        for col_name in expected_categorical:
            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit("Unknown"))
            else:
                spark_df = spark_df.fillna("Unknown", subset=[col_name])

        # Align numerical columns
        engineered_cols = {
            "TransactionHour", "TransactionDay", "HighAmount", "EmailMatch",
            "IsWeekend", "AmtLog", "AmtToCard1Ratio",
            "PEmailHasDigit", "PEmailUnknown", "CardTypeUnknown",
            "CSum", "DMax"
        }
        base_numeric_cols = [c for c in expected_numeric if c not in engineered_cols]

        for col_name in ["TransactionDT", "TransactionAmt"]:
            default = 0 if col_name == "TransactionDT" else 0.0
            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit(default))
            else:
                spark_df = spark_df.fillna(default, subset=[col_name])

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
                except (ValueError, TypeError):
                    default_val = 0.0

            if col_name not in spark_df.columns:
                spark_df = spark_df.withColumn(col_name, lit(default_val))
            else:
                spark_df = spark_df.fillna(default_val, subset=[col_name])

        if "TransactionID" not in spark_df.columns:
            spark_df = spark_df.withColumn(
                "TransactionID", monotonically_increasing_id() + 1000000
            )

        # Feature Engineering
        from src.etl.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        spark_df = fe.create_features(spark_df)

        # Predict
        transformed_df = pipeline.transform(spark_df)
        predictions = model.transform(transformed_df)

        pred_pd = predictions.select("TransactionID", "probability", "prediction").toPandas()
        pred_pd["FraudProbability"] = pred_pd["probability"].apply(
            lambda v: float(v[1]) if v is not None else 0.0
        )
        pred_pd["PredictionLabel"] = pred_pd["prediction"].apply(
            lambda p: "Fraud" if p == 1.0 else "Legitimate"
        )

        # Merge predictions back to original data
        pdf = pd.read_csv(temp_path)
        pdf["TransactionID"] = pdf["TransactionID"].astype(pred_pd["TransactionID"].dtype)

        pred_pd_subset = pred_pd[["TransactionID", "FraudProbability", "PredictionLabel"]]
        result_pdf = pd.merge(pdf, pred_pd_subset, on="TransactionID", how="left")

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
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
