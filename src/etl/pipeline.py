"""
pipeline.py

Orchestrates the full ETL flow:
  1. Extract & merge raw CSVs
  2. Drop high-null columns + fill categorical nulls (pre-split — safe)
  3. Train/Test split
  4. Fit numeric imputer on train only, transform both (no leakage)
  5. Feature engineering (row-level, safe after split)
  6. Compute class weights from training set
  7. Prepare features for LR and tree-based models
"""

from pyspark.sql.functions import col, when

from src.etl.extract import Extract
from src.etl.cleaning import DataCleaner
from src.etl.feature_engineering import FeatureEngineering
from src.ml.prepare_data import DataPreparation
from src.utils.logger import logger


class ETLPipeline:

    def __init__(self, spark):

        self.spark = spark
        self.extract = Extract(spark)
        self.cleaner = DataCleaner()
        self.feature = FeatureEngineering()
        self.prepare = DataPreparation()

    def run(self):

        # ----------------------------------
        # 1. Extract
        # ----------------------------------

        logger.info("Step 1: Extracting and merging train data")
        df = self.extract.merge_train()

        # ----------------------------------
        # 2. Pre-split cleaning (safe steps only)
        # ----------------------------------

        logger.info("Step 2: Pre-split cleaning (drop high-null cols, fill categoricals)")
        df = self.cleaner.clean_pre_split(df)

        # ----------------------------------
        # 3. Train / Test Split (BEFORE numeric imputation)
        # ----------------------------------

        logger.info("Step 3: Splitting into train/test before imputation")
        train_df, test_df = self.prepare.train_test_split(df)

        logger.info(f"  Train size: {train_df.count()} | Test size: {test_df.count()}")

        # ----------------------------------
        # 4. Numeric imputation (fit on train only — no leakage)
        # ----------------------------------

        logger.info("Step 4: Fitting numeric imputer on train only (leakage-safe)")
        train_df, test_df = self.cleaner.fit_transform_numeric(train_df, test_df)

        # ----------------------------------
        # 5. Feature Engineering (row-level — safe after split)
        # ----------------------------------

        logger.info("Step 5: Creating engineered features")
        train_df = self.feature.create_features(train_df)
        test_df  = self.feature.create_features(test_df)

        # ----------------------------------
        # 6. Compute class weights from training set (handles imbalance)
        # ----------------------------------

        logger.info("Step 6: Computing class weights to handle imbalance")
        fraud_count  = train_df.filter(col("isFraud") == 1).count()
        total_count  = train_df.count()
        legit_count  = total_count - fraud_count

        fraud_rate = fraud_count / total_count
        logger.info(f"  Fraud rate: {fraud_rate*100:.2f}% ({fraud_count} / {total_count})")

        if fraud_count > 0 and legit_count > 0:
            weight_fraud = total_count / (2.0 * fraud_count)
            weight_legit = total_count / (2.0 * legit_count)
        else:
            weight_fraud = 1.0
            weight_legit = 1.0

        logger.info(f"  Weight (fraud): {weight_fraud:.4f} | Weight (legit): {weight_legit:.4f}")

        train_df = train_df.withColumn(
            "classWeight",
            when(col("isFraud") == 1, weight_fraud).otherwise(weight_legit)
        )

        # ----------------------------------
        # 7. Prepare features for LR and tree models
        # ----------------------------------

        logger.info("Step 7: Building feature pipelines for LR and tree models")

        lr_train, lr_test, _ = self.prepare.prepare_lr(train_df, test_df)

        tree_train, tree_test, tree_pipeline = self.prepare.prepare_tree(
            train_df, test_df
        )

        logger.info("ETL Pipeline complete.")

        return (
            lr_train,
            lr_test,
            tree_train,
            tree_test,
            tree_pipeline
        )