"""
prepare_data.py

Prepares features for:
  - Logistic Regression: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
  - Tree Models:         StringIndexer → VectorAssembler (trees don't need scaling)
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)


class DataPreparation:

    def __init__(self):

        self.label = "isFraud"

        self.exclude = {
            "TransactionID",
            "TransactionDT",
            "classWeight",
            self.label
        }

        # Categorical columns for model encoding.
        # Includes low-cardinality transactional + email domain + identity columns.
        # All use StringIndexer(handleInvalid="keep") so unseen values at
        # inference time are handled gracefully.
        self.categorical_cols = [
            # Core transaction categoricals
            "ProductCD",
            "card4",
            "card6",
            # Match columns (binary text flags)
            "M1", "M2", "M3", "M4", "M5",
            "M6", "M7", "M8", "M9",
            # Device
            "DeviceType",
            # Email domains — strong fraud signal
            "P_emaildomain",
            "R_emaildomain",
            # Identity columns (low-to-medium cardinality)
            "id_12", "id_15", "id_16",
            "id_23", "id_27", "id_28",
            "id_29", "id_31", "id_35",
            "id_36", "id_37", "id_38",
        ]

    def train_test_split(
        self,
        df,
        train_size=0.8,
        seed=42
    ):
        return df.randomSplit(
            [train_size, 1 - train_size],
            seed
        )

    def _numeric_columns(self, df):

        return [
            field.name
            for field in df.schema.fields
            if (
                field.dataType.simpleString()
                in ("double", "float", "int", "bigint")
                and field.name not in self.exclude
            )
        ]

    def _categorical_columns(self, df):
        """Returns only the categorical columns that actually exist in df."""
        return [
            column
            for column in self.categorical_cols
            if column in df.columns
        ]

    # -----------------------------
    # Logistic Regression Pipeline
    # StringIndexer → OHE → VectorAssembler → StandardScaler
    # -----------------------------

    def prepare_lr(self, train_df, test_df):

        categorical_cols = self._categorical_columns(train_df)
        numeric_cols = self._numeric_columns(train_df)

        indexers = [
            StringIndexer(
                inputCol=c,
                outputCol=f"{c}_idx",
                handleInvalid="keep"
            )
            for c in categorical_cols
        ]

        encoder = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in categorical_cols],
            outputCols=[f"{c}_vec" for c in categorical_cols]
        )

        assembler = VectorAssembler(
            inputCols=numeric_cols +
                      [f"{c}_vec" for c in categorical_cols],
            outputCol="features_raw",
            handleInvalid="keep"
        )

        # StandardScaler — critical for LR convergence with mixed-scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=False,    # sparse vector safety (OHE produces sparse)
            withStd=True
        )

        pipeline = Pipeline(
            stages=indexers + [encoder, assembler, scaler]
        )

        pipeline_model = pipeline.fit(train_df)

        return (
            pipeline_model.transform(train_df),
            pipeline_model.transform(test_df),
            pipeline_model
        )

    # -----------------------------
    # Tree Models Pipeline
    # StringIndexer → VectorAssembler
    # (Trees don't require scaling; skip OHE and StandardScaler)
    # -----------------------------

    def prepare_tree(self, train_df, test_df):

        categorical_cols = self._categorical_columns(train_df)
        numeric_cols = self._numeric_columns(train_df)

        indexers = [
            StringIndexer(
                inputCol=c,
                outputCol=f"{c}_idx",
                handleInvalid="keep"
            )
            for c in categorical_cols
        ]

        assembler = VectorAssembler(
            inputCols=numeric_cols +
                      [f"{c}_idx" for c in categorical_cols],
            outputCol="features",
            handleInvalid="keep"
        )

        pipeline = Pipeline(
            stages=indexers + [assembler]
        )

        pipeline_model = pipeline.fit(train_df)

        return (
            pipeline_model.transform(train_df),
            pipeline_model.transform(test_df),
            pipeline_model
        )