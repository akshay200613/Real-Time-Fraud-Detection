from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)


class DataPreparation:

    def __init__(self):

        self.label = "isFraud"

        self.exclude = {
            "TransactionID",
            "TransactionDT",
            self.label
        }

        # Low-cardinality categorical columns
        self.categorical_cols = [
            "ProductCD",
            "card4",
            "card6",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "DeviceType"
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

        return [
            column
            for column in self.categorical_cols
            if column in df.columns
        ]

    # -----------------------------
    # Logistic Regression Pipeline
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
            outputCol="features",
            handleInvalid="keep"
        )

        pipeline = Pipeline(
            stages=indexers + [encoder, assembler]
        )

        pipeline_model = pipeline.fit(train_df)

        return (
            pipeline_model.transform(train_df),
            pipeline_model.transform(test_df),
            pipeline_model
        )

    # -----------------------------
    # Tree Models Pipeline
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