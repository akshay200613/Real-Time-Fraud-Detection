from pyspark.ml.feature import Imputer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when
from typing import Tuple


class DataCleaner:

    def __init__(self, threshold=0.80):
        self.threshold = threshold

    # ----------------------------------
    # Drop columns with high missing values
    # ----------------------------------

    def drop_high_missing_columns(self, df: DataFrame) -> DataFrame:

        total_rows = df.count()

        # Single-pass aggregation — much more efficient than per-column filter+count
        missing = df.select([
            count(
                when(col(c).isNull(), c)
            ).alias(c)
            for c in df.columns
        ]).first().asDict()

        drop_cols = [
            c
            for c, v in missing.items()
            if v / total_rows > self.threshold
        ]

        return df.drop(*drop_cols)

    # ----------------------------------
    # Fill categorical columns
    # ----------------------------------

    def fill_categorical_nulls(self, df: DataFrame) -> DataFrame:

        string_cols = [
            f.name
            for f in df.schema.fields
            if f.dataType.simpleString() == "string"
        ]

        return df.fillna("Unknown", subset=string_cols)

    # ----------------------------------
    # Fit imputer on train, apply to both (prevents data leakage)
    # ----------------------------------

    def fit_transform_numeric(
        self,
        train_df: DataFrame,
        test_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Fits the Imputer ONLY on train_df to prevent leakage,
        then transforms both train_df and test_df.
        """

        numeric_cols = [
            f.name
            for f in train_df.schema.fields
            if (
                f.dataType.simpleString()
                in ("double", "float", "int", "bigint")
                and f.name != "isFraud"
            )
        ]

        if not numeric_cols:
            return train_df, test_df

        imputer = Imputer(
            strategy="median",
            inputCols=numeric_cols,
            outputCols=numeric_cols
        )

        # Fit ONLY on training data
        imputer_model = imputer.fit(train_df)

        return (
            imputer_model.transform(train_df),
            imputer_model.transform(test_df)
        )

    # ----------------------------------
    # Complete Cleaning Pipeline (training phase — before split)
    # Only use for drop + categorical; numeric is done after split.
    # ----------------------------------

    def clean_pre_split(self, df: DataFrame) -> DataFrame:

        df = self.drop_high_missing_columns(df)
        df = self.fill_categorical_nulls(df)

        return df