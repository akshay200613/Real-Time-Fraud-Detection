from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col, count, when


class DataCleaner:

    def __init__(self, threshold=0.80):
        self.threshold = threshold

    # ----------------------------------
    # Drop columns with high missing values
    # ----------------------------------

    def drop_high_missing_columns(self, df):

        total_rows = df.count()

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
    # Fill numeric columns
    # ----------------------------------

    def fill_numeric_nulls(self, df):

        numeric_cols = [

            f.name

            for f in df.schema.fields

            if (
                f.dataType.simpleString()
                in ("double", "float", "int", "bigint")
                and f.name != "isFraud"
            )

        ]

        if not numeric_cols:
            return df

        imputer = Imputer(

            strategy="median",

            inputCols=numeric_cols,

            outputCols=numeric_cols

        )

        return imputer.fit(df).transform(df)

    # ----------------------------------
    # Fill categorical columns
    # ----------------------------------

    def fill_categorical_nulls(self, df):

        string_cols = [

            f.name

            for f in df.schema.fields

            if f.dataType.simpleString() == "string"

        ]

        return df.fillna(
            "Unknown",
            subset=string_cols
        )

    # ----------------------------------
    # Complete Cleaning Pipeline
    # ----------------------------------

    def clean(self, df):

        df = self.drop_high_missing_columns(df)

        df = self.fill_numeric_nulls(df)

        df = self.fill_categorical_nulls(df)

        return df