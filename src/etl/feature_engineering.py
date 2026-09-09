"""
feature_engineering.py

Creates domain-driven features from the raw transaction and identity data.
All features are derived from row-level operations (no aggregation across rows),
so this is safe to run AFTER the train/test split.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, log1p, least, greatest,
    regexp_extract, lit
)


class FeatureEngineering:

    def create_features(self, df: DataFrame) -> DataFrame:
        """
        Adds engineered features to the DataFrame.
        """

        df = self._time_features(df)
        df = self._amount_features(df)
        df = self._email_features(df)
        df = self._card_features(df)
        df = self._aggregated_features(df)

        return df

    # ----------------------------------
    # Time Features
    # ----------------------------------

    def _time_features(self, df: DataFrame) -> DataFrame:

        return (
            df
            .withColumn(
                "TransactionHour",
                ((col("TransactionDT") / 3600) % 24).cast("int")
            )
            .withColumn(
                "TransactionDay",
                (col("TransactionDT") / 86400).cast("int")
            )
            .withColumn(
                "IsWeekend",
                when((col("TransactionDT") / 86400).cast("int") % 7 >= 5, 1)
                .otherwise(0)
            )
        )

    # ----------------------------------
    # Amount Features
    # ----------------------------------

    def _amount_features(self, df: DataFrame) -> DataFrame:

        return (
            df
            # Log-transformed amount — reduces skew for LR
            .withColumn(
                "AmtLog",
                log1p(col("TransactionAmt"))
            )
            # Binary high-amount flag — kept for backward compatibility
            .withColumn(
                "HighAmount",
                when(col("TransactionAmt") > 500, 1).otherwise(0)
            )
            # Ratio of amount to card1 (a proxy for card-level spending baseline)
            .withColumn(
                "AmtToCard1Ratio",
                when(
                    col("card1").isNotNull() & (col("card1") > 0),
                    col("TransactionAmt") / (col("card1").cast("double") + 1.0)
                ).otherwise(0.0)
            )
        )

    # ----------------------------------
    # Email Features
    # ----------------------------------

    def _email_features(self, df: DataFrame) -> DataFrame:

        return (
            df
            # Whether purchaser and recipient share the same email domain
            .withColumn(
                "EmailMatch",
                when(
                    col("P_emaildomain") == col("R_emaildomain"),
                    1
                ).otherwise(0)
            )
            # Heuristic: numeric characters in email domain often indicate disposable addresses
            .withColumn(
                "PEmailHasDigit",
                when(
                    regexp_extract(col("P_emaildomain"), r"\d", 0) != "",
                    1
                ).otherwise(0)
            )
            # Flag unknown/missing email domain
            .withColumn(
                "PEmailUnknown",
                when(
                    col("P_emaildomain").isNull() |
                    (col("P_emaildomain") == "Unknown"),
                    1
                ).otherwise(0)
            )
        )

    # ----------------------------------
    # Card Features
    # ----------------------------------

    def _card_features(self, df: DataFrame) -> DataFrame:

        return (
            df
            # Flag transactions where card type is unknown
            .withColumn(
                "CardTypeUnknown",
                when(
                    col("card6").isNull() | (col("card6") == "Unknown"),
                    1
                ).otherwise(0)
            )
        )

    # ----------------------------------
    # Aggregated / C & D Column Features
    # ----------------------------------

    def _aggregated_features(self, df: DataFrame) -> DataFrame:
        """
        Creates aggregate features from the C (count) and D (time-delta) columns.
        Columns are only used if present; missing ones are treated as 0.
        """

        c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        d_cols = ["D1", "D2", "D4", "D10", "D15"]

        # Build safe expressions — use 0 if column not present
        def safe_col(name):
            if name in df.columns:
                return col(name).cast("double")
            return lit(0.0)

        # Sum of C-columns (count-based activity signal)
        c_sum_expr = safe_col(c_cols[0])
        for c in c_cols[1:]:
            c_sum_expr = c_sum_expr + safe_col(c)

        # Minimum of D-columns (shortest recency — high value = suspicious)
        d_exprs = [safe_col(d) for d in d_cols]
        d_min_expr = greatest(*d_exprs) if len(d_exprs) > 1 else d_exprs[0]

        return (
            df
            .withColumn("CSum", c_sum_expr)
            .withColumn("DMax", d_min_expr)
        )