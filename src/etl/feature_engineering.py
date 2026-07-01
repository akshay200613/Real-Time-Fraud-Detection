from pyspark.sql.functions import col, when


class FeatureEngineering:

    def create_features(self, df):

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
                "HighAmount",
                when(col("TransactionAmt") > 500, 1).otherwise(0)
            )

            .withColumn(
                "EmailMatch",
                when(
                    col("P_emaildomain") ==
                    col("R_emaildomain"),
                    1
                ).otherwise(0)
            )

        )