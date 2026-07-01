"""
extract.py

Handles data extraction and profiling for the
IEEE-CIS Fraud Detection dataset.
"""

from logging import error
import os
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from src.config import RAW_DATA
from src.utils.logger import logger


class Extract:
    """
    Extracts datasets and performs initial profiling.
    """

    def __init__(self, spark):

        self.spark = spark

    # --------------------------------------------------

    def read_csv(self, file_name: str) -> DataFrame:
        """
        Reads a CSV file using Spark.
        """

        path = os.path.join(RAW_DATA, file_name)

        logger.info(f"Reading {file_name}")

        try:

            df = (
                self.spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(path)
            )

            # Rename all columns containing '-' to '_'
            df = df.toDF(*[
                column.replace("-", "_")
                for column in df.columns
            ])

            logger.info(f"{file_name} loaded successfully")

            return df

        except Exception as error:

            logger.error(error)

            raise error

    # --------------------------------------------------

    def load_train_transaction(self):

        return self.read_csv("train_transaction.csv")

    # --------------------------------------------------

    def load_train_identity(self):

        return self.read_csv("train_identity.csv")

    # --------------------------------------------------

    def load_test_transaction(self):

        return self.read_csv("test_transaction.csv")

    # --------------------------------------------------

    def load_test_identity(self):

        return self.read_csv("test_identity.csv")

    # --------------------------------------------------

    def merge_train(self) -> DataFrame:
        """
        Merge training datasets.
        """

        logger.info("Merging training datasets")

        transaction = self.load_train_transaction()

        identity = self.load_train_identity()

        return (
            transaction.join(
                identity,
                on="TransactionID",
                how="left"
            )
        )

    # --------------------------------------------------

    def merge_test(self) -> DataFrame:
        """
        Merge testing datasets.
        """

        logger.info("Merging testing datasets")

        transaction = self.load_test_transaction()

        identity = self.load_test_identity()

        return (
            transaction.join(
                identity,
                on="TransactionID",
                how="left"
            )
        )

    # --------------------------------------------------

    @staticmethod
    def shape(df: DataFrame) -> Tuple[int, int]:
        """
        Returns rows and columns.
        """

        return df.count(), len(df.columns)

    # --------------------------------------------------

    @staticmethod
    def schema(df: DataFrame):

        df.printSchema()

    # --------------------------------------------------

    @staticmethod
    def summary(df: DataFrame):

        df.describe().show()

    # --------------------------------------------------

    @staticmethod
    def duplicate_count(df: DataFrame):

        duplicates = df.count() - df.dropDuplicates().count()

        logger.info(f"Duplicate rows : {duplicates}")

        print(f"Duplicate Rows : {duplicates}")

    # --------------------------------------------------

    @staticmethod
    def missing_report(df: DataFrame):

        logger.info("Generating Missing Value Report")

        total_rows = df.count()

        report = []

        for column in df.columns:

            missing = df.filter(
                col(column).isNull()
            ).count()

            percentage = round(
                (missing / total_rows) * 100,
                2
            )

            report.append(
                (
                    column,
                    missing,
                    percentage
                )
            )

        report.sort(
            key=lambda x: x[2],
            reverse=True
        )

        print()

        print("-" * 60)

        print("Missing Value Report")

        print("-" * 60)

        for row in report:

            print(
                f"{row[0]:25}"
                f"{row[1]:10}"
                f"{row[2]:10}%"
            )

    # --------------------------------------------------

    def profile_dataset(
        self,
        df: DataFrame,
        dataset_name: str
    ):

        logger.info(
            f"Profiling {dataset_name}"
        )

        rows, columns = self.shape(df)

        print()

        print("=" * 70)

        print(dataset_name)

        print("=" * 70)

        print(f"Rows    : {rows}")

        print(f"Columns : {columns}")

        self.summary(df)

        self.duplicate_count(df)

        self.missing_report(df)