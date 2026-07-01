from pyspark.sql import SparkSession
from src.config import (
    APP_NAME,
    MASTER,
    SHUFFLE_PARTITIONS,
    DRIVER_MEMORY,
    EXECUTOR_MEMORY
)


def create_spark_session():

    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .master(MASTER)
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.executor.memory", EXECUTOR_MEMORY)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark