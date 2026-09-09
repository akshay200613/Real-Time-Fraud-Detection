import os
import sys
import ctypes

def setup_spark_windows_env():
    if sys.platform == "win32":
        def get_short_path(path):
            if not path or not os.path.exists(path):
                return path
            buf = ctypes.create_unicode_buffer(500)
            ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 500)
            return buf.value or path

        # Set JAVA_HOME
        java_home = os.environ.get("JAVA_HOME")
        if not java_home or not os.path.exists(java_home):
            default_jdk = r"C:\Users\aksha\.jdk\jdk-17.0.12+7"
            if os.path.exists(default_jdk):
                java_home = default_jdk
        if java_home:
            short_java = get_short_path(java_home)
            os.environ["JAVA_HOME"] = short_java
            bin_path = os.path.join(short_java, "bin")
            if bin_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_path + ";" + os.environ.get("PATH", "")

        # Set HADOOP_HOME
        hadoop_home = os.environ.get("HADOOP_HOME")
        if not hadoop_home or not os.path.exists(hadoop_home):
            default_hadoop = r"C:\Users\aksha\.hadoop"
            if os.path.exists(default_hadoop):
                hadoop_home = default_hadoop
        if hadoop_home:
            short_hadoop = get_short_path(hadoop_home)
            os.environ["HADOOP_HOME"] = short_hadoop
            hbin_path = os.path.join(short_hadoop, "bin")
            if hbin_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = hbin_path + ";" + os.environ.get("PATH", "")

        # Set PySpark Python interpreter
        short_py = get_short_path(sys.executable)
        os.environ["PYSPARK_PYTHON"] = short_py
        os.environ["PYSPARK_DRIVER_PYTHON"] = short_py

        # Set SPARK_HOME
        try:
            import pyspark
            pyspark_dir = os.path.dirname(pyspark.__file__)
            os.environ["SPARK_HOME"] = get_short_path(pyspark_dir)
        except Exception:
            pass

setup_spark_windows_env()

from pyspark.sql import SparkSession
from src.config import (
    APP_NAME,
    MASTER,
    SHUFFLE_PARTITIONS,
    DRIVER_MEMORY,
    EXECUTOR_MEMORY
)


def create_spark_session():
    setup_spark_windows_env()

    builder = (
        SparkSession.builder
        .appName(APP_NAME)
        .master(MASTER)
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.executor.memory", EXECUTOR_MEMORY)
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.ui.enabled", "false")
        .config("spark.python.worker.reuse", "true")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    return spark

