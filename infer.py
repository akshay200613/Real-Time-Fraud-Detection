from src.spark_session import create_spark_session

from src.etl.extract import Extract
from src.etl.cleaning import DataCleaner
from src.etl.feature_engineering import FeatureEngineering

from src.ml.predict import FraudPredictor


def main():

    # ----------------------------------
    # Create Spark Session
    # ----------------------------------

    spark = create_spark_session()

    # ----------------------------------
    # Load Test Dataset
    # ----------------------------------

    extractor = Extract(spark)

    test_df = extractor.merge_test()

    # ----------------------------------
    # Data Cleaning
    # ----------------------------------

    cleaner = DataCleaner()

    test_df = cleaner.clean(test_df)

    # ----------------------------------
    # Feature Engineering
    # ----------------------------------

    feature = FeatureEngineering()

    test_df = feature.create_features(test_df)

    # ----------------------------------
    # Prediction
    # ----------------------------------

    predictor = FraudPredictor()

    predictor.load_models()

    print(test_df.columns)

    predictions = predictor.predict(test_df)

    predictor.show_predictions(predictions)

    spark.stop()


if __name__ == "__main__":
    main()