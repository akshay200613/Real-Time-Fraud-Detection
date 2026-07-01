from src.etl.extract import Extract
from src.etl.cleaning import DataCleaner
from src.etl.feature_engineering import FeatureEngineering
from src.ml.prepare_data import DataPreparation


class ETLPipeline:

    def __init__(self, spark):

        self.extract = Extract(spark)
        self.cleaner = DataCleaner()
        self.feature = FeatureEngineering()
        self.prepare = DataPreparation()

    def run(self):

        # Extract
        df = self.extract.merge_train()

        # Cleaning
        df = self.cleaner.clean(df)

        # Feature Engineering
        df = self.feature.create_features(df)

        # Train/Test Split
        train_df, test_df = self.prepare.train_test_split(df)

        # Prepare Data
        lr_train, lr_test, _ = self.prepare.prepare_lr(
            train_df,
            test_df
        )

        tree_train, tree_test, tree_pipeline = self.prepare.prepare_tree(
            train_df,
            test_df
        )

        return (
            lr_train,
            lr_test,
            tree_train,
            tree_test,
            tree_pipeline
        )