from src.spark_session import create_spark_session

from src.etl.pipeline import ETLPipeline

from src.ml.train_model import FraudModel
from src.ml.evaluate_model import ModelEvaluator
from src.ml.hyperparameter import HyperParameterTuning
from src.ml.predict import FraudPredictor


def main():

    # ----------------------------------
    # Create Spark Session
    # ----------------------------------

    spark = create_spark_session()

    # ----------------------------------
    # Run ETL Pipeline
    # ----------------------------------

    pipeline = ETLPipeline(spark)

    (
        lr_train,
        lr_test,
        tree_train,
        tree_test,
        tree_pipeline
    ) = pipeline.run()

    print(tree_train.columns)

    # ----------------------------------
    # Train Models
    # ----------------------------------

    trainer = FraudModel()

    models = trainer.train_all(
        lr_train,
        tree_train
    )

    # ----------------------------------
    # Evaluate Models
    # ----------------------------------

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(
        models,
        lr_test,
        tree_test
    )

    print("\n========== MODEL PERFORMANCE ==========\n")

    for result in results:
        print(result)

    # ----------------------------------
    # Hyperparameter Tuning
    # ----------------------------------

    tuner = HyperParameterTuning()

    best_rf = tuner.tune_random_forest(
        tree_train
    )

    # ----------------------------------
    # Save Models
    # ----------------------------------

    best_rf.write().overwrite().save(
        "models/random_forest_model"
    )

    tree_pipeline.write().overwrite().save(
        "models/tree_pipeline"
    )

    print("\n===================================")
    print(" Project Completed Successfully ")
    print("===================================\n")

    spark.stop()


if __name__ == "__main__":
    main()