from src.spark_session import create_spark_session

from src.etl.pipeline import ETLPipeline

from src.ml.train_model import FraudModel
from src.ml.evaluate_model import ModelEvaluator
from src.ml.hyperparameter import HyperParameterTuning
from src.ml.predict import FraudPredictor
from src.ml.metrics_store import save_metrics

from datetime import datetime


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
        print(f"\n--- {result['Model']} ---")
        print(f"  Accuracy       : {result['Accuracy']:.4f}")
        print(f"  ROC-AUC        : {result['ROC_AUC']:.4f}")
        print(f"  PR-AUC         : {result['PR_AUC']:.4f}")
        print(f"  Fraud Precision: {result['FraudPrecision']:.4f}")
        print(f"  Fraud Recall   : {result['FraudRecall']:.4f}")
        print(f"  Fraud F1       : {result['FraudF1']:.4f}")
        print(f"  TP={result['TP']}  FP={result['FP']}  FN={result['FN']}  TN={result['TN']}")

    # ----------------------------------
    # Persist metrics to JSON
    # ----------------------------------

    save_metrics(results)

    print("\nMetrics saved to models/metrics.json")

    # ----------------------------------
    # Hyperparameter Tuning
    # ----------------------------------

    best_rf = HyperParameterTuning().tune_random_forest(tree_train)

    # ----------------------------------
    # Save Models (latest + timestamped run)
    # ----------------------------------

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save canonical latest (used by API)
    best_rf.write().overwrite().save("models/random_forest_model")
    tree_pipeline.write().overwrite().save("models/tree_pipeline")

    # Save versioned run (preserves history across retraining)
    best_rf.write().overwrite().save(f"models/runs/{run_id}/random_forest_model")
    tree_pipeline.write().overwrite().save(f"models/runs/{run_id}/tree_pipeline")

    print(f"\n===================================")
    print(f" Project Completed Successfully")
    print(f" Run ID: {run_id}")
    print(f"===================================\n")

    spark.stop()


if __name__ == "__main__":
    main()