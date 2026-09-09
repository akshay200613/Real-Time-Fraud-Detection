"""
evaluate_model.py

Evaluates all trained models with a comprehensive set of metrics:

  Weighted metrics (overall):
    - Accuracy, Weighted Precision, Weighted Recall, Weighted F1, ROC-AUC

  Fraud-class-specific metrics (the ones that actually matter for imbalanced data):
    - FraudPrecision  — of all flagged transactions, how many are real fraud?
    - FraudRecall     — of all real fraud, how many did we catch?
    - FraudF1         — harmonic mean of the above two
    - PR_AUC          — area under Precision-Recall curve

  Confusion matrix components:
    - TP, FP, TN, FN
"""

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.sql.functions import col
from src.utils.logger import logger


class ModelEvaluator:

    def evaluate(self, models, lr_test, tree_test):

        accuracy_eval = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="accuracy"
        )

        precision_eval = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        recall_eval = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        f1_eval = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="f1"
        )

        roc_auc_eval = BinaryClassificationEvaluator(
            labelCol="isFraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        pr_auc_eval = BinaryClassificationEvaluator(
            labelCol="isFraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        results = []

        for name, model in models.items():

            test_df = (
                lr_test
                if name == "Logistic Regression"
                else tree_test
            )

            prediction = model.transform(test_df)

            # ------------------------------------------
            # Confusion Matrix components
            # ------------------------------------------
            tp = prediction.filter(
                (col("prediction") == 1) & (col("isFraud") == 1)
            ).count()

            fp = prediction.filter(
                (col("prediction") == 1) & (col("isFraud") == 0)
            ).count()

            fn = prediction.filter(
                (col("prediction") == 0) & (col("isFraud") == 1)
            ).count()

            tn = prediction.filter(
                (col("prediction") == 0) & (col("isFraud") == 0)
            ).count()

            # ------------------------------------------
            # Per-class Fraud metrics
            # ------------------------------------------
            fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fraud_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fraud_f1        = (
                2 * fraud_precision * fraud_recall
                / (fraud_precision + fraud_recall + 1e-10)
            )

            result = {
                "Model":           name,
                # Overall metrics
                "Accuracy":        round(accuracy_eval.evaluate(prediction), 4),
                "WeightedPrecision": round(precision_eval.evaluate(prediction), 4),
                "WeightedRecall":  round(recall_eval.evaluate(prediction), 4),
                "WeightedF1":      round(f1_eval.evaluate(prediction), 4),
                "ROC_AUC":         round(roc_auc_eval.evaluate(prediction), 4),
                "PR_AUC":          round(pr_auc_eval.evaluate(prediction), 4),
                # Fraud-class specific
                "FraudPrecision":  round(fraud_precision, 4),
                "FraudRecall":     round(fraud_recall, 4),
                "FraudF1":         round(fraud_f1, 4),
                # Confusion matrix
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }

            logger.info(
                f"[{name}] "
                f"Acc={result['Accuracy']:.3f} | "
                f"PR-AUC={result['PR_AUC']:.3f} | "
                f"FraudRecall={result['FraudRecall']:.3f} | "
                f"FraudPrecision={result['FraudPrecision']:.3f} | "
                f"TP={tp} FP={fp} FN={fn} TN={tn}"
            )

            results.append(result)

        return results