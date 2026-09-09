"""
hyperparameter.py

Hyperparameter tuning for the Random Forest classifier using CrossValidator.

Key improvements over original:
  - Uses Precision-Recall AUC (areaUnderPR) instead of ROC-AUC — much more
    informative for imbalanced datasets like this one (~3.5% fraud rate).
  - Wider parameter grid: adds minInstancesPerNode and subsamplingRate.
  - numFolds=5 for more stable CV estimates.
  - parallelism=4 to speed up the grid search.
  - weightCol passed through to handle class imbalance.
"""

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.utils.logger import logger


class HyperParameterTuning:

    def tune_random_forest(self, train_df):

        logger.info("Starting Random Forest hyperparameter tuning (PR-AUC optimized)...")

        rf = RandomForestClassifier(
            labelCol="isFraud",
            featuresCol="features",
            weightCol="classWeight",
            maxBins=128,
            seed=42
        )

        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees,            [100, 200])
            .addGrid(rf.maxDepth,            [8, 12])
            .addGrid(rf.minInstancesPerNode, [1, 5])
            .addGrid(rf.subsamplingRate,     [0.8, 1.0])
            .build()
        )

        # Use PR-AUC — more meaningful than ROC-AUC for imbalanced fraud data
        evaluator = BinaryClassificationEvaluator(
            labelCol="isFraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5,
            parallelism=4,   # Run 4 param combos in parallel
            seed=42
        )

        logger.info(f"Grid size: {len(param_grid)} combinations × 5 folds")

        cv_model = cv.fit(train_df)

        best_model = cv_model.bestModel
        best_pr_auc = max(cv_model.avgMetrics)

        logger.info(f"Best PR-AUC from CV: {best_pr_auc:.4f}")
        logger.info(f"Best params: numTrees={best_model.getNumTrees}, maxDepth={best_model.getMaxDepth()}")

        return best_model