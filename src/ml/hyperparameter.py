from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class HyperParameterTuning:

    def tune_random_forest(self, train_df):

        rf = RandomForestClassifier(
            labelCol="isFraud",
            featuresCol="features",
            maxBins=128,
            seed=42
        )

        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [5, 10])
            .addGrid(rf.maxBins, [64, 128])
            .build()
        )

        evaluator = BinaryClassificationEvaluator(
            labelCol="isFraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )

        model = cv.fit(train_df)

        return model.bestModel