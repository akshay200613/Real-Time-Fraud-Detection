from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


class ModelEvaluator:

    def evaluate(self, models, lr_test, tree_test):

        accuracy = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="accuracy"
        )

        precision = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        recall = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        f1 = MulticlassClassificationEvaluator(
            labelCol="isFraud",
            predictionCol="prediction",
            metricName="f1"
        )

        auc = BinaryClassificationEvaluator(
            labelCol="isFraud",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        results = []

        for name, model in models.items():

            test_df = (
                lr_test
                if name == "Logistic Regression"
                else tree_test
            )

            prediction = model.transform(test_df)

            results.append({

                "Model": name,

                "Accuracy": accuracy.evaluate(prediction),

                "Precision": precision.evaluate(prediction),

                "Recall": recall.evaluate(prediction),

                "F1": f1.evaluate(prediction),

                "ROC-AUC": auc.evaluate(prediction)

            })

        return results