from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import when, col
from pyspark.ml.functions import vector_to_array

class FraudPredictor:

    def __init__(self):

        self.pipeline = None
        self.model = None

    # ----------------------------------
    # Load Models
    # ----------------------------------

    def load_models(self):

        self.pipeline = PipelineModel.load(
            "models/tree_pipeline"
        )

        self.model = RandomForestClassificationModel.load(
            "models/random_forest_model"
        )

    # ----------------------------------
    # Predict
    # ----------------------------------

    def predict(self, df):

        # Apply the same preprocessing pipeline
        df = self.pipeline.transform(df)

        # Predict using the trained Random Forest model
        return self.model.transform(df)

    # ----------------------------------
    # Display Predictions
    # ----------------------------------

    def show_predictions(self, prediction):

        prediction = (
            prediction
            .withColumn(
                "FraudProbability",
                vector_to_array(col("probability"))[1]
            )
            .withColumn(
                "PredictionLabel",
                when(col("prediction") == 1, "Fraud")
                .otherwise("Legitimate")
            )
        )

        prediction.select(
            "TransactionID",
            "FraudProbability",
            "PredictionLabel"
        ).show(20, truncate=False)
    