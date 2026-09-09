"""
train_model.py

Trains classification models for fraud detection.
All models use weightCol="classWeight" to handle the severe class imbalance
(~3.5% fraud / ~96.5% legitimate) in the IEEE-CIS dataset.
"""

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)
from src.utils.logger import logger


class FraudModel:

    def train_logistic_regression(self, train_df):

        logger.info("Training Logistic Regression...")

        model = LogisticRegression(
            labelCol="isFraud",
            featuresCol="features",
            weightCol="classWeight",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0
        )

        return model.fit(train_df)

    def train_decision_tree(self, train_df):

        logger.info("Training Decision Tree...")

        model = DecisionTreeClassifier(
            labelCol="isFraud",
            featuresCol="features",
            weightCol="classWeight",
            maxDepth=10,
            maxBins=128,
            seed=42
        )

        return model.fit(train_df)

    def train_random_forest(self, train_df):

        logger.info("Training Random Forest...")

        model = RandomForestClassifier(
            labelCol="isFraud",
            featuresCol="features",
            weightCol="classWeight",
            numTrees=100,
            maxDepth=10,
            maxBins=128,
            seed=42
        )

        return model.fit(train_df)

    def train_gbt(self, train_df):

        logger.info("Training GBT Classifier...")

        model = GBTClassifier(
            labelCol="isFraud",
            featuresCol="features",
            weightCol="classWeight",
            maxIter=100,
            maxDepth=5,
            maxBins=128,
            stepSize=0.1,
            seed=42
        )

        return model.fit(train_df)

    def train_all(self, lr_train, tree_train):

        models = {

            "Logistic Regression":
                self.train_logistic_regression(lr_train),

            "Decision Tree":
                self.train_decision_tree(tree_train),

            "Random Forest":
                self.train_random_forest(tree_train),

            "GBT":
                self.train_gbt(tree_train)

        }

        return models