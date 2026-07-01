from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)


class FraudModel:

    def train_logistic_regression(self, train_df):

        model = LogisticRegression(
            labelCol="isFraud",
            featuresCol="features",
            maxIter=100
        )

        return model.fit(train_df)

    def train_decision_tree(self, train_df):

        model = DecisionTreeClassifier(
            labelCol="isFraud",
            featuresCol="features",
            maxDepth=10,
            maxBins=128,
            seed=42
        )

        return model.fit(train_df)

    def train_random_forest(self, train_df):

        model = RandomForestClassifier(
            labelCol="isFraud",
            featuresCol="features",
            numTrees=100,
            maxDepth=10,
            maxBins=128,
            seed=42
        )

        return model.fit(train_df)

    def train_gbt(self, train_df):

        model = GBTClassifier(
            labelCol="isFraud",
            featuresCol="features",
            maxIter=100,
            maxDepth=5,
            maxBins=128,
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