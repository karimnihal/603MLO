from metaflow import FlowSpec, step
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class AmesHousingTrainFlow(FlowSpec):

    @step
    def start(self):
        """Load pre-cleaned training data and set MLFlow tracking URI."""
        mlflow.set_tracking_uri("https://mlflow-server-931658252548.us-west2.run.app")
        mlflow.set_experiment("ames-housing-models")

        self.X_train = pd.read_parquet("data_cleaned/X_train_reduced.parquet")
        self.y_train = pd.read_parquet("data_cleaned/y_train.parquet").values.ravel()
        self.X_test = pd.read_parquet("data_cleaned/X_test_reduced.parquet")
        self.y_test = pd.read_parquet("data_cleaned/y_test.parquet").values.ravel()

        self.top_models = [
            {'n_estimators': 210, 'max_depth': 15, 'max_features': 0.51873},
            {'n_estimators': 230, 'max_depth': 11, 'max_features': 0.42220},
            {'n_estimators': 280, 'max_depth': 11, 'max_features': 0.54942}
        ]
        self.next(self.train_models)

    @step
    def train_models(self):
        """Train models and evaluate their performance."""
        self.results = []
        for idx, params in enumerate(self.top_models, 1):
            with mlflow.start_run(run_name=f"Top_Model_{idx}"):
                mlflow.set_tag("model_type", "random_forest_top")
                mlflow.log_params(params)

                model = RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    max_features=params['max_features'],
                    random_state=42
                )
                model.fit(self.X_train, self.y_train)
                preds = model.predict(self.X_test)

                mse = mean_squared_error(self.y_test, preds)
                r2 = r2_score(self.y_test, preds)
                mlflow.log_metric("test_mse", mse)
                mlflow.log_metric("test_r2", r2)

                mlflow.sklearn.log_model(model, "model")

                self.results.append((model, r2))
                mlflow.end_run()

        self.next(self.choose_model)

    @step
    def choose_model(self):
        """Select and register the best model."""
        best_model, best_score = sorted(self.results, key=lambda x: -x[1])[0]

        with mlflow.start_run(run_name="Register Final Model"):
            mlflow.set_tag("final_model", True)
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "ames_housing_final_model")
            mlflow.end_run()

        print(f"Best model R2: {best_score}")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed.")

if __name__ == '__main__':
    AmesHousingTrainFlow()