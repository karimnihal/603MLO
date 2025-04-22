from metaflow import FlowSpec, step, Parameter
import mlflow
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

class ScoringFlow(FlowSpec):

    input_path = Parameter("input_path", help="Path to test parquet file", default="data_cleaned/X_test_reduced.parquet")
    target_path = Parameter("target_path", help="Path to y_test parquet file", default="data_cleaned/y_test.parquet")

    @step
    def start(self):
        """Load input data and target data"""
        self.X_test = pd.read_parquet(self.input_path)
        self.y_test = pd.read_parquet(self.target_path).squeeze()
        print(f"Loaded test data from {self.input_path} with shape {self.X_test.shape}")
        self.next(self.load_model)

    @step
    def load_model(self):
        """Load the registered model from MLflow"""
        mlflow.set_tracking_uri("https://mlflow-server-931658252548.us-west2.run.app")
        model_uri = "models:/ames_housing_final_model/1"  # adjust version if needed
        self.model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")
        self.next(self.predict)

    @step
    def predict(self):
        """Generate predictions and calculate metrics"""
        self.preds = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.preds)
        self.r2 = r2_score(self.y_test, self.preds)
        print(f"Scoring complete. R2: {self.r2:.4f}, MSE: {self.mse:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """Final step to log metrics or save results if needed"""
        mlflow.set_tracking_uri("https://mlflow-server-931658252548.us-west2.run.app")
        mlflow.set_experiment("ames-housing-scoring")

        with mlflow.start_run(run_name="Model Scoring"):
            mlflow.log_metric("test_r2", self.r2)
            mlflow.log_metric("test_mse", self.mse)

        print("Metrics logged to MLflow. Flow complete.")

if __name__ == '__main__':
    ScoringFlow()