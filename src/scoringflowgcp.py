# src/scoringflowgcp.py

from metaflow import FlowSpec, step, conda_base, kubernetes, Parameter, current
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import os

@conda_base(
    python="3.10", # Standardize python version
    libraries={
        "pandas": "1.5.3",
        "pyarrow": "12.0.1",
        "scikit-learn": "1.2.2",
        "mlflow": "2.15.1",         # This version supports aliases but load_model needs help
        "google-cloud-storage": "", # Let conda resolve compatible version
        "google-auth": ""           # Let conda resolve compatible version
    },
    packages={
        "fsspec": "2023.10.0",       # Specify recent versions for pip
        "gcsfs": "2023.10.0"        # Specify recent versions for pip
    }
)
class ScoringFlowGCP(FlowSpec):

    # --- Data Parameters ---
    input_path = Parameter(
        "input_path",
        help="GCS path to X_test parquet file",
        default="gs://mlflow-nihalk-bucket/data_cleaned/X_test_reduced.parquet"
    )
    target_path = Parameter(
        "target_path",
        help="GCS path to y_test parquet file",
        default="gs://mlflow-nihalk-bucket/data_cleaned/y_test.parquet"
    )

    # --- MLflow Parameters ---
    mlflow_tracking_uri = Parameter(
        'mlflow_tracking_uri',
        default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow-server-931658252548.us-west2.run.app"),
        help="MLflow Tracking Server URI"
    )
    mlflow_scoring_experiment_name = Parameter(
        'mlflow_experiment_name',
        default="ames-housing-scoring",
        help="MLflow Experiment Name for Scoring Runs"
    )
    registered_model_name = Parameter(
        'registered_model_name',
        default="ames_housing_final_model",
        help="Name of the registered model in MLflow"
    )
    model_alias_or_version = Parameter(
        'model_alias_or_version',
        default="champion", # Default to a common alias pattern
        help="Alias (e.g., champion) or Version (e.g., 1) of the registered model to load"
    )


    @kubernetes # Add resource requests if needed: @kubernetes(cpu=1, memory=2048)
    @step
    def start(self):
        """Pull test data from GCS."""
        print(f"Loading test features from: {self.input_path}")
        self.X_test = pd.read_parquet(self.input_path)

        print(f"Loading test target from: {self.target_path}")
        # .squeeze() converts single-column DataFrame/Series to Series/Array
        self.y_test = pd.read_parquet(self.target_path).squeeze()

        print(f"Loaded test data: X_test shape {self.X_test.shape}, y_test length {len(self.y_test)}")
        self.next(self.load_model)

    @kubernetes # Add resource requests if needed
    @step
    def load_model(self):
        """Fetch the specified registered model from MLflow by resolving alias/version first."""
        print(f"Setting MLflow tracking URI to: {self.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Setting experiment isn't strictly necessary for loading, but good practice
        mlflow.set_experiment(self.mlflow_scoring_experiment_name)

        model_identifier = self.model_alias_or_version
        print(f"Attempting to resolve model identifier: '{model_identifier}' for model '{self.registered_model_name}'")

        try:
            client = mlflow.tracking.MlflowClient()
            if model_identifier.isdigit():
                # Identifier is likely a version number
                print(f"Identifier '{model_identifier}' looks like a version number.")
                self.loaded_model_version = model_identifier
                # Construct URI with specific version
                model_uri = f"models:/{self.registered_model_name}/{self.loaded_model_version}"
            else:
                # Identifier is likely an alias
                print(f"Identifier '{model_identifier}' looks like an alias. Resolving...")
                # Use client to get version details from alias
                model_version_details = client.get_model_version_by_alias(
                    self.registered_model_name, model_identifier
                )
                self.loaded_model_version = model_version_details.version
                # Construct URI with the resolved specific version
                model_uri = f"models:/{self.registered_model_name}/{self.loaded_model_version}"
                print(f"Alias '{model_identifier}' resolved to version '{self.loaded_model_version}'.")

        except mlflow.exceptions.MlflowException as e:
            # Catch errors during alias resolution (e.g., alias not found)
            print(f"❌ Failed to resolve alias or version '{model_identifier}' for model '{self.registered_model_name}'. Error: {e}")
            print("Ensure the model name and alias/version exist in MLflow.")
            raise e # Fail the flow if resolution fails
        except Exception as e:
             print(f"❌ An unexpected error occurred during model identifier resolution. Error: {e}")
             raise e

        # --- Now load using the specific version URI ---
        print(f"Loading model from resolved MLflow URI: {model_uri}")
        try:
            # Load using the URI that now contains only the version number
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Model version {self.loaded_model_version} loaded successfully from MLflow.")
        except Exception as e:
            print(f"❌ Failed to load model using resolved URI '{model_uri}'. Error: {e}")
            raise e # Fail the flow if loading fails

        self.next(self.predict)

    @kubernetes # Add resource requests if needed
    @step
    def predict(self):
        """Score the test data using the loaded model."""
        # Use getattr for safety in case load_model failed before setting loaded_model_version
        print(f"Generating predictions for {len(self.X_test)} samples using model version {getattr(self, 'loaded_model_version', 'Unknown')}...")
        try:
            preds = self.model.predict(self.X_test)
            self.mse = mean_squared_error(self.y_test, preds)
            self.r2  = r2_score(self.y_test, preds)
            print(f"Scoring complete — R2: {self.r2:.4f}, MSE: {self.mse:.4f}")
        except Exception as e:
            print(f"❌ Failed during prediction or metric calculation. Error: {e}")
            raise e
        self.next(self.end)


    @kubernetes # Add resource requests if needed
    @step
    def end(self):
        """Log scoring metrics back into a dedicated MLflow experiment."""
        print(f"Setting MLflow tracking URI to: {self.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print(f"Setting MLflow experiment to: {self.mlflow_scoring_experiment_name}")
        mlflow.set_experiment(self.mlflow_scoring_experiment_name)

        # Create a unique run name including the Metaflow run ID
        run_name = f"Model_Scoring_Run_{current.run_id}"
        print(f"Starting MLflow run '{run_name}' to log metrics...")

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric("test_r2", self.r2)
                mlflow.log_metric("test_mse", self.mse)
                # Log the identifier requested and the actual version loaded
                mlflow.log_param("requested_model_alias_or_version", self.model_alias_or_version)
                if hasattr(self, 'loaded_model_version'):
                    mlflow.log_param("loaded_model_version", self.loaded_model_version)
                mlflow.log_param("scored_model_name", self.registered_model_name)

                print("📊 Metrics logged to MLflow.")
        except Exception as e:
            print(f"❌ Failed to log metrics to MLflow. Error: {e}")
            # raise e # Optionally fail flow

        print("🎉 Scoring flow completed on Kubernetes!")


if __name__ == "__main__":
    ScoringFlowGCP()

# Ran with:
# python scoringflowgcp.py --environment=conda run --with kubernetes --model_alias_or_version=champion