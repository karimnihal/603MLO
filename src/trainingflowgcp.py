# src/trainingflowgcp.py

from metaflow import FlowSpec, step, conda_base, kubernetes, Parameter, current
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

@conda_base(
    python="3.10", # Using a more standard minor version like 3.10 often helps avoid obscure issues
    libraries={
        "pandas": "1.5.3",
        "pyarrow": "12.0.1",
        "scikit-learn": "1.2.2",
        "mlflow": "2.15.1",
        "google-cloud-storage": "", # Let conda resolve
        "google-auth": ""           # Let conda resolve
    },
    packages={
        "fsspec": "2023.10.0", # Specify recent versions
        "gcsfs": "2023.10.0"   # Specify recent versions
    }
)
class AmesHousingTrainFlowGCP(FlowSpec):

    # Define parameters for configuration flexibility
    mlflow_tracking_uri = Parameter(
        'mlflow_tracking_uri',
        default="https://mlflow-server-931658252548.us-west2.run.app",
        help="MLflow Tracking Server URI"
    )
    mlflow_experiment_name = Parameter(
        'mlflow_experiment_name',
        default="ames-housing-models",
        help="MLflow Experiment Name"
    )
    gcs_data_prefix = Parameter(
        'gcs_data_prefix',
        default="gs://mlflow-nihalk-bucket/data_cleaned",
        help="GCS path prefix for cleaned data"
    )

    @kubernetes # Add resource requests if needed: @kubernetes(cpu=2, memory=4096)
    @step
    def start(self):
        """Configure MLflow & pull data from GCS."""
        print(f"Setting MLflow tracking URI to: {self.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print(f"Setting MLflow experiment to: {self.mlflow_experiment_name}")
        mlflow.set_experiment(self.mlflow_experiment_name)

        print(f"Reading data from: {self.gcs_data_prefix}")
        self.X_train = pd.read_parquet(f"{self.gcs_data_prefix}/X_train_reduced.parquet")
        self.y_train = (
            pd.read_parquet(f"{self.gcs_data_prefix}/y_train.parquet")
              .values
              .ravel()
        )
        self.X_test = pd.read_parquet(f"{self.gcs_data_prefix}/X_test_reduced.parquet")
        self.y_test = (
            pd.read_parquet(f"{self.gcs_data_prefix}/y_test.parquet")
              .values
              .ravel()
        )
        print("Data loaded successfully.")

        # your top‐3 configs from hyperopt
        self.top_models_params = [
            {"n_estimators": 210, "max_depth": 15, "max_features": 0.51873},
            {"n_estimators": 230, "max_depth": 11, "max_features": 0.42220},
            {"n_estimators": 280, "max_depth": 11, "max_features": 0.54942}
        ]

        self.next(self.train_models)

    @kubernetes # Add resource requests if needed
    @step
    def train_models(self):
        """Fit each candidate and log to MLflow."""
        # Set tracking URI and experiment again in this step's context
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        self.model_results = [] # Store (model object, r2 score, run_id)
        print(f"Training {len(self.top_models_params)} models...")
        for idx, params in enumerate(self.top_models_params, 1):
            run_name = f"Top_Model_{idx}_Run_{current.run_id}" # Make run name more unique
            with mlflow.start_run(run_name=run_name):
                run_id = mlflow.active_run().info.run_id
                print(f"Starting MLflow run: {run_name} ({run_id})")
                mlflow.set_tag("model_type", "random_forest_top")
                mlflow.log_params(params)

                m = RandomForestRegressor(
                    n_estimators=int(params["n_estimators"]), # Ensure integers
                    max_depth=int(params["max_depth"]),       # Ensure integers
                    max_features=params["max_features"],
                    random_state=42,
                    n_jobs=-1 # Use all available cores
                )
                print(f"Fitting model {idx}...")
                m.fit(self.X_train, self.y_train)
                print(f"Predicting with model {idx}...")
                preds = m.predict(self.X_test)

                mse = mean_squared_error(self.y_test, preds)
                r2  = r2_score(self.y_test, preds)
                print(f"Model {idx} - Test MSE: {mse:.4f}, Test R2: {r2:.4f}")
                mlflow.log_metric("test_mse", mse)
                mlflow.log_metric("test_r2",  r2)

                print(f"Logging model {idx} artifact...")
                mlflow.sklearn.log_model(m, "model")

                # Store the fitted model, its R2 score, and the MLflow run_id
                self.model_results.append({
                    "model": m,
                    "r2": r2,
                    "run_id": run_id,
                    "params": params # Keep params for reference if needed
                 })
                # 'with' block handles mlflow.end_run()

        print("Model training and logging complete.")
        self.next(self.choose_model)

    @kubernetes # Add resource requests if needed
    @step
    def choose_model(self):
        """Find the best-scoring model and register it in MLflow Model Registry."""
        # Set tracking URI and experiment again in this step's context
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)

        print("Choosing the best model...")
        if not self.model_results:
            print("No models were trained. Exiting.")
            self.next(self.end)
            return

        # Find the result dictionary with the highest R2 score
        best_result = max(self.model_results, key=lambda x: x["r2"])
        best_model = best_result["model"]
        best_r2 = best_result["r2"]
        best_run_id = best_result["run_id"]

        print(f"Best model found from run ID: {best_run_id} with R2: {best_r2:.4f}")

        # Register the model from the specific run where it was originally logged
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "ames_housing_final_model" # Name for the registered model

        try:
            print(f"Registering model '{model_name}' from URI: {model_uri}")
            registered_model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"✅ Successfully registered model '{model_name}' version {registered_model_version.version}")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            # Decide if you want the flow to fail here or just log the error
            # raise e # Uncomment to make the flow fail if registration fails

        self.best_model_r2 = best_r2 # Save for end step if needed
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Training flow completed on Kubernetes!")
        # Accessing self.best_model_r2 if needed
        # print(f"Best model registered had R2: {self.best_model_r2:.4f}")


if __name__ == "__main__":
    AmesHousingTrainFlowGCP()


# Ran with:
# python trainingflowgcp.py --environment=conda run --with kubernetes   
