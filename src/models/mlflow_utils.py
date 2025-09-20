import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.auth.add_app_token(os.getenv("DAGSHUB_AUTH_TOKEN"))
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO"),
    mlflow=True,
)

class MLflowTracker:
    @staticmethod
    def track_model(name: str, model, metrics: dict, params: dict = None, framework: str = "sklearn"):
        with mlflow.start_run():
            mlflow.log_param("model_name", name)
            if params:
                mlflow.log_params(params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            if framework == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif framework == "pytorch":
                mlflow.pytorch.log_model(model, "model")