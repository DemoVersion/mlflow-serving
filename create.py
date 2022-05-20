import mlflow
from huggingfacewrapper import HuggingFaceWrapper 


mlflow.set_registry_uri('http://127.0.0.1:5000')
mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run(run_name="LASTDAYRUN") as run:
    hfw = HuggingFaceWrapper()

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_param("label_smoothing_factor", 0.1)
    mlflow.log_metrics({"dev_accuracy": 78.62, "dev_f1": 79.74})

    mlflow.pyfunc.log_model(
        python_model=hfw,
        artifact_path="hfw-model",
        code_path=['huggingfacewrapper.py'],
        registered_model_name="hf-zero-shot-classifier",
        pip_requirements=["transformers"]
    )