FROM python:3-slim
ARG MLFLOW_VERSION=1.25.1

WORKDIR /mlflow/
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION
RUN pip install --no-cache-dir boto3
EXPOSE 5000

ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT s3://mlflow

ENV MLFLOW_S3_ENDPOINT_URL http://127.0.0.1:9000
ENV MLFLOW_S3_IGNORE_TLS true
ENV AWS_ACCESS_KEY_ID mlflow
ENV AWS_SECRET_ACCESS_KEY mlflow1234

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000
