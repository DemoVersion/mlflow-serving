
This repository is made to create a working example of using MLFlow in docker container having Minio S3 as its artifacts store. You could use any parts of this tuturial in creating a workflow using MLflow, but be aware to not use `--network host` in production because its not recommended by Docker for security issue. Also it shouldn't be required if your S3 storage have a static IP.

# Requirements 
The following packages are required in order to complete this cookbook. Please install the python packages via anaconda to avoid issues.

 - Docker
 - Anaconda
 - Boto3
 - MLFlow

# Setting Up S3 
First of all we need to setup S3 we use this storage for storing our artifacts. To do so we could simply use minio as a docker instance. If you have you S3 instance running or using Amazon AWS S3 you can simply ignore this phase.

Lets create a folder for our S3 instance:

    mkdir -p ~/minio/data

We will start our Minio using bellow command:

    docker run \
	  -p 9000:9000 \
	  -p 9001:9001 \
	  --name minio1 \
	  -v ~/minio/data:/data \
	  -e "MINIO_ROOT_USER=AKIAIOSFODNN7EXAMPLE" \
	  -e "MINIO_ROOT_PASSWORD=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
	  quay.io/minio/minio server /data --console-address ":9001"

From now on, we could simply start S3 instance by running this:

    sudo docker start minio1
After this you need to use the minio UI from the local link of `http://127.0.0.1:9001` to create a user named `mlflow` (with the password of `mlflow1234`) with bucket named `mlflow`

# Setting Up MLFlow Server 
After cloning into this repository, We could use this command to build our Docker image:

    sudo docker build . -t mlflow-server

After that we could run our docker image using this:

    sudo docker run -it --rm -v ~/mlflow:/mlflow --network host --name mlflowserver mlflow-server

# Configurations
We need to add these lines at the end of ~/.bashrc to set configs used by mlflow.

    export  MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:9000"  
	export  MLFLOW_S3_IGNORE_TLS="true"  
	export  AWS_ACCESS_KEY_ID="mlflow"  
	export  AWS_SECRET_ACCESS_KEY="mlflow1234"

# Creating Model
In this part, we are going to create our model and upload it to the s3 using MLflow library, since we are using HuggingFace here and HuggingFace isn't supported by default by MLflow we are going to use the generic pyfunc provided by MLflow to create a wrapper for HuggingFace, note that in our case there is no training happening and we are just loading some models. but using MLFlow would be useful because switching between models in the future would be very smooth.

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
We could use the command line below to run this specific code, but writing again by yourself in jupyter-notebook is advised.

    python3 create.py
    
After execution of this part, we should be able to see the model uploaded in the MLFlow UI available at `127.0.0.1:5000` we could also use the UI to archive or publish the model.


# Serving Model
For serving the model we need to copy the model S3 path from MLFlow UI, and use the command bellow to create a REST API on the target machine:

    mlflow models serve --model-uri s3://mlflow/0/78d942d3f1df465f931c1e12e0864fe0/artifacts/hfw-model -p 5123 --env-manager local

# Testing Model on Production
We could simply use curl to test our model on production, the following bash line is going to send two samples to our model for prediction:

    curl -X POST -H "Content-Type:application/json; format=pandas-split"  --data '{"columns":["text"],"data":[["مسابقه فوتبال لغو شد."],["المپیاد نجوم فردا برگزار می‌شود."]]}'  http://127.0.0.1:5123/invocations

Which will result in the following, Fortunately predicting the right labels in these two example.

    [{"sequence": "\u0645\u0633\u0627\u0628\u0642\u0647 \u0641\u0648\u062a\u0628\u0627\u0644 \u0644\u063a\u0648 \u0634\u062f.", "labels": ["\u0648\u0631\u0632\u0634\u06cc", "\u0641\u0631\u0647\u0646\u06af\u06cc", "\u0639\u0644\u0645\u06cc", "\u0633\u06cc\u0627\u0633\u06cc"], "scores": [0.323702871799469, 0.27515798807144165, 0.2746085822582245, 0.12653057277202606]}, {"sequence": "\u0627\u0644\u0645\u067e\u06cc\u0627\u062f \u0646\u062c\u0648\u0645 \u0641\u0631\u062f\u0627 \u0628\u0631\u06af\u0632\u0627\u0631 \u0645\u06cc\u200c\u0634\u0648\u062f.", "labels": ["\u0639\u0644\u0645\u06cc", "\u0633\u06cc\u0627\u0633\u06cc", "\u0641\u0631\u0647\u0646\u06af\u06cc", "\u0648\u0631\u0632\u0634\u06cc"], "scores": [0.27652570605278015, 0.25322988629341125, 0.2385851889848709, 0.23165926337242126]}]


