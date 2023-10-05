# MLOps Pipeline with DVC and MLFlow

In this example we will see how can we run the ML pipeline using mlflow for tracking experiments and dvc for data versioning, the pipline contains three python codes and one yaml file:
- **prepare.py:** Save the jpg images on csv files and split the data into train and test.
- **train.py:** Use random forest algorithm for training and mlflow for tracking.
- **evaluate.py:** Test the model and calculate the accuracy.
- **dvc.yaml:** Run the pipeline.

## Steps to run the pipeline: 

Installing MLFlow and DVC:
```console
$ pip install mlflow
$ pip install dvc
$ pip install dvc-s3
```
Initialize Git & DVC:
```console
$ git init
$ dvc init
```
Add your OBS configurations:
```console
$ dvc remote add -d storage s3://bucketname/foldername
$ dvc remote modify storage endpointurl http://obs.me-east-214.ncai.cloud
$ dvc remote modify storage access_key_id 'xxxxx'
$ dvc remote modify storage secret_access_key 'xxxxx'
$ dvc remote modify storage version_aware true
```
Run MLFlow UI on another window:
```console
$ mlflow ui
```
Add the raw data to DVC:
```console
$ dvc add data/raw/train
$ dvc add data/raw/val
```
Run the pipeline:
```console
$ dvc repro 
```
Push codes & data:
```console
$ dvc push
$ git add --all
$ git commit -m " Reproducible Pipelines "
$ git push
```
**When you run the pipeline you be able to see the model versions and files in mlflow ui and the data versions in Object Storage Service.**

![mlflow-ui](https://github.com/Meshary10/MLOps-pipeline/blob/main/mlflow-ui.png?raw=true)

**Happy coding!**
