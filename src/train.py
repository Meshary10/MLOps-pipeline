from joblib import dump
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import mlflow
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

# Tracking the experiments on this IP and port
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def load_images(data_frame, column_name):
    filelist = data_frame[column_name].to_list()
    image_list = imread_collection(filelist)
    return image_list


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    return label_list


def preprocess(image):
    resized = resize(image, (100, 100, 3))
    reshaped = resized.reshape((1, 30000))
    return reshaped


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    raw_images = load_images(data_frame=df, column_name="filename")
    processed_images = [preprocess(image) for image in raw_images]
    data = np.concatenate(processed_images, axis=0)
    return data, labels


def main(repo_path):
    train_csv_path = repo_path / "data/prepared/train.csv"
    train_data, labels = load_data(train_csv_path)
    with mlflow.start_run(run_name="Test-pipeline"):
        rf = RandomForestClassifier()
        trained_model = rf.fit(train_data, labels)
        pickle.dump(trained_model, open('/home/ma-user/work/data-version-control/model/model.pkl', 'wb'))
        mlflow.sklearn.log_model(trained_model, "model", registered_model_name="RandomForestClassifier")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
