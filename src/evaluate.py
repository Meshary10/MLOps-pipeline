from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from train import load_data


def main(repo_path):
    test_csv_path = repo_path / "data/prepared/test.csv"
    test_data, labels = load_data(test_csv_path)
    model = load(repo_path / "model/model.pkl")
    predictions = model.predict(test_data)
    accuracy = accuracy_score(labels, predictions)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
