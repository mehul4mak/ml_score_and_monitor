"""Scoring Script
"""

import json
import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator

from training import prepare_data

#################Load config.json and get path variables
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


#################Function for model scoring
def score_model(
    ml_model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> float:
    """Score a ML Model with test data"""
    # this function should take a trained model, load test data, and calculate
    # an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    y_pred = ml_model.predict(X)

    f1_score: float = metrics.f1_score(y, y_pred)
    print(f1_score)

    # Open the file in write mode with UTF-8 encoding
    with open("latestscore.txt", "w", encoding="utf-8") as file:
        # Write the F1 score to the file
        file.write(f"{f1_score}\n")

    return f1_score


def main() -> None:
    """Main function"""
    with open(
        os.path.join(config["output_model_path"], "trainedmodel.pkl"), "rb"
    ) as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = prepare_data(test_df)

    score_model(model, X, y)


if __name__ == "__main__":
    main()
