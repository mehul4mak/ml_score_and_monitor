"""Training Script
"""

import json
import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

###################Load config.json and get path variables
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])

DROP_COLUMNS = ["corporation"]
TARGET = "exited"

df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for training

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        X and y dataframe for ML
    """

    data.drop(DROP_COLUMNS, axis=1, inplace=True)
    X = data.drop(TARGET, axis=1).values
    y = data[TARGET].values

    return X, y


#################Function for training the model
def train_model(X, y):
    """Train a ML Model"""

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, "trainedmodel.pkl"), "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    """main"""
    X, y = prepare_data(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
