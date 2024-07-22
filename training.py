from re import S
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])

DROP_COLUMNS = ["corporation"]
TARGET = "exited"

df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))


def prepare_data(data: pd.DataFrame):
    data.drop(DROP_COLUMNS, axis=1, inplace=True)
    X = data.drop(TARGET, axis=1).values
    y = data[TARGET].values

    return X, y


#################Function for training the model
def train_model(X, y):

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
    with open(os.path.join(model_path, "trainedmodel.pkl"), "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    """main"""
    X, y = prepare_data(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
