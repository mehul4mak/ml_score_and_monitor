from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from training import prepare_data


#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    with open(os.path.join(config["output_model_path"], "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = prepare_data(test_df)

    y_pred = model.predict(X)

    f1_score: float = metrics.f1_score(y, y_pred)
    print(f1_score)

    # Open the file in write mode with UTF-8 encoding
    with open("latestscore.txt", "w", encoding="utf-8") as file:
        # Write the F1 score to the file
        file.write(f"F1 Score: {f1_score}\n")


score_model()
