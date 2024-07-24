"""APP 
"""

import json
import os
import pickle

# import create_prediction_model
# import diagnosis
import numpy as np
import pandas as pd

# import predict_exited_from_saved_model
from flask import Flask, jsonify, request, session

from diagnostics import (
    dataframe_na,
    dataframe_summary,
    execution_time,
    model_predictions,
)
from scoring import score_model
from training import prepare_data

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])

with open(
    os.path.join(config["prod_deployment_path"], "trainedmodel.pkl"), "rb"
) as f:
    prediction_model = pickle.load(f)

test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

X, y = prepare_data(test_df)


# Prediction Endpoint
@app.route("/prediction")
def predict():
    """Prediction endpoint"""
    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = prepare_data(test_df)
    y_pred = model_predictions(X)
    return str(y_pred)  # add return value for prediction outputs


# Scoring Endpoint
@app.route("/scoring")
def scoring():
    """Scoring endpoint"""
    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = prepare_data(test_df)
    score = score_model(prediction_model, X, y)
    return str(score)  # add return value (a single F1 score number)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    """Summary endpoint"""
    # check means, medians, and modes for each column
    summary_df = dataframe_summary()
    return str(
        summary_df
    )  # return a list of all calculated summary statistics


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    """Diagnostics endpoint"""
    # check timing and percent NA values
    exec_time_list = execution_time()
    na_summary_list = dataframe_na()
    return str(
        [*exec_time_list, *na_summary_list]
    )  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
