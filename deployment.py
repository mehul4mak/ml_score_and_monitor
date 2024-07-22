from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path: str = os.path.join(config["output_folder_path"])
prod_deployment_path: str = os.path.join(config["prod_deployment_path"])

model_path: str = os.path.join(config["output_model_path"])


####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into
    # the deployment directory

    shutil.copy(
        os.path.join(config["output_model_path"], "trainedmodel.pkl"),
        os.path.join(prod_deployment_path, "trainedmodel.pkl"),
    )

    shutil.copy(
        "latestscore.txt", os.path.join(prod_deployment_path, "latestscore.txt")
    )

    shutil.copy(
        "ingestedfiles.txt", os.path.join(prod_deployment_path, "ingestedfiles.txt")
    )


store_model_into_pickle()
