import ast
import json
import os
import pickle
from datetime import datetime

import pandas as pd

import deployment
import diagnostics
import reporting
import scoring
import training

with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_path = config["prod_deployment_path"]


def run():
    ##################Check and read new data
    # first, read ingestedfiles.txt
    # second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    files = os.listdir(input_folder_path)

    try:
        with open(
            os.path.join(prod_path, "ingestedfiles.txt"),
            "r+",
            encoding="utf-8",
        ) as file:
            ingestedfiles = file.readlines()
        ingestedfiles = [record.split(", ")[1] for record in ingestedfiles]
        files = [file for file in files if file not in ingestedfiles]

    except FileNotFoundError as e:
        print(e)

    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if files == []:
        print("No New Files to ingest!")
        return

    ##################Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    # os.system("python3 ingestion.py")

    ##################Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    with open(os.path.join(prod_path, "latestscore.txt"), "r") as fp:
        latest_score = ast.literal_eval(fp.read())

    print(latest_score)

    with open(
        os.path.join(config["prod_deployment_path"], "trainedmodel.pkl"), "rb"
    ) as f:
        model = pickle.load(f)

    # test_df = pd.read_csv(os.path.join(input_folder_path, max(files)))

    test_df = pd.concat(
        [
            pd.read_csv(os.path.join(input_folder_path, file))
            for file in (files)
        ]
    )
    X, y = training.prepare_data(test_df)

    score = scoring.score_model(model, X, y)

    if score >= latest_score:
        return
    print("Model Drift occured...")

    training.train_model(X, y)

    deployment.store_model_into_pickle()

    os.system("python diagnostics.py")
    os.system("python reporting.py")


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model


if __name__ == "__main__":
    run()
