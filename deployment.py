"""Deployment Script
"""

import json
import os
import shutil

# Load config.json and correct path variable
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

dataset_csv_path: str = os.path.join(config["output_folder_path"])
prod_deployment_path: str = os.path.join(config["prod_deployment_path"])

model_path: str = os.path.join(config["output_model_path"])


# function for deployment
def store_model_into_pickle():
    """Store the model into prod folder path"""
    # copy the latest pickle file, the latestscore.txt value, and the
    # ingestfiles.txt file into
    # the deployment directory

    shutil.copy(
        os.path.join(config["output_model_path"], "trainedmodel.pkl"),
        os.path.join(prod_deployment_path, "trainedmodel.pkl"),
    )

    shutil.copy(
        "latestscore.txt",
        os.path.join(prod_deployment_path, "latestscore.txt"),
    )

    shutil.copy(
        "ingestedfiles.txt",
        os.path.join(prod_deployment_path, "ingestedfiles.txt"),
    )


store_model_into_pickle()
