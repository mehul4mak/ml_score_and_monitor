import json
import os
import pickle
import subprocess
import timeit

import numpy as np
import pandas as pd

from training import prepare_data

# Load config.json and get environment variables
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

X, _ = prepare_data(test_df)


# Function to get model predictions
def model_predictions(X: pd.DataFrame):
    # read the deployed model and a test dataset, calculate predictions
    with open(
        os.path.join(config["prod_deployment_path"], "trainedmodel.pkl"), "rb"
    ) as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    y_pred = y_pred.flatten()
    assert y_pred.shape[0] == X.shape[0]

    print(y_pred.tolist())
    # return value should be a list containing all predictions
    return y_pred.tolist()


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    summary_df = df.describe(include="number")
    summary_df = summary_df.loc[["mean", "std", "50%"], :]

    print(summary_df.isnull().mean().values.tolist())
    print(summary_df)
    # return value should be a list containing all summary statistics

    return summary_df


# Function to get summary statistics
def dataframe_na():
    # calculate summary statistics here

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    print(df.isnull().mean().values.tolist())

    # return value should be a list containing all summary statistics

    return df.isnull().mean().values.tolist()


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    time_list = []
    start_time = timeit.default_timer()
    os.system("python ingestion.py")
    end_time = timeit.default_timer()
    print(f"Ingestion process took: {end_time-start_time:.2f} seconds")
    time_list.append(end_time - start_time)
    start_time = timeit.default_timer()
    os.system("python training.py")
    end_time = timeit.default_timer()
    print(f"Training process took: {end_time-start_time:.2f} seconds")
    time_list.append(end_time - start_time)
    print(time_list)
    return time_list  # return a list of 2 timing values in seconds


# Function to check dependencies
def outdated_packages_list():
    # get a list of
    outdated_pkgs = subprocess.check_output(["pip", "list", "--outdated"])
    print(outdated_pkgs)

    with open("outdated_pkgs2.txt", "wb") as file:
        file.write(outdated_pkgs)


if __name__ == "__main__":
    model_predictions(X)
    dataframe_summary()
    dataframe_na()
    execution_time()
    outdated_packages_list()
