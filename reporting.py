"""Reporting Script
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

from scoring import score_model
from training import prepare_data

###############Load config.json and get path variables
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


# Function for reporting
def plot_confusion_matrix(model, X, y):
    """Score a ML model and saave confusion matrix"""
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    y_pred = model.predict(X)

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y, y_pred)

    # Create a confusion matrix display
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(y)
    )

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)

    # Save the confusion matrix as a PNG file
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(config["output_model_path"], "confusionmatrix2.png")
    )


def main() -> None:
    try:
        with open(
            os.path.join(config["prod_deployment_path"], "trainedmodel.pkl"),
            "rb",
        ) as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(e)
        return

    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X, y = prepare_data(test_df)
    plot_confusion_matrix(model, X, y)


if __name__ == "__main__":
    main()
