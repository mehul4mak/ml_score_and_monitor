"""Ingestion Script
"""

import json
import os
from datetime import datetime

import pandas as pd

# Load config.json and get input and output paths
with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


# Function for data ingestion
def merge_multiple_dataframe() -> None:
    """Ingestion data from various csv to final csv and save it to disk"""

    files = os.listdir(input_folder_path)
    # if files == []:
    #     print("No Files to ingest!")
    #     return
    try:
        with open(
            os.path.join("ingestedfiles.txt"), "r+", encoding="utf-8"
        ) as file:
            ingestedfiles = file.readlines()
        ingestedfiles = [record.split(", ")[1] for record in ingestedfiles]
        files = [file for file in files if file not in ingestedfiles]

    except FileNotFoundError as e:
        print(e)

    if files == []:
        print("No New Files to ingest!")
        return
    # check for datasets, compile them together, and write to an output file
    allrecords = []

    merged_df = pd.DataFrame()
    # if os.path.exists(os.path.join(output_folder_path, "finaldata.csv")):
    #     print("Finaldata.csv exist, loading...")
    #     merged_df = pd.read_csv(
    #         os.path.join(output_folder_path, "finaldata.csv")
    #     )

    for file in files:
        input_file_name = os.path.join(os.getcwd(), input_folder_path, file)

        temp_df = pd.read_csv(input_file_name)

        records = [
            input_folder_path,
            file,
            temp_df.shape[0],
            datetime.now().strftime("%Y-%m-%d"),
        ]

        merged_df = pd.concat((merged_df, temp_df), axis=0)
        allrecords.append(records)

    merged_df.drop_duplicates(inplace=True)

    os.makedirs(output_folder_path, exist_ok=True)
    merged_df.to_csv(
        os.path.join(output_folder_path, "finaldata.csv"), index=False
    )

    # Open the file in write mode
    print("logging ingested file metadata...")
    with open("ingestedfiles.txt", "a+", encoding="utf-8") as file:
        # Iterate over each item in the list
        for records in allrecords:
            # Convert the item to a string and write it to the file
            file.write(", ".join(map(str, records)) + "\n")

    print("Data written!")


def main() -> None:
    """Main"""
    merge_multiple_dataframe()


if __name__ == "__main__":
    main()
