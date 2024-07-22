import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    files = os.listdir(input_folder_path)
    print(files)
    allrecords = []

    merged_df = pd.DataFrame()
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

    # print(merged_df.shape)
    merged_df.drop_duplicates(inplace=True)
    # print(merged_df.shape)
    # os.mkdir(output_folder_path) if not os.path.exists(output_folder_path) else None

    os.makedirs(output_folder_path, exist_ok=True)
    merged_df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    # Open the file in write mode
    with open("ingestedfiles.txt", "w", encoding="utf-8") as file:
        # Iterate over each item in the list
        for records in allrecords:
            # Convert the item to a string and write it to the file
            file.write(", ".join(map(str, records)) + "\n")

    print(f"Data written!")


def main() -> None:
    """Main"""
    merge_multiple_dataframe()


if __name__ == "__main__":
    main()
