import os
import pandas as pd
import pyarrow.parquet as pq

# Directory containing the parquet files
directory = "/Users/lukas/TU Delft (Macintosh HD)/ML_for_Transport_and_Multi_Machine_Systems/Assignment/Data"

# Output filenames for green & yellow taxi data
green_parquet_file = os.path.join(directory, "green_taxi_data_NO_SCALING.parquet")
yellow_parquet_file = os.path.join(directory, "yellow_taxi_data_NO_SCALING.parquet")

# Build lists of all relevant Parquet files
green_files = []
yellow_files = []

for filename in os.listdir(directory):
    if filename.endswith(".parquet"):
        if filename.startswith("green"):
            green_files.append(os.path.join(directory, filename))
        elif filename.startswith("yellow"):
            yellow_files.append(os.path.join(directory, filename))

# Process green taxi files: simply read and append
green_dfs = []
for file_path in green_files:
    df = pq.read_table(file_path).to_pandas()
    green_dfs.append(df)

if green_dfs:
    combined_green = pd.concat(green_dfs, ignore_index=True)
    combined_green.to_parquet(green_parquet_file, engine="pyarrow")
    print(f"Combined {len(green_files)} green files into {green_parquet_file}")
else:
    print("No green Parquet files found.")

# Process yellow taxi files: simply read and append
yellow_dfs = []
for file_path in yellow_files:
    df = pq.read_table(file_path).to_pandas()
    yellow_dfs.append(df)

if yellow_dfs:
    combined_yellow = pd.concat(yellow_dfs, ignore_index=True)
    combined_yellow.to_parquet(yellow_parquet_file, engine="pyarrow")
    print(f"Combined {len(yellow_files)} yellow files into {yellow_parquet_file}")
else:
    print("No yellow Parquet files found.")