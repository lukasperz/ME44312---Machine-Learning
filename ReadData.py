import pandas as pd
import os
import fastparquet

# Define the absolute path to the directory containing the data
DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_TRAINING_GREEN"

# Initialize a dictionary to store data
nyc_taxi_data = {}

# Required columns
columns_to_keep = ["tpep_pickup_datetime", "lpep_pickup_datetime", "RatecodeID", "trip_distance",
                   "fare_amount", "tip_amount", "payment_type"]

# Loop through all files in the directory
for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet"):
        # print(f"Processing file: {file}")
        # Identify taxi type (Green or Yellow) from filename
        if "green" in file.lower():
            taxi_type = "green"
        elif "yellow" in file.lower():
            taxi_type = "yellow"
        else:
            print(f"Skipping file {file} due to unknown taxi type.")
            continue

        # print(f"Taxi type for {file}: {taxi_type}")
        # Extract year and month from the filename
        try:
            parts = file.split('_')[-1].split('-')  # Extracting '2024-01.parquet'
            year = int(parts[0])
            month = int(parts[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Skipping file {file} due to incorrect format.")
            continue

        # Read the parquet file
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_parquet(file_path)

        df = df.loc[:, df.columns.intersection(columns_to_keep)]
        df["taxi_type"] = taxi_type

        # Store the DataFrame in a nested dictionary
        if year not in nyc_taxi_data:
            nyc_taxi_data[year] = {}
        nyc_taxi_data[year][month] = df


# Display available years and months
years = sorted(nyc_taxi_data.keys())
print(f"Data available for years: {years}")
for year in years:
    print(f"Year {year}: Months available -> {sorted(nyc_taxi_data[year].keys())}")

pd.set_option('display.max_columns', None)

# Example usage: Access January 2024 data
def get_taxi_data(year, month):
    if year in nyc_taxi_data and month in nyc_taxi_data[year]:
        return nyc_taxi_data[year][month]
    else:
        print(f"No data available for {year}-{month}.")
        return None

df_jan_2024 = get_taxi_data(2021, 1)
if df_jan_2024 is not None:
    print(df_jan_2024.head())
    # print(df_jan_2024.tail())







