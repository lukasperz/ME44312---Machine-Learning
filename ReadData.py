# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:35:22 2025

@author: makro
"""

import pandas as pd
import os

# Define the path to the directory containing the data
DATA_DIR = os.path.join(os.getcwd(), 'Data')

# Initialize a dictionary to store data
nyc_taxi_data = {}

# Loop through all files in the directory
for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet"):
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

        # Store the DataFrame in a nested dictionary
        if year not in nyc_taxi_data:
            nyc_taxi_data[year] = {}
        nyc_taxi_data[year][month] = df

# Display available years and months
years = sorted(nyc_taxi_data.keys())
print(f"Data available for years: {years}")
for year in years:
    print(f"Year {year}: Months available -> {sorted(nyc_taxi_data[year].keys())}")

# Example usage: Access January 2024 data
year, month = 2024, 1
if year in nyc_taxi_data and month in nyc_taxi_data[year]:
    print(nyc_taxi_data[year][month].head())

    print(nyc_taxi_data[year][month].head())
