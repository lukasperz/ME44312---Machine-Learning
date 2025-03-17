# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:35:22 2025

@author: makro
"""

import pandas as pd
import os

# Define the file path (Ensure it's a single file, not a directory)
DATA_FILE = os.path.join(os.getcwd(), 'green_taxi_data') #adapt for yellow as well

# Check if the file exists before proceeding
if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError(f"Error: {DATA_FILE} does not exist.")

# Read the Parquet file into a DataFrame
df = pd.read_parquet(DATA_FILE)

# Display the first few rows to confirm it's loaded correctly
print("Loaded Data:")
print(df.head())

# Extract unique years and months if the dataset has a 'date' column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime if not already
    years = sorted(df['date'].dt.year.unique())
    months = sorted(df['date'].dt.month.unique())

    print(f"Data available for years: {years}")
    print(f"Months available: {months}")

    # Example usage: Filter January 2024 data (if applicable)
    year, month = 2024, 1
    df_filtered = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]

    if not df_filtered.empty:
        print(f"\nSample data for {year}-{month}:")
        print(df_filtered.head())
    else:
        print(f"No data available for {year}-{month}.")
else:
    print("No 'date' column found in the dataset.")
