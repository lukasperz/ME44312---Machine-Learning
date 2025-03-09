import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import fastparquet

# Define the absolute path to the directory containing the data
DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_EVALUATE_YELLOW"

# Initialize a dictionary to store data
nyc_taxi_data = {}

# Required columns
columns_to_keep = ["tpep_pickup_datetime", "RatecodeID", "trip_distance", "fare_amount", "PULocationID", "payment_type", "tip_amount"]

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

df_jan_2024 = get_taxi_data(2023, 1)
if df_jan_2024 is not None:
    print(df_jan_2024.head())
    # print(df_jan_2024.tail())



##Functions for linear regression part START

def computeCost(X, y, theta):
    m = len(y)
    h = X @ theta.T
    return np.sum((h - y) ** 2) / (2 * m)  # We create a function to compute the Cost function J(theta) = 1/2*m * sum for i = 1 to m of residual; m = number of training point = number of X rows

def gradientDescent(X, y, theta, alpha, iters):  #optimize theta using gradient descent
    m = len(y)
    cost_history = []
    for _ in range(iters): #thetaj is computed iteratively for the hypothesis function
        gradient = (X.T @ ((X @ theta.T) - y)) / m
        theta = theta - alpha * gradient.T
        cost_history.append(computeCost(X, y, theta))
    return theta, cost_history

def predict_tip_amount():
    while True:
        print("\n=== Predict Tip Amount ===")

        # User inputs
        weekday = int(input("Enter Weekday (1 for weekday, 0 for weekend): "))
        ratecode_id = int(input("Enter RatecodeID: "))
        trip_distance = float(input("Enter Trip Distance: "))
        fare_amount = float(input("Enter Fare Amount: "))
        pu_location_id = int(input("Enter PULocationID: "))
        payment_type = int(input("Enter Payment Type: "))

        # Ask for time of day category
        time_of_day = input("Enter Time of Day (morning, noon, afternoon, evening, night): ").strip().lower()

        # Extract theta values
        theta_values = theta.flatten()

        # Base formula with bias term
        tip_amount = theta_values[0] + (theta_values[1] * weekday) + (theta_values[2] * trip_distance) + (
                    theta_values[3] * fare_amount)

        # Handle one-hot encoding for time_of_day
        for i, col in enumerate(df.filter(like='time_of_day_').columns, start=4):
            if f'time_of_day_{time_of_day}' == col:
                tip_amount += theta_values[i]

        # Handle one-hot encoding for PULocationID
        for i, col in enumerate(df.filter(like='PULocationID_').columns,
                                start=4 + len(df.filter(like='time_of_day_').columns)):
            if f'PULocationID_{pu_location_id}' == col:
                tip_amount += theta_values[i]

        # Handle one-hot encoding for RatecodeID
        for i, col in enumerate(df.filter(like='RatecodeID_').columns,
                                start=4 + len(df.filter(like='time_of_day_').columns) + len(
                                        df.filter(like='PULocationID_').columns)):
            if f'RatecodeID_{ratecode_id}' == col:
                tip_amount += theta_values[i]

        # Handle one-hot encoding for Payment Type
        for i, col in enumerate(df.filter(like='payment_type_').columns,
                                start=4 + len(df.filter(like='time_of_day_').columns) + len(
                                        df.filter(like='PULocationID_').columns) + len(
                                        df.filter(like='RatecodeID_').columns)):
            if f'payment_type_{payment_type}' == col:
                tip_amount += theta_values[i]

        # Ensure tip amount is non-negative
        tip_amount = max(0, tip_amount)

        print(f"\nPredicted Tip Amount: {tip_amount:.4f}")

def predict_tip_amount_s():
    while True:
        print("\n=== Predict Tip Amount ===")

        trip_distance = float(input("Enter Trip Distance: "))
        fare_amount = float(input("Enter Fare Amount: "))

        # Normalize inputs
        trip_distance = (trip_distance - X_mean[0]) / X_std[0]
        fare_amount = (fare_amount - X_mean[1]) / X_std[1]

        # Compute prediction
        tip_amount = theta[0, 0] + (theta[0, 1] * trip_distance) + (theta[0, 2] * fare_amount)

        # Reverse normalization
        tip_amount = (tip_amount * y_std) + y_mean
        tip_amount = max(0, tip_amount)

        print(f"\nPredicted Tip Amount: {tip_amount:.4f}")
##Function for linear regression part END







