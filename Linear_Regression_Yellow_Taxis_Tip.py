import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#!!! You have to create a folder with all the required parquet files inside. I created one for yellow and one for green taxis for 2015 -2022. To run the code you have to do the same
#!!!For the green taxis the cell related to the pickup time is called "tpep_pickup_datetime", while for yellow taxis "lpep_pickup_datetime"

#%% PART A: LOAD AND PREPROCESS DATA
# path = r'/\Exercise1-2023\Exercise1-2023\data\ex1data1.txt'
# data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_Try" # Path to folder

columns_to_keep = ["tpep_pickup_datetime", "RatecodeID", "trip_distance", "fare_amount", "PULocationID", "payment_type", "tip_amount"]

all_data = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet"):
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_parquet(file_path)
        df = df.loc[:, df.columns.intersection(columns_to_keep)]
        all_data.append(df)

if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    raise ValueError("No data files found.")

df = df.dropna()
df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0) & (df["tip_amount"] > 0)]

pickup_time = pd.to_datetime(df["tpep_pickup_datetime"])
df["weekday"] = pickup_time.dt.weekday < 5  # 1 if weekday, 0 if weekend
df["weekday"] = df["weekday"].astype(int)

bins = [0, 6, 12, 16, 20, 24]
labels = ["night", "morning", "noon", "afternoon", "evening"]
df["time_of_day"] = pd.cut(pickup_time.dt.hour, bins=bins, labels=labels, right=False)

df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["PULocationID", "RatecodeID", "payment_type"], drop_first=True)

X = df[["weekday", "trip_distance", "fare_amount"] + list(df.filter(like='time_of_day_')) + list(df.filter(like='PULocationID_')) + list(df.filter(like='RatecodeID_')) + list(df.filter(like='payment_type_'))]
y = df["tip_amount"]

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
X_std = np.atleast_1d(X_std)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y, ddof=1)
y_std = np.atleast_1d(y_std)
y_std[y_std == 0] = 1
y = (y - y_mean) / y_std

X = np.column_stack((np.ones(X.shape[0]), X))
theta = np.zeros((1, X.shape[1]))

#%% PART B: COMPUTE COST AND GRADIENT DESCENT

def computeCost(X, y, theta):
    m = len(y)
    h = X @ theta.T
    return np.sum((h - y) ** 2) / (2 * m)

def gradientDescent(X, y, theta, alpha, iters):
    m = len(y)
    cost_history = []
    for _ in range(iters):
        gradient = (X.T @ ((X @ theta.T) - y)) / m
        theta = theta - alpha * gradient.T
        cost_history.append(computeCost(X, y, theta))
    return theta, cost_history

alpha = 0.01
iters = 400
theta, cost_history = gradientDescent(X, y, theta, alpha, iters)

print("Optimized Theta:", theta)

plt.figure(figsize=(8,6))
plt.plot(range(iters), cost_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Iterations')
plt.show()

#%% PART C: LINEAR REGRESSION USING SCIKIT-LEARN TO COMPARE RESULTS

model = LinearRegression()
model.fit(X, y)
print("Scikit-learn coefficients:", model.coef_)

#%% PART D: Plot formula
print("Optimized Theta:")
print(theta.flatten())

formula = "Tip Amount = {:.4f}".format(theta[0, 0])
feature_names = ["Weekday", "Trip Distance", "Fare Amount"] + list(df.filter(like='time_of_day_')) + list(df.filter(like='PULocationID_')) + list(df.filter(like='RatecodeID_')) + list(df.filter(like='payment_type_'))
for i, feature in enumerate(feature_names):
    formula += " + ({:.4f} * {})".format(theta[0, i+1], feature)

print("Regression Formula:")
print(formula)


# %% PART E: PREDICTION FUNCTION

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

predict_tip_amount()



