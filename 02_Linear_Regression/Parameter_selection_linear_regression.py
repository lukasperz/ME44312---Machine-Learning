import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#%% PART A: LOAD AND PREPROCESS DATA

# The correct directory is defined
data_dir = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\ME44312---Machine-Learning"

if not os.path.isdir(data_dir):
    raise ValueError(f"Directory not found: {data_dir}")

# We just want to keep certain columns
columns_to_keep = ["tpep_pickup_datetime", "RatecodeID", "trip_distance", "fare_amount", "PULocationID",
                   "payment_type", "tip_amount", "passenger_count"]

all_data = []

# Iterate through all .parquet files in the directory
for file in os.listdir(data_dir):
    if file.endswith("yellow_taxi_data.parquet"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_parquet(file_path)
        df = df.loc[:, df.columns.intersection(columns_to_keep)]
        all_data.append(df)

# Here we concatenate data from all parquet files (initially we had multiple parquet files)
if all_data:
    df = pd.concat(all_data, ignore_index=True)
else:
    raise ValueError("No data files found in the directory.")

# Preprocessing steps with one hot encoding
df = df.dropna()
df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0) & (df["tip_amount"] > 0) & (df["passenger_count"] > 0)]

pickup_time = pd.to_datetime(df["tpep_pickup_datetime"])
df["weekday"] = (pickup_time.dt.weekday < 5).astype(int)  # 1 if weekday, 0 if weekend

bins = [0, 6, 12, 16, 20, 24]
labels = ["night", "morning", "noon", "afternoon", "evening"]
df["time_of_day"] = pd.cut(pickup_time.dt.hour, bins=bins, labels=labels, right=False)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)
df = pd.get_dummies(df, columns=["PULocationID", "RatecodeID", "payment_type"], drop_first=True)

# Prepare X (features) and y (target variable)
X = df[["weekday", "trip_distance", "fare_amount", "passenger_count"] + list(df.filter(like='time_of_day_')) +
       list(df.filter(like='PULocationID_')) + list(df.filter(like='RatecodeID_')) +
       list(df.filter(like='payment_type_'))]

y = df["tip_amount"]

# Convert to numpy arrays
X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
X_std[X_std == 0] = 1  # Avoid division by zero
X = (X - X_mean) / X_std

# Normalize y (Fixed `y_std` issue)
y_mean = np.mean(y)
y_std = np.std(y, ddof=1)
y_std = 1 if y_std == 0 else y_std  # Ensure y_std is never zero
y = (y - y_mean) / y_std

# Add bias term (column of ones) to X
X = np.column_stack((np.ones(X.shape[0]), X))

# Initialize theta
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
feature_names = ["Weekday", "Trip Distance", "Fare Amount", "Passenger Count"] + list(df.filter(like='time_of_day_')) + list(df.filter(like='PULocationID_')) + list(df.filter(like='RatecodeID_')) + list(df.filter(like='payment_type_'))
for i, feature in enumerate(feature_names):
    formula += " + ({:.4f} * {})".format(theta[0, i+1], feature)

print("Regression Formula:")
print(formula)

