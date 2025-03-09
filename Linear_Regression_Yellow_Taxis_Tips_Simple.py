import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# !!! You have to create a folder with all the required parquet files inside. I created one for yellow and one for green taxis for 2015 -2022. To run the code you have to do the same

# %% PART A: LOAD AND PREPROCESS DATA

# path = r'/\Exercise1-2023\Exercise1-2023\data\ex1data1.txt'
# data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_Try"  # Path to folder

columns_to_keep = ["trip_distance", "fare_amount", "tip_amount"]

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

# Remove NaN and filter invalid values
df = df.dropna()
df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0) & (df["tip_amount"] > 0)]

# Define input features and target variable
X = df[["trip_distance", "fare_amount"]]
y = df["tip_amount"]

# Convert to numpy arrays
X = np.asarray(X, dtype=np.float64)
y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=1)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y, ddof=1)
y = (y - y_mean) / y_std

# Add bias term
X = np.column_stack((np.ones(X.shape[0]), X))
theta = np.zeros((1, X.shape[1]))


# %% PART B: COMPUTE COST AND GRADIENT DESCENT


from ReadData import computeCost
from ReadData import gradientDescent

alpha = 0.01
iters = 400
theta, cost_history = gradientDescent(X, y, theta, alpha, iters)

print("Optimized Theta:", theta)

plt.figure(figsize=(8, 6))
plt.plot(range(iters), cost_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Iterations')
plt.show()

# %% PART C: LINEAR REGRESSION USING SCIKIT-LEARN TO COMPARE RESULTS

model = LinearRegression()
model.fit(X, y)
print("Scikit-learn coefficients:", model.coef_)

# %% PART D: PLOT REGRESSION FORMULA

print("Optimized Theta:")
print(theta.flatten())

formula = "Tip Amount = {:.4f}".format(theta[0, 0])
feature_names = ["Trip Distance", "Fare Amount"]
for i, feature in enumerate(feature_names):
    formula += " + ({:.4f} * {})".format(theta[0, i + 1], feature)

print("Regression Formula:")
print(formula)


# %% PART E: PREDICTION FUNCTION

from ReadData import predict_tip_amount_s
predict_tip_amount_s()
