import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#!!! You have to create a folder with all the required parquet files inside. I created one for yellow and one for green taxis for 2015 -2022. To run the code you have to do the same
#!!!For the green taxis the cell related to the pickup time is called "tpep_pickup_datetime", while for yellow taxis "lpep_pickup_datetime"

#%% PART A: LOAD AND PREPROCESS DATA


DATA_DIR = r"/Users/lukas/Library/CloudStorage/OneDrive-Personal/Documents/TU Delft/03_EFPT/Q3/05_ML_for_Transport_and_Multi_Machine_Systems/06_Software_and_Programming/Data" # path to folder in order to read data


columns_to_keep = ["tpep_pickup_datetime", "RatecodeID", "trip_distance", "fare_amount"] # we just need a few columns from the entire dataset


all_data = [] # we read the parquet file and add the columns to keep

for file in os.listdir(DATA_DIR):
    if file.endswith(".parquet"):
        file_path = os.path.join(DATA_DIR, file)
        df = pd.read_parquet(file_path)
        df = df.loc[:, df.columns.intersection(columns_to_keep)]
        all_data.append(df)


if all_data: # We put together all years
    df = pd.concat(all_data, ignore_index=True)
else:
    raise ValueError("No data files found.")


df = df.dropna() # We decided to remove rows where one of our inputs is not given

# In the following we applied Feature Engineering to extract the weekday or weekend
df["weekday"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.weekday < 5  # True for weekday, False for weekend
df["weekday"] = df["weekday"].astype(int)  # Convert boolean to int (1 = weekday, 0 = weekend)

X = df[["weekday", "RatecodeID", "trip_distance"]] # We define the variables X1,X2,X3 of our hypothesis function
y = df["fare_amount"] # The inputs generate the output h = Y

X = np.array(X) # We want to use numpy which is why we need to convert the arrays
y = np.array(y).reshape(-1, 1)


X_mean = X.mean(axis=0) # We apply a so called feature scaling to normalize
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std


X = np.column_stack((np.ones(X.shape[0]), X)) #Add an intercept term (initially a vector of all ones) in our dataset --> theta = 1 for each x value

theta = np.zeros((1, X.shape[1])) # We initialize the parameters theta that will be improved with gradient descent

#%% PART B: COMPUTE COST AND GRADIENT DESCENT

def computeCost(X, y, theta):
    m = len(y)
    h = X @ theta.T
    return np.sum((h - y) ** 2) / (2 * m) # We create a function to compute the Cost function J(theta) = 1/2*m * sum for i = 1 to m of residual; m = number of training point = number of X rows


def gradientDescent(X, y, theta, alpha, iters): #optimize theta using gradient descent
    m = len(y)
    cost_history = []
    for _ in range(iters): #thetaj is computed iteratively for the hypothesis function
        gradient = (X.T @ ((X @ theta.T) - y)) / m
        theta = theta - alpha * gradient.T
        cost_history.append(computeCost(X, y, theta))
    return theta, cost_history


alpha = 0.01  # The learning rate has to be defined to train the model
iters = 1000  # Number of Iterations 1000 to be sure to converge
theta, cost_history = gradientDescent(X, y, theta, alpha, iters) # We execute the function and print the result afterwards

print("Optimized Theta:", theta)


plt.figure(figsize=(8,6)) # We create a figure to plot the cost function
plt.plot(range(iters), cost_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Iterations')
plt.show()

#%% PART C: LINEAR REGRESSION USING SCIKIT-LEARN TO COMPARE IF WHAT WE COMPUTED IS RIGHT

# We did the same what we did in part B, but now with a preinstalled library
model = LinearRegression()
model.fit(X, y)
print("Scikit-learn coefficients:", model.coef_)

#%% PART D: DISPLAY REGRESSION FORMULA

# We extract the final theta values to display our regression formula that can than be used to predict the fare_amount once the "lpep_pickup_datetime", "RatecodeID", "trip_distance" are known
theta0, theta1, theta2, theta3 = theta.flatten()

# We print the regression formula
print("Regression Formula:")
print(f"Fare Amount = {theta0:.4f} + ({theta1:.4f} * Weekday) + ({theta2:.4f} * RatecodeID) + ({theta3:.4f} * Trip Distance)")





