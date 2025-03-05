import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#!!! You have to create a folder with all the required parquet files inside. I created one for yellow and one for green taxis for 2015 -2022. To run the code you have to do the same
#!!!For the green taxis the cell related to the pickup time is called "tpep_pickup_datetime", while for yellow taxis "lpep_pickup_datetime"

#%% PART A: LOAD AND PREPROCESS DATA

DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_Try" # Path to folder

columns_to_keep = ["tpep_pickup_datetime", "RatecodeID", "trip_distance", "fare_amount", "PULocationID", "payment_type", "tip_amount"]

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

# We ensure that all independent variables are greater than 0 since otherwise also if a certain input is present in the cell it is not physical
df = df[(df["RatecodeID"] > 0) & (df["trip_distance"] > 0) & (df["fare_amount"] > 0) & (df["PULocationID"] > 0) & (df["payment_type"] > 0) & (df["tip_amount"] > 0)]

# Since the time is no number per se we divide it into the categories weekday and weekend (1 or 0) and morning, noon, afternoon, evening and night (principle: 0 0 1 0 0 if happened during afternoon)
pickup_time = pd.to_datetime(df["tpep_pickup_datetime"])
df["weekday"] = pickup_time.dt.weekday < 5  # 1 if weekday, 0 if weekend
df["weekday"] = df["weekday"].astype(int)

# Categorize time of day
bins = [0, 6, 12, 16, 20, 24]
labels = ["night", "morning", "noon", "afternoon", "evening"] # This associates a 1 for the time_of_day where the trip happens and a zero for all others. If the trip happens at night then all are zero.
df["time_of_day"] = pd.cut(pickup_time.dt.hour, bins=bins, labels=labels, right=False)

df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)  # One-hot encoding: used to assign numbers to the bins chosen

# Define X (independent variables) and y (dependent variable)
X = df[["weekday", "RatecodeID", "trip_distance", "fare_amount", "PULocationID", "payment_type"] + list(df.filter(like='time_of_day_'))] # We define the variables X1,X2,X3 of our hypothesis function
y = df["tip_amount"] # The inputs generate the output h = Y

X = np.asarray(X, dtype=np.float64) # We convert the array into a NumPy array and force it to be a float #do: do we have non float before that is than forced transformed into float?
y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

print("X shape:", X.shape) # Just checking the shape to see if we really get arrays of independent variables for one dependent variable
print("y shape:", y.shape)

X_mean = np.mean(X, axis=0) # We apply a so called feature scaling to normalize
X_std = np.std(X, axis=0, ddof=1)

# Ensuring that the array stay an array and if we would have a division by zero we set it to a division by one #do: How does this affect the calculation/prediction
X_std = np.atleast_1d(X_std)
X_std[X_std == 0] = 1

X = (X - X_mean) / X_std # We apply a so called feature scaling to normalize

y_mean = np.mean(y)
y_std = np.std(y, ddof=1)

# Ensuring that the array stay an array and if we would have a division by zero we set it to a division by one #do: How does this affect the calculation/prediction
y_std = np.atleast_1d(y_std)
y_std[y_std == 0] = 1

y = (y - y_mean) / y_std

X = np.column_stack((np.ones(X.shape[0]), X))
theta = np.zeros((1, X.shape[1]))

#%% PART B: COMPUTE COST AND GRADIENT DESCENT

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

alpha = 0.01 # The learning rate has to be defined to train the model
iters = 400 # Number of Iterations 1000 to be sure to converge
theta, cost_history = gradientDescent(X, y, theta, alpha, iters) # We execute the function and print the result afterwards

print("Optimized Theta:", theta)

plt.figure(figsize=(8,6)) # We create a figure to plot the cost function
plt.plot(range(iters), cost_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Iterations')
plt.show()

#%% PART C: LINEAR REGRESSION USING SCIKIT-LEARN TO COMPARE RESULTS

# We did the same what we did in part B, but now with a preinstalled library
model = LinearRegression()
model.fit(X, y)
print("Scikit-learn coefficients:", model.coef_)

#%% PART D: DISPLAY REGRESSION FORMULA

# We extract the final theta values to display our regression formula that can than be used to predict the tip_amount
theta_values = theta.flatten()
print("Regression Formula:")
formula = "Tip Amount = {:.4f}".format(theta_values[0])
feature_names = ["Weekday", "RatecodeID", "Trip Distance", "Fare Amount", "PULocationID", "Payment Type"] + list(df.filter(like='time_of_day_'))
for i, feature in enumerate(feature_names):
    formula += " + ({:.4f} * {})".format(theta_values[i+1], feature)

# We print the regression formula
print(formula)

#%% PART E: Evaluation for random point

def predict_tip_amount():
    while True:
        print("\n=== Predict Tip Amount ===")

        # User inputs
        weekday = int(input("Enter Weekday (1 for weekday, 0 for weekend): "))
        ratecode_id = float(input("Enter RatecodeID: "))
        trip_distance = float(input("Enter Trip Distance: "))
        fare_amount = float(input("Enter Fare Amount: "))
        pu_location_id = float(input("Enter PULocationID: "))
        payment_type = float(input("Enter Payment Type: "))

        # Time of day inputs (One-Hot Encoded)
        print("\nSelect Time of Day:")
        print("1: Night (12 AM - 6 AM)")
        print("2: Morning (6 AM - 12 PM)")
        print("3: Noon (12 PM - 4 PM)")
        print("4: Afternoon (4 PM - 8 PM)")
        print("5: Evening (8 PM - 12 AM)")
        time_of_day_choice = int(input("Enter your choice (1-5): "))

        # One-hot encoding the time_of_day variables
        time_of_day_morning = 1 if time_of_day_choice == 2 else 0
        time_of_day_noon = 1 if time_of_day_choice == 3 else 0
        time_of_day_afternoon = 1 if time_of_day_choice == 4 else 0
        time_of_day_evening = 1 if time_of_day_choice == 5 else 0

        # Applying the formula
        tip_amount = (
                -0.0000 +
                (0.0166 * weekday) +
                (0.0790 * ratecode_id) +
                (0.0169 * trip_distance) +
                (0.7293 * fare_amount) +
                (-0.0165 * pu_location_id) +
                (0.0072 * payment_type) +
                (-0.0153 * time_of_day_morning) +
                (-0.0053 * time_of_day_noon) +
                (0.0157 * time_of_day_afternoon) +
                (0.0016 * time_of_day_evening)
        )

        print(f"\nPredicted Tip Amount: {tip_amount:.4f}")

        # Ask if the user wants to run again
        choice = input("\nDo you want to predict another tip amount? (yes/no): ").strip().lower()
        if choice != 'yes':
            print("Exiting program. Goodbye!")
            break


# Run the function
predict_tip_amount()

