import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
import os

#%% PART A: LOAD AND PLOT THE DATA

DATA_DIR = r"C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\DATA_TRAINING_GREEN" # path to folder in order to read data

columns_to_keep = ["fare_amount", "tip_amount", "payment_type"] # We just need a few columns from the entire dataset

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

df = df.dropna() # We remove rows with missing values and keep only payment types 1 and 2 (which are the only ones present)
df = df[df["payment_type"].isin([1, 2])]

df["payment_type"] = df["payment_type"].replace(2, 0) # to apply our logistic regression we must have a binary output and therefore we convert payment type to binary (1 stays 1, 2 becomes 0)

## We decided not to plot the data since we would have more than 10**7 data points
# fig, ax = plt.subplots(figsize=(12, 8))
# positive = df[df["payment_type"] == 1]
# negative = df[df["payment_type"] == 0]
#
# ax.scatter(positive['fare_amount'], positive['tip_amount'], c='blue', marker='o', label='Payment Type 1')
# ax.scatter(negative['fare_amount'], negative['tip_amount'], c='red', marker='x', label='Payment Type 2')
# ax.set_xlabel('Fare Amount')
# ax.set_ylabel('Tip Amount')
# ax.set_title('Scatter Plot of Fare Amount vs Tip Amount')
# ax.legend()
# plt.show()

#%% PART B: PLOT THE SIGMOID FUNCTION

def sigmoid(z):
    z = np.clip(z, -500, 500) # we clip the z values not to generate an overflow
    return 1 / (1 + np.exp(-z)) # we define a function that returns us the sigmoid function

nums = np.arange(-10, 10, step=0.1) # Create an array of values from -10 to 10 (smaller steps for smoother curve)
sigmoid_values = sigmoid(nums) # Here we can now compute the sigmoid_function

fig, ax = plt.subplots(figsize=(12,8)) # We can plot the sigmoid_function with proper labels, title and legend
ax.plot(nums, sigmoid_values, 'r-', label='Sigmoid Function')
ax.set_xlabel('z')
ax.set_ylabel('Sigmoid(z)')
ax.set_title('Sigmoid Function Plot')
ax.legend()
plt.show()

#%% PART C: CALCULATE THE COST FUNCTION

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    h = sigmoid(X @ theta.T) # We compute the hypothesis h_theta(x)
    h = np.clip(h, 1e-9, 1 - 1e-9)  # We avoid log(0) by clipping values to very small numbers not influencing our result
    first = np.multiply(-y, np.log(h)) # y * log(h_theta(x))
    second = np.multiply((1 - y), np.log(1 - h)) # (1 - y) * log(1 - h_theta(x))
    return np.sum(first - second) / len(X) # Compute final cost value

df.insert(0, 'Ones', 1) # we add a 'ones' column for matrix multiplication (just as in exercise 1)

X = df[['Ones', 'fare_amount', 'tip_amount']].values  # we set X (training data) and y (target variable)
y = df[['payment_type']].values

theta = np.zeros(X.shape[1]) # Initialize theta to ensure that it has the right shape

cost1 = cost(theta, X, y) # We compute the initial cost
print(f"Initial Cost: {cost1}")

#%% PART D: FIND THE OPTIMAL THETA VALUES

# Compute the gradient (parameter updates) given our training data, labels, and some parameters theta
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X @ theta.T) - y # Compute error term

    for i in range(parameters): # We iterate over each parameter manually
        term = np.multiply(error, X[:, i]) # Compute gradient term
        grad[i] = np.sum(term) / len(X) # Average gradient term

    return grad

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y)) # We find the optimized theta values with SciPy's truncated Newton (TNC)
theta_optimized = np.matrix(result[0])
cost2 = cost(theta_optimized, X, y)
print(f"Optimized Cost: {cost2}")

#%% PART E: PREDICTION ACCURACY

# Function to make predictions based on learned parameters
def predict(theta, X):
    probability = sigmoid(X @ theta.T) # Compute probability using sigmoid function
    return [1 if x >= 0.5 else 0 for x in probability]  # Convert probabilities to 0 or 1

# Use the optimized theta values from Part D
predictions = predict(theta_optimized, X) # Generate predictions

# Compute accuracy by comparing predictions with actual values
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(correct) / len(correct)) * 100 # Compute accuracy percentage

# Print the accuracy
print(f'Prediction Accuracy: {accuracy:.2f}%')

#%% PART F: LOGISTIC REGRESSION WITH SCIKIT-LEARN

# Initialize the model with SCIKIT-LEARN
model = LogisticRegression()

# Fit the model using the training data
model.fit(X, y.ravel()) # Ensure y is in the correct shape using .ravel()

# Print the accuracy of the model on the training data
print(f'Scikit-learn Model Accuracy: {model.score(X, y) * 100:.2f}%')

#%% PART G: PLOT DECISION BOUNDARY

# Define that payment type 1 is positive
positive = df[df["payment_type"] == 1]
negative = df[df["payment_type"] == 0]


# Create a new figure for the decision boundary
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['fare_amount'], positive['tip_amount'], c='blue', marker='o', label='Payment Type 1')
ax.scatter(negative['fare_amount'], negative['tip_amount'], c='red', marker='x', label='Payment Type 2')

# Compute the decision boundary
x_values = np.linspace(df['fare_amount'].min(), df['fare_amount'].max(), 100)
y_values = -(theta_optimized[0, 0] + theta_optimized[0, 1] * x_values) / theta_optimized[0, 2]

# Plot the decision boundary
ax.plot(x_values, y_values, 'g-', label='Decision Boundary')

# Labels and legend
ax.set_xlabel('Fare Amount')
ax.set_ylabel('Tip Amount')
ax.set_title('Decision Boundary for Logistic Regression')
ax.legend()

# Show the new plot
plt.show()
