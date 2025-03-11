import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

path = r'C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\ME44312---Machine-Learning\sampled_yellow_data.parquet'   # Load the dataset from the parquet file
data = pd.read_parquet(path)                                                                                                                    # Read the loaded file

data = data[['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']]                                                                  # Select the relevant columns
data.head()

data.describe()

# Handle NaN values
data = data.dropna()                                                                                                                            # Remove rows with NaN values

# To have an idea of what is going on we try to visualize the individual relationships of the output to the inputs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feature in enumerate(['trip_distance', 'fare_amount', 'passenger_count']):
    axes[i].scatter(data[feature], data['tip_amount'])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Tip Amount')
    if feature == 'trip_distance':
        axes[i].set_xlim(0, data[feature].quantile(0.99))                                                                                       # Limit to 99th percentile to remove outliers, because we saw that some single points where far off the majority
        axes[i].ticklabel_format(style='plain', axis='x')                                                                                       # Ensure readable formatting
plt.show()

# %% ----- Setup X and Y -----

# Initializing the variables, setting X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]                                                                                                                    # Select features: trip_distance, fare_amount, passenger_count and associate them to the input X "vector"
y = data.iloc[:, cols - 1:cols]                                                                                                                 # Target variable: tip_amount

# %% ----- Linear Regression using SCIKIT-LEARN -----

model = LinearRegression()
model.fit(X, y)

# Get the model coefficients
theta0 = model.intercept_[0]                                                                                                                    # Intercept term
theta1, theta2, theta3 = model.coef_[0]                                                                                                         # Directly extract all three coefficients

# Print the regression formula
print(f"Regression Equation: tip_amount = {theta0:.4f} + {theta1:.4f} * trip_distance + {theta2:.4f} * fare_amount + {theta3:.4f} * passenger_count")

# %% ----- Evaluate our model -----

# In this section we validate our model by predicting the tip_amount of a new dataset with the parameters (theta) calculated with our training dataset
path_green = r'C:\Users\danie\OneDrive - Delft University of Technology\Q3\Machine Learning\ME44312---Machine-Learning\sampled_green_data.parquet'
data_green = pd.read_parquet(path_green)

data_green = data_green[['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']]                                                      # Select the same relevant columns
data_green = data_green.dropna()                                                                                                                # Remove NaN values

sample_green = data_green.sample(n=25, random_state=42)                                                                                         # Randomly select 25 samples to compare

# Extract input (X_test) and actual output (y_actual)
X_test = sample_green[['trip_distance', 'fare_amount', 'passenger_count']]                                                                      # Features
y_actual = sample_green['tip_amount']                                                                                                           # True tip_amount values

# Predicting the tip_amount with our trained model
y_pred = theta0 + (theta1 * X_test['trip_distance']) + (theta2 * X_test['fare_amount']) + (theta3 * X_test['passenger_count'])

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_actual, y_pred)                                                                                                     # MAE = (1/n) * sum from 1 to n of (y_true - y_predicted)**2, where n = sample size
                                                                                                                                                # This tells us the model's average error per prediction in absolute numbers
# Compute R-squared Score (R**2)
r2 = r2_score(y_actual, y_pred)                                                                                                                 # R**2 = 1 - sum(y_true - y-pred)**2 / sum(y_true - ymean)**2
                                                                                                                                                # This is a metric that measures how well the regression model firs the data
                                                                                                                                                # If R**2 = 1 --> perfect model; R**2 = 0.9 --> good model; R**2 = 0.5 --> explains 50% of variance (not good); R**2 = 0 --> model explains nothing (random guess); R**2 < 0 --> model is worse than guessing the mean;
# Print and plot accuracy metrics
print(f"\nModel Performance on Sampled Green Data:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_actual, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction Line')
text_str = f"MAE: {mae:.4f}\nR² Score: {r2:.4f}"
plt.text(0.05, 0.9, text_str, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tip Amount for Green Data (Sample of 25)")
plt.legend()
plt.show()