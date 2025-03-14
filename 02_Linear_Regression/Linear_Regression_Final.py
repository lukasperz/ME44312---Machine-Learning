import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

base_dir = os.getcwd()                                                                                                                          # Get the current working directory

# Define the relative path to the file (it has to be in your directory)
file_path = os.path.join(base_dir, "yellow_taxi_data.parquet")

# # To check if the file exists locally, otherwise raise an error
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_parquet(file_path)                                                                                                               # Load the dataset
data = data[['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']]                                                                  # Select the relevant columns

# To have an idea of what is going on we try to visualize the individual relationships of the output to the inputs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feature in enumerate(['trip_distance', 'fare_amount', 'passenger_count']):
    axes[i].scatter(data[feature], data['tip_amount'])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Tip Amount')
    # if feature == 'trip_distance':
    #     axes[i].set_xlim(0, data[feature].quantile(0.99))                                                                                       # Limit to 99th percentile to remove outliers, because we saw that some single points where far off the majority
    #     axes[i].ticklabel_format(style='plain', axis='x')                                                                                       # Ensure readable formatting
plt.show()

# %% ----- Setup X and Y -----

# Initializing the variables, setting X (input data) and y (target variable)
features = ['trip_distance', 'fare_amount', 'passenger_count']
target = 'tip_amount'

X = data[features].values                                                                                                                       # Extract feature values as NumPy array
y = data[target].values                                                                                                                         # Extract target values as NumPy array

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)                                                  # Split the data in train and test dataset


# %% ----- Linear Regression using SCIKIT-LEARN -----

model = LinearRegression()
model.fit(X_train, y_train)

# Get the model coefficients
theta0 = model.intercept_                                                                                                                    # Intercept term
theta1, theta2, theta3 = model.coef_                                                                                                         # Directly extract all three coefficients

# Print the regression formula
print(f"Regression Equation: tip_amount = {theta0:.4f} + {theta1:.4f} * trip_distance + {theta2:.4f} * fare_amount + {theta3:.4f} * passenger_count")

# %% ----- Evaluate our model -----

y_pred = theta0 + (theta1 * X_val[:, 0]) + (theta2 * X_val[:, 1]) + (theta3 * X_val[:, 2])                                                      # Predicting the tip_amount with our trained model
y_pred = np.maximum(y_pred, 0)                                                                                                                  # The predicted tip amount has to be at least zero

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_val, y_pred)                                                                                                        # MAE = (1/n) * sum from 1 to n of (y_true - y_predicted)**2, where n = sample size
                                                                                                                                                # This tells us the model's average error per prediction in absolute numbers
# Compute R-squared Score (R**2)
r2 = r2_score(y_val, y_pred)                                                                                                                    # R**2 = 1 - sum(y_true - y-pred)**2 / sum(y_true - ymean)**2
                                                                                                                                                # This is a metric that measures how well the regression model firs the data
# Print and plot accuracy metrics                                                                                                               # If R**2 = 1 --> perfect model; R**2 = 0.9 --> good model; R**2 = 0.5 --> explains 50% of variance (not good); R**2 = 0 --> model explains nothing (random guess); R**2 < 0 --> model is worse than guessing the mean;
print(f"\nModel Performance on Sampled Comparison Data:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Pred. amount = Act. amount')
text_str = f"MAE: {mae:.4f}\nR² Score: {r2:.4f}"
plt.text(0.05, 0.9, text_str, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tip Amount for Comparison Data")
plt.legend()
plt.show()