

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Load trained models
with open("nn_green_full.pkl", "rb") as f:
    nn_green = pickle.load(f)

with open("nn_yellow_full.pkl", "rb") as f:
    nn_yellow = pickle.load(f)

# Load test datasets
X_test_green = np.load("X_test_green.npy")
y_test_green = np.load("y_test_green.npy")
X_test_yellow = np.load("X_test_yellow.npy")
y_test_yellow = np.load("y_test_yellow.npy")

# Define feature names manually
feature_names = ['fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
                 'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
                 'tolls_amount']

# Ensure shape consistency before converting to DataFrame
if X_test_green.ndim == 2 and X_test_green.shape[1] == len(feature_names):
    X_test_green = pd.DataFrame(X_test_green, columns=feature_names)
elif X_test_green.ndim == 2 and X_test_green.shape[0] == len(feature_names):
    print(f"Warning: X_test_green shape {X_test_green.shape} mismatches feature_names length, transposing array.")
    X_test_green = pd.DataFrame(X_test_green.T, columns=feature_names)
else:
    print(f"Warning: X_test_green has shape {X_test_green.shape}, expected {len(feature_names)} columns.")

if X_test_yellow.ndim == 2 and X_test_yellow.shape[1] == len(feature_names):
    X_test_yellow = pd.DataFrame(X_test_yellow, columns=feature_names)
elif X_test_yellow.ndim == 2 and X_test_yellow.shape[0] == len(feature_names):
    print(f"Warning: X_test_yellow shape {X_test_yellow.shape} mismatches feature_names length, transposing array.")
    X_test_yellow = pd.DataFrame(X_test_yellow.T, columns=feature_names)
else:
    print(f"Warning: X_test_yellow has shape {X_test_yellow.shape}, expected {len(feature_names)} columns.")

# Make predictions
y_pred_green = nn_green.predict(X_test_green)
y_pred_yellow = nn_yellow.predict(X_test_yellow)

# Compute feature importance for Green Taxi
result_green = permutation_importance(nn_green, X_test_green, y_test_green, scoring="neg_mean_squared_error", random_state=0)
importance_green = result_green.importances_mean

# Compute feature importance for Yellow Taxi
result_yellow = permutation_importance(nn_yellow, X_test_yellow, y_test_yellow, scoring="neg_mean_squared_error", random_state=0)
importance_yellow = result_yellow.importances_mean

# Sort and display top features
sorted_indices_green = np.argsort(importance_green)[::-1]
sorted_indices_yellow = np.argsort(importance_yellow)[::-1]

print("\nTop 5 Features for Green Taxi Model:")
for i in range(5):
    print(f"{feature_names[sorted_indices_green[i]]}: {importance_green[sorted_indices_green[i]]:.4f}")

print("\nTop 5 Features for Yellow Taxi Model:")
for i in range(5):
    print(f"{feature_names[sorted_indices_yellow[i]]}: {importance_yellow[sorted_indices_yellow[i]]:.4f}")

# Plot Feature Importance
plt.figure(figsize=(10, 5))
plt.barh([feature_names[i] for i in sorted_indices_green], importance_green[sorted_indices_green], label="Green Taxi", alpha=0.7)
plt.barh([feature_names[i] for i in sorted_indices_yellow], importance_yellow[sorted_indices_yellow], label="Yellow Taxi", alpha=0.7)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance for Tip Prediction")
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# Plot Fare Amount vs. Tip Amount
plt.figure(figsize=(10, 5))
plt.scatter(X_test_green['fare_amount'], y_test_green, alpha=0.5, label="Green Taxi Actual", color='blue')
plt.scatter(X_test_yellow['fare_amount'], y_test_yellow, alpha=0.5, label="Yellow Taxi Actual", color='orange')
plt.xlabel("Fare Amount ($)")
plt.ylabel("Tip Amount ($)")
plt.title("Fare Amount vs. Tip Amount")
plt.legend()
plt.show()

# Plot Trip Distance vs. Tip Amount
plt.figure(figsize=(10, 5))
plt.scatter(X_test_green['trip_distance'], y_test_green, alpha=0.5, label="Green Taxi Actual", color='blue')
plt.scatter(X_test_yellow['trip_distance'], y_test_yellow, alpha=0.5, label="Yellow Taxi Actual", color='orange')
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Tip Amount ($)")
plt.title("Trip Distance vs. Tip Amount")
plt.legend()
plt.show()

# Evaluate model performance
mae_green = mean_absolute_error(y_test_green, y_pred_green)
mse_green = mean_squared_error(y_test_green, y_pred_green)
r2_green = r2_score(y_test_green, y_pred_green)

mae_yellow = mean_absolute_error(y_test_yellow, y_pred_yellow)
mse_yellow = mean_squared_error(y_test_yellow, y_pred_yellow)
r2_yellow = r2_score(y_test_yellow, y_pred_yellow)

print("\nGreen Taxi Model Performance:")
print(f"MAE: {mae_green:.2f}, MSE: {mse_green:.2f}, R²: {r2_green:.2f}")

print("\nYellow Taxi Model Performance:")
print(f"MAE: {mae_yellow:.2f}, MSE: {mse_yellow:.2f}, R²: {r2_yellow:.2f}")

# Scatter Plot of Actual vs. Predicted Tips
plt.figure(figsize=(10, 5))
plt.scatter(y_test_green, y_pred_green, alpha=0.5, label="Green Taxi", color='blue')
plt.scatter(y_test_yellow, y_pred_yellow, alpha=0.5, label="Yellow Taxi", color='orange')
plt.plot([min(y_test_green), max(y_test_green)], [min(y_test_green), max(y_test_green)], 'r', linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Tip Amount")
plt.ylabel("Predicted Tip Amount")
plt.title("Predicted vs. Actual Tips")
plt.legend()
plt.show()