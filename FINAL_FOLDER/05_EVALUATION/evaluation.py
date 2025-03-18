import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pyarrow import parquet as pq

def accuracy_within_threshold(y_true, y_pred, threshold=1.0):
    """
    Custom accuracy function for regression:
    Returns the fraction of predictions within `threshold` dollars of the actual value.
    """
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# %% 1. LOAD TEST DATA
green_test_path = "FINAL_FOLDER/00_DATA/green_taxi_data_test.parquet"
yellow_test_path = "FINAL_FOLDER/00_DATA/yellow_taxi_data_test.parquet"

green_test_df = pd.read_parquet(green_test_path)
yellow_test_df = pd.read_parquet(yellow_test_path)

# Define feature sets for test data using the same columns as the training data
X_green_test_full = np.nan_to_num(green_test_df[['fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
                                                 'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
                                                 'tolls_amount']].values)
X_yellow_test_full = np.nan_to_num(yellow_test_df[['fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
                                                   'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
                                                   'tolls_amount']].values)

X_fare_trip_green_test = np.nan_to_num(green_test_df[['fare_amount', 'trip_distance', 'tolls_amount']].values)
X_fare_trip_yellow_test = np.nan_to_num(yellow_test_df[['fare_amount', 'trip_distance', 'tolls_amount']].values)

X_payment_passenger_green_test = np.nan_to_num(green_test_df[['payment_type', 'passenger_count']].values)
X_payment_passenger_yellow_test = np.nan_to_num(yellow_test_df[['payment_type', 'passenger_count']].values)

X_location_green_test = np.nan_to_num(green_test_df[['PULocationID', 'DOLocationID']].values)
X_location_yellow_test = np.nan_to_num(yellow_test_df[['PULocationID', 'DOLocationID']].values)

X_no_location_green_test = np.nan_to_num(green_test_df[['fare_amount', 'trip_distance', 'payment_type',
                                                         'passenger_count', 'RatecodeID', 'congestion_surcharge',
                                                         'tolls_amount']].values)
X_no_location_yellow_test = np.nan_to_num(yellow_test_df[['fare_amount', 'trip_distance', 'payment_type',
                                                           'passenger_count', 'RatecodeID', 'congestion_surcharge',
                                                           'tolls_amount']].values)

X_minimal_green_test = np.nan_to_num(green_test_df[['fare_amount', 'trip_distance']].values)
X_minimal_yellow_test = np.nan_to_num(yellow_test_df[['fare_amount', 'trip_distance']].values)

feature_sets_test = {
    "full": (X_green_test_full, X_yellow_test_full),
    "fare_trip": (X_fare_trip_green_test, X_fare_trip_yellow_test),
    "payment_passenger": (X_payment_passenger_green_test, X_payment_passenger_yellow_test),
    "location": (X_location_green_test, X_location_yellow_test),
    "no_location": (X_no_location_green_test, X_no_location_yellow_test),
    "minimal": (X_minimal_green_test, X_minimal_yellow_test),
}

# === Select a Feature Set for Evaluation ===
# To switch datasets, simply change the value of selected_feature_set below to one of the following options:
# "full", "fare_trip", "payment_passenger", "location", "no_location", "minimal"
selected_feature_set = "full"  # Change this value as needed

# Automatically set the test feature arrays based on the selected feature set
X_test_green = feature_sets_test[selected_feature_set][0]
X_test_yellow = feature_sets_test[selected_feature_set][1]

print(f"Selected feature set for evaluation: {selected_feature_set}")

y_test_green = np.nan_to_num(green_test_df['tip_amount'].values)

y_test_yellow = np.nan_to_num(yellow_test_df['tip_amount'].values)

# %% 2. LOAD TRAINED NETWORKS (MODELS)
# Adjust filenames to match how you saved your models
green_model_path = f"FINAL_FOLDER/04_NEURAL_NETWORK/nn_green_{selected_feature_set}.pkl"
yellow_model_path = f"FINAL_FOLDER/04_NEURAL_NETWORK/nn_yellow_{selected_feature_set}.pkl"

with open(green_model_path, "rb") as f:
    nn_green = pickle.load(f)

with open(yellow_model_path, "rb") as f:
    nn_yellow = pickle.load(f)

# %% 3. PREDICT AND EVALUATE

# --- Green Taxi ---
y_pred_green = nn_green.predict(X_test_green)

rmse_green = np.sqrt(mean_squared_error(y_test_green, y_pred_green))
mae_green = mean_absolute_error(y_test_green, y_pred_green)
r2_green = r2_score(y_test_green, y_pred_green)
acc_green = accuracy_within_threshold(y_test_green, y_pred_green, threshold=1.0)

print("\n--- Green Taxi Evaluation ---")
print(f"RMSE: {rmse_green:.2f}")
print(f"MAE: {mae_green:.2f}")
print(f"R²: {r2_green:.2f}")
print(f"Accuracy (±$1): {acc_green:.2f}")

# --- Yellow Taxi ---
y_pred_yellow = nn_yellow.predict(X_test_yellow)

rmse_yellow = np.sqrt(mean_squared_error(y_test_yellow, y_pred_yellow))
mae_yellow = mean_absolute_error(y_test_yellow, y_pred_yellow)
r2_yellow = r2_score(y_test_yellow, y_pred_yellow)
acc_yellow = accuracy_within_threshold(y_test_yellow, y_pred_yellow, threshold=1.0)

print("\n--- Yellow Taxi Evaluation ---")
print(f"RMSE: {rmse_yellow:.2f}")
print(f"MAE: {mae_yellow:.2f}")
print(f"R²: {r2_yellow:.2f}")
print(f"Accuracy (±$1): {acc_yellow:.2f}")

# %% 4. PLOT RESULTS (OVERLAID) WITH OUTLIER REMOVAL

# Compute residuals for both taxi types
residuals_green = y_test_green - y_pred_green
residuals_yellow = y_test_yellow - y_pred_yellow

# Define a function to compute a mask for outliers using the IQR method
def get_outlier_mask(residuals):
    q1 = np.percentile(residuals, 15)
    q3 = np.percentile(residuals, 85)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (residuals >= lower_bound) & (residuals <= upper_bound)

# Get masks for green and yellow residuals
mask_green = get_outlier_mask(residuals_green)
mask_yellow = get_outlier_mask(residuals_yellow)

# Filter the arrays using the masks
y_test_green_filtered = y_test_green[mask_green]
y_pred_green_filtered = y_pred_green[mask_green]
residuals_green_filtered = residuals_green[mask_green]

y_test_yellow_filtered = y_test_yellow[mask_yellow]
y_pred_yellow_filtered = y_pred_yellow[mask_yellow]
residuals_yellow_filtered = residuals_yellow[mask_yellow]

# Further filter out negative values (since tip values cannot be negative)
mask_green_nonnegative = (y_test_green_filtered >= 0) & (y_pred_green_filtered >= 0)

y_test_green_filtered = y_test_green_filtered[mask_green_nonnegative]
y_pred_green_filtered = y_pred_green_filtered[mask_green_nonnegative]
residuals_green_filtered = residuals_green_filtered[mask_green_nonnegative]

mask_yellow_nonnegative = (y_test_yellow_filtered >= 0) & (y_pred_yellow_filtered >= 0)

y_test_yellow_filtered = y_test_yellow_filtered[mask_yellow_nonnegative]
y_pred_yellow_filtered = y_pred_yellow_filtered[mask_yellow_nonnegative]
residuals_yellow_filtered = residuals_yellow_filtered[mask_yellow_nonnegative]

# Overlay Scatter Plot: Actual vs Predicted for both Green and Yellow (using filtered data)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_green_filtered, y_pred_green_filtered, alpha=0.5, color='green', label='Green Taxis')
plt.scatter(y_test_yellow_filtered, y_pred_yellow_filtered, alpha=0.5, color='gold', label='Yellow Taxis')
# Determine combined min and max for the diagonal line from filtered data
combined_min = min(y_test_green_filtered.min(), y_test_yellow_filtered.min())
combined_max = max(y_test_green_filtered.max(), y_test_yellow_filtered.max())
plt.plot([combined_min, combined_max], [combined_min, combined_max], 'k--', lw=2)
plt.xlabel('Actual Tip Amount [$]')
plt.ylabel('Predicted Tip Amount [$]')
plt.title('Actual vs Predicted Tip')
plt.legend()

# Add a single text annotation for both green and yellow metrics
metrics_text = (
    f"Green Taxi:\n"
    f"RMSE: {rmse_green:.2f}\nR²: {r2_green:.2f}\n\n"
    f"Yellow Taxi:\n"
    f"RMSE: {rmse_yellow:.2f}\nR²: {r2_yellow:.2f}"
)

# Place both metrics in one box in the upper left
plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Overlay Histogram: Residual Distribution for both Green and Yellow (using filtered data)
plt.subplot(1, 2, 2)
plt.hist(residuals_green_filtered, bins=30, alpha=0.5, color='green', label='Green Taxis')
plt.hist(residuals_yellow_filtered, bins=30, alpha=0.5, color='gold', label='Yellow Taxis')
plt.xlabel('Residuals (Actual - Predicted) [$]')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.legend()

plt.tight_layout()
plt.show()
