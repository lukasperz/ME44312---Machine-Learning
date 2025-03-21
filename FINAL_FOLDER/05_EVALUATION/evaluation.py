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

# --- "old" feature set: without time-related columns ---
X_green_test_old = np.nan_to_num(green_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values)
X_yellow_test_old = np.nan_to_num(yellow_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values)

# Updated "full" feature set includes time-related columns
X_green_test_full = np.nan_to_num(green_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount', 'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values)
X_yellow_test_full = np.nan_to_num(yellow_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount', 'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values)

# No location feature set (unchanged)
X_no_location_green_test = np.nan_to_num(green_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'RatecodeID', 'congestion_surcharge', 'tolls_amount'
]].values)
X_no_location_yellow_test = np.nan_to_num(yellow_test_df[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'RatecodeID', 'congestion_surcharge', 'tolls_amount'
]].values)

# Minimal feature set (unchanged)
X_minimal_green_test = np.nan_to_num(green_test_df[['fare_amount', 'trip_distance']].values)
X_minimal_yellow_test = np.nan_to_num(yellow_test_df[['fare_amount', 'trip_distance']].values)

# New time_features set (only time-related columns)
X_time_green_test = np.nan_to_num(green_test_df[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values)
X_time_yellow_test = np.nan_to_num(yellow_test_df[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values)

# Define the test feature sets dictionary with the new "old" set added
feature_sets_test = {
    "old": (X_green_test_old, X_yellow_test_old),
    "full": (X_green_test_full, X_yellow_test_full),
    "no_location": (X_no_location_green_test, X_no_location_yellow_test),
    "minimal": (X_minimal_green_test, X_minimal_yellow_test),
    "time_features": (X_time_green_test, X_time_yellow_test)
}

# === Select a Feature Set for Evaluation ===
# Options: "old", "full", "no_location", "minimal", "time_features"
selected_feature_set = "full"  

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
combined_min = min(y_test_green_filtered.min(), y_test_yellow_filtered.min())
combined_max = max(y_test_green_filtered.max(), y_test_yellow_filtered.max())
plt.plot([combined_min, combined_max], [combined_min, combined_max], 'k--', lw=2)
plt.xlabel('Actual Tip Amount [$]')
plt.ylabel('Predicted Tip Amount [$]')
plt.title('Actual vs Predicted Tip')
plt.legend()

# Add a single text annotation for both green and yellow metrics
metrics_text = (
    f"Green Taxi:\nRMSE: {rmse_green:.2f}\nR²: {r2_green:.2f}\nMAE: {mae_green:.2f}\n\n"
    f"Yellow Taxi:\nRMSE: {rmse_yellow:.2f}\nR²: {r2_yellow:.2f}\nMAE: {mae_yellow:.2f}"
)
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

# %% 5. PERMUTATION-BASED FEATURE IMPORTANCE
from sklearn.inspection import permutation_importance

# Define a dictionary to map each feature set to its corresponding column names
feature_names_dict = {
    'old': [
        'Fare Amount', 'Trip Distance', 'Payment Type', 'Passenger Count',
        'Pick Up Location', 'Drop Off Location', 'Ratecode-ID', 'Surcharges', 'Tolls Amount'
    ],
    'full': [
        'Fare Amount', 'Trip Distance', 'Payment Type', 'Passenger Count',
        'Pick Up Location', 'Drop Off Location', 'Ratecode-ID', 'Surcharges',
        'Tolls Amount', 'Pickup Hour', 'Dropoff Hour', 'Pickup Day of Week'
    ],
    'no_location': [
        'Fare Amount', 'Trip Distance', 'Payment Type', 'Passenger Count',
        'Ratecode-ID', 'Surcharges', 'Tolls Amount'
    ],
    'minimal': [
        'Fare Amount', 'Trip Distance'
    ],
    'time_features': [
        'Pickup Hour', 'Dropoff Hour', 'Pickup Day of Week'
    ]
}

# Retrieve the chosen feature names based on the selected feature set
chosen_feature_names = feature_names_dict[selected_feature_set]

# --- Compute Permutation Importances for Both Taxi Types ---
results_green = permutation_importance(
    nn_green,
    X_test_green,
    y_test_green,
    scoring="neg_mean_squared_error",
    n_repeats=5,
    random_state=42
)
importances_green = results_green.importances_mean
stds_green = results_green.importances_std

results_yellow = permutation_importance(
    nn_yellow,
    X_test_yellow,
    y_test_yellow,
    scoring="neg_mean_squared_error",
    n_repeats=5,
    random_state=42
)
importances_yellow = results_yellow.importances_mean
stds_yellow = results_yellow.importances_std

# --- Plot Grouped Bar Chart for Comparison in Sorted Order ---
avg_importances = (importances_green + importances_yellow) / 2
sorted_idx = np.argsort(avg_importances)[::-1]  # descending order

importances_green_sorted = importances_green[sorted_idx]
importances_yellow_sorted = importances_yellow[sorted_idx]
feature_names_sorted = [chosen_feature_names[i] for i in sorted_idx]

indices = np.arange(len(chosen_feature_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(indices - width/2, importances_green_sorted, width,
        label='Green Taxi', color='green', alpha=0.7)
plt.bar(indices + width/2, importances_yellow_sorted, width,
        label='Yellow Taxi', color='gold', alpha=0.7)

plt.xticks(indices, feature_names_sorted, rotation=45, ha='right')
plt.ylabel('Mean Importance Increase')
plt.title('Permutation Importances Comparison')
plt.legend()
plt.tight_layout()
plt.show()