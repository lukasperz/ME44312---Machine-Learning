import numpy as np
import pandas as pd
import datetime as datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle
import sys
import traceback
import os

# %% Debugging log file & System Check before training
debug_log_path = "debug_log.txt"

def log_debug(message):
    """Logs debugging messages to both console and file."""
    print(message, flush=True)
    with open(debug_log_path, "a") as f:
        f.write(message + "\n")

# %% Read Data
data_green = pd.read_parquet('green_taxi_data.parquet')
data_yellow = pd.read_parquet('yellow_taxi_data.parquet')

# List of categorical columns to encode (Turns String values into Integers)
categorical_features = ['payment_type', 'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID']

# Apply Label Encoding to categorical features in both datasets
LE = LabelEncoder()

for col in categorical_features:
    if col in data_green.columns:
        data_green[col] = LE.fit_transform(data_green[col])
    if col in data_yellow.columns:
        data_yellow[col] = LE.fit_transform(data_yellow[col])

# %% Defining the various Inputs X and the output y for both green and yellow data sets

data_green.head()
data_yellow.head()

# Define target variable (tip amount) for both datasets
y_green = data_green['tip_amount'].values
y_yellow = data_yellow['tip_amount'].values

# %% FEATURE SETS Definition
# Define different feature sets for both datasets

# Full feature set using all potentially useful variables
X_green = data_green[['fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
                      'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
                      'tolls_amount']].values

X_yellow = data_yellow[['fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
                        'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
                        'tolls_amount']].values

# Feature set focusing on fare and trip details
X_fare_trip_green = data_green[['fare_amount', 'trip_distance', 'tolls_amount']].values
X_fare_trip_yellow = data_yellow[['fare_amount', 'trip_distance', 'tolls_amount']].values

# Feature set focusing on payment and passenger details
X_payment_passenger_green = data_green[['payment_type', 'passenger_count']].values
X_payment_passenger_yellow = data_yellow[['payment_type', 'passenger_count']].values

# Feature set with location-based data
X_location_green = data_green[['PULocationID', 'DOLocationID']].values
X_location_yellow = data_yellow[['PULocationID', 'DOLocationID']].values

# Feature set excluding location but keeping financial and trip details
X_no_location_green = data_green[['fare_amount', 'trip_distance', 'payment_type', 
                                  'passenger_count', 'RatecodeID', 'congestion_surcharge', 
                                  'tolls_amount']].values

X_no_location_yellow = data_yellow[['fare_amount', 'trip_distance', 'payment_type', 
                                    'passenger_count', 'RatecodeID', 'congestion_surcharge', 
                                    'tolls_amount']].values

# Minimal feature set for quick testing
X_minimal_green = data_green[['fare_amount', 'trip_distance']].values
X_minimal_yellow = data_yellow[['fare_amount', 'trip_distance']].values

# Define all feature sets
feature_sets = {
    "full": (X_green, X_yellow),
    "fare_trip": (X_fare_trip_green, X_fare_trip_yellow),
    "payment_passenger": (X_payment_passenger_green, X_payment_passenger_yellow),
    "location": (X_location_green, X_location_yellow),
    "no_location": (X_no_location_green, X_no_location_yellow),
    "minimal": (X_minimal_green, X_minimal_yellow),
}

# Loop over every feature set and train models
for FEATURE_SET, (X_feat_green, X_feat_yellow) in feature_sets.items():
    log_debug(f"Starting training for feature set: {FEATURE_SET}")

    # Split green dataset
    X_train_green, X_test_green, y_train_green, y_test_green = train_test_split(X_feat_green, y_green, test_size=0.2, random_state=0)
    X_train_green, X_val_green, y_train_green, y_val_green = train_test_split(X_train_green, y_train_green, test_size=0.25, random_state=0)  # 25% of 80% = 20%

    # Split yellow dataset
    X_train_yellow, X_test_yellow, y_train_yellow, y_test_yellow = train_test_split(X_feat_yellow, y_yellow, test_size=0.2, random_state=0)
    X_train_yellow, X_val_yellow, y_train_yellow, y_val_yellow = train_test_split(X_train_yellow, y_train_yellow, test_size=0.25, random_state=0)

    # Remove NaN values from training and test data
    X_train_green = np.nan_to_num(X_train_green)
    y_train_green = np.nan_to_num(y_train_green)
    X_test_green = np.nan_to_num(X_test_green)
    y_test_green = np.nan_to_num(y_test_green)

    X_train_yellow = np.nan_to_num(X_train_yellow)
    y_train_yellow = np.nan_to_num(y_train_yellow)
    X_test_yellow = np.nan_to_num(X_test_yellow)
    y_test_yellow = np.nan_to_num(y_test_yellow)

    log_debug("NaN values replaced in dataset.")

    # Scale data using StandardScaler for green taxi data
    scaler_green = StandardScaler().fit(X_train_green)
    X_train_green = scaler_green.transform(X_train_green)
    X_val_green = scaler_green.transform(X_val_green)
    X_test_green = scaler_green.transform(X_test_green)

    # Scale data using StandardScaler for yellow taxi data
    scaler_yellow = StandardScaler().fit(X_train_yellow)
    X_train_yellow = scaler_yellow.transform(X_train_yellow)
    X_val_yellow = scaler_yellow.transform(X_val_yellow)
    X_test_yellow = scaler_yellow.transform(X_test_yellow)

    # Create directory for models of this feature set
    model_dir = f"NN_Models_{FEATURE_SET}"
    os.makedirs(model_dir, exist_ok=True)

    # Save test datasets
    np.save(os.path.join(model_dir, f"X_test_green_{FEATURE_SET}.npy"), X_test_green)
    np.save(os.path.join(model_dir, f"y_test_green_{FEATURE_SET}.npy"), y_test_green)
    np.save(os.path.join(model_dir, f"X_test_yellow_{FEATURE_SET}.npy"), X_test_yellow)
    np.save(os.path.join(model_dir, f"y_test_yellow_{FEATURE_SET}.npy"), y_test_yellow)
    print(f"Test datasets saved successfully in {model_dir} with feature set: {FEATURE_SET}.")

    log_debug(f"Training started... Using feature set: {FEATURE_SET}")
    log_debug(f"Dataset sizes: Green {X_train_green.shape}, Yellow {X_train_yellow.shape}")

    # Reshape targets
    y_train_green = y_train_green.reshape(-1, 1)
    y_train_yellow = y_train_yellow.reshape(-1, 1)

    if np.isnan(X_train_green).any() or np.isnan(y_train_green).any():
        log_debug("Warning: NaN values detected in Green dataset!")
    if np.isnan(X_train_yellow).any() or np.isnan(y_train_yellow).any():
        log_debug("Warning: NaN values detected in Yellow dataset!")

    log_debug("Preprocessing complete. Starting training...")

    # Training the model for Green Taxi data
    log_debug("Training Green Taxi Model...")
    nn_green = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=0
    )
    nn_green.fit(X_train_green, y_train_green.ravel())
    log_debug("Green Taxi Model Training Complete!")

    # Training the model for Yellow Taxi data
    log_debug("Training Yellow Taxi Model...")
    nn_yellow = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=0
    )
    nn_yellow.fit(X_train_yellow, y_train_yellow.ravel())
    log_debug("Yellow Taxi Model Training Complete!")

    log_debug("Saving models...")
    with open(os.path.join(model_dir, f"nn_green_{FEATURE_SET}.pkl"), "wb") as f:
        pickle.dump(nn_green, f)
    with open(os.path.join(model_dir, f"nn_yellow_{FEATURE_SET}.pkl"), "wb") as f:
        pickle.dump(nn_yellow, f)
    log_debug("Models saved successfully.")

    # Evaluate the models on test data
    log_debug("Evaluating Models...")
    y_pred_green = nn_green.predict(X_test_green)
    y_pred_yellow = nn_yellow.predict(X_test_yellow)
    test_loss_green = np.mean((y_pred_green - y_test_green) ** 2)
    test_loss_yellow = np.mean((y_pred_yellow - y_test_yellow) ** 2)
    log_debug(f"\nGreen Taxi Model - Test Loss (MSE): {test_loss_green:.6f}")
    log_debug(f"Yellow Taxi Model - Test Loss (MSE): {test_loss_yellow:.6f}")
    log_debug(f"Completed training for feature set: {FEATURE_SET}\n")

log_debug("Script completed successfully.")
