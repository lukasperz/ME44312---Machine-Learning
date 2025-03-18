import numpy as np
import pandas as pd
import datetime as datetime
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from Simple_NN_Lukas import SimpleNN
import pickle
import sys
import traceback
import os
import psutil  # For memory monitoring

# Debugging log file
debug_log_path = "debug_log.txt"

def log_debug(message):
    """Logs debugging messages to both console and file."""
    print(message, flush=True)
    with open(debug_log_path, "a") as f:
        f.write(message + "\n")

# Check system memory usage before training
process = psutil.Process(os.getpid())
mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
log_debug(f"Memory Usage Before Training: {mem_usage:.2f} MB")

data_green = pd.read_parquet('green_taxi_data.parquet')
data_yellow = pd.read_parquet('yellow_taxi_data.parquet')

# List of categorical columns to encode
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

# Feature set selection (Change this to switch between feature sets)
FEATURE_SET = "full"  # Options: "full", "fare_trip", "payment_passenger", "location", "no_location", "minimal"

# Map selected feature set to data variables
feature_sets = {
    "full": (X_green, X_yellow),
    "fare_trip": (X_fare_trip_green, X_fare_trip_yellow),
    "payment_passenger": (X_payment_passenger_green, X_payment_passenger_yellow),
    "location": (X_location_green, X_location_yellow),
    "no_location": (X_no_location_green, X_no_location_yellow),
    "minimal": (X_minimal_green, X_minimal_yellow),
}

X_green, X_yellow = feature_sets[FEATURE_SET]

# %% Splitting of the data into train/test/validation (60:20:20) for both green and yellow data

# Splitting green dataset
X_train_green, X_test_green, y_train_green, y_test_green = train_test_split(X_green, y_green, test_size=0.2, random_state=0)
X_train_green, X_val_green, y_train_green, y_val_green = train_test_split(X_train_green, y_train_green, test_size=0.25, random_state=0)  # 25% of 80% = 20%

# Splitting yellow dataset
X_train_yellow, X_test_yellow, y_train_yellow, y_test_yellow = train_test_split(X_yellow, y_yellow, test_size=0.2, random_state=0)
X_train_yellow, X_val_yellow, y_train_yellow, y_val_yellow = train_test_split(X_train_yellow, y_train_yellow, test_size=0.25, random_state=0)

# Scale data using StandardScaler
scaler_green = StandardScaler().fit(X_train_green)
X_train_green = scaler_green.transform(X_train_green)
X_val_green = scaler_green.transform(X_val_green)
X_test_green = scaler_green.transform(X_test_green)

scaler_yellow = StandardScaler().fit(X_train_yellow)
X_train_yellow = scaler_yellow.transform(X_train_yellow)
X_val_yellow = scaler_yellow.transform(X_val_yellow)
X_test_yellow = scaler_yellow.transform(X_test_yellow)

# Save the used data 
# Save the used data
np.save("X_test_green.npy", X_test_green)
np.save("y_test_green.npy", y_test_green)
np.save("X_test_yellow.npy", X_test_yellow)
np.save("y_test_yellow.npy", y_test_yellow)
print("Test datasets saved successfully.")



#%% Training the Networks

# Debugging: Check dataset sizes before training
log_debug(f"Training started... Using feature set: {FEATURE_SET}")
log_debug(f"Dataset sizes: Green {X_train_green.shape}, Yellow {X_train_yellow.shape}")

# Ensure y_train is correctly shaped (batch_size, 1)
y_train_green = y_train_green.reshape(-1, 1)
y_train_yellow = y_train_yellow.reshape(-1, 1)

# Handle NaN or infinite values in training data
if np.isnan(X_train_green).any() or np.isnan(y_train_green).any():
    log_debug("Warning: NaN values detected in Green dataset!")

if np.isnan(X_train_yellow).any() or np.isnan(y_train_yellow).any():
    log_debug("Warning: NaN values detected in Yellow dataset!")

X_train_green = np.nan_to_num(X_train_green)
y_train_green = np.nan_to_num(y_train_green)

X_train_yellow = np.nan_to_num(X_train_yellow)
y_train_yellow = np.nan_to_num(y_train_yellow)

log_debug("Preprocessing complete. Starting training...")

learning_rate = 0.01
epochs = 500  # Define number of epochs

try:
    # Training the model for Green Taxi data
    log_debug("Training Green Taxi Model...")
    nn_green = SimpleNN(X_train_green.shape[1], hidden_size1=64, hidden_size2=32)

    # Dry-run test to check network stability
    try:
        y_pred_test = nn_green.forward(X_train_green[:5])
        nn_green.backward(X_train_green[:5], y_train_green[:5], learning_rate)
        log_debug("Dry-run test passed for Green model.")
    except Exception as e:
        log_debug(f"Error in Green model dry-run test: {str(e)}")
        raise

    for epoch in range(epochs):
        y_pred = nn_green.forward(X_train_green)
        loss = np.mean((y_pred - y_train_green) ** 2)  # MSE Loss
        nn_green.backward(X_train_green, y_train_green, learning_rate)

        # Print loss every 50 epochs
        if epoch % 50 == 0:
            log_debug(f"Epoch {epoch}/{epochs} - Green Loss: {loss:.6f}")

    log_debug("Green Taxi Model Training Complete!")

    # Training the model for Yellow Taxi data
    log_debug("Training Yellow Taxi Model...")
    nn_yellow = SimpleNN(X_train_yellow.shape[1], hidden_size1=64, hidden_size2=32)

    # Dry-run test for Yellow model
    try:
        y_pred_test = nn_yellow.forward(X_train_yellow[:5])
        nn_yellow.backward(X_train_yellow[:5], y_train_yellow[:5], learning_rate)
        log_debug("Dry-run test passed for Yellow model.")
    except Exception as e:
        log_debug(f"Error in Yellow model dry-run test: {str(e)}")
        raise

    for epoch in range(epochs):
        y_pred = nn_yellow.forward(X_train_yellow)
        loss = np.mean((y_pred - y_train_yellow) ** 2)  # MSE Loss
        nn_yellow.backward(X_train_yellow, y_train_yellow, learning_rate)

        # Print loss every 50 epochs
        if epoch % 50 == 0:
            log_debug(f"Epoch {epoch}/{epochs} - Yellow Loss: {loss:.6f}")

    log_debug("Yellow Taxi Model Training Complete!")

except Exception as e:
    log_debug(f"Error during training: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Save trained models
log_debug("Saving models...")
with open(f"nn_green_{FEATURE_SET}.pkl", "wb") as f:
    pickle.dump(nn_green, f)

with open(f"nn_yellow_{FEATURE_SET}.pkl", "wb") as f:
    pickle.dump(nn_yellow, f)

log_debug("Models saved successfully.")

# Evaluate the models on test data
log_debug("Evaluating Models...")
y_pred_green = nn_green.forward(X_test_green)
y_pred_yellow = nn_yellow.forward(X_test_yellow)

test_loss_green = np.mean((y_pred_green - y_test_green) ** 2)
test_loss_yellow = np.mean((y_pred_yellow - y_test_yellow) ** 2)

# Print test performance
log_debug(f"\nGreen Taxi Model - Test Loss (MSE): {test_loss_green:.6f}")
log_debug(f"Yellow Taxi Model - Test Loss (MSE): {test_loss_yellow:.6f}")
log_debug("Script completed successfully.")

