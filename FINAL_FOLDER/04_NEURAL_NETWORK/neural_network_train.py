import numpy as np
import pandas as pd
import datetime as datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import os

model_dir = os.path.dirname(os.path.abspath(__file__))

# %% Read Data

data_green_train = pd.read_parquet('FINAL_FOLDER/00_DATA/green_taxi_data_train.parquet')
data_green_test = pd.read_parquet('FINAL_FOLDER/00_DATA/green_taxi_data_test.parquet')
data_yellow_train = pd.read_parquet('FINAL_FOLDER/00_DATA/yellow_taxi_data_train.parquet')
data_yellow_test = pd.read_parquet('FINAL_FOLDER/00_DATA/yellow_taxi_data_test.parquet')

# Display a preview of the training data
print(data_green_train.head())
print(data_yellow_train.head())

# Define target variable (tip amount) for both datasets using the training data
y_green = data_green_train['tip_amount'].values
y_yellow = data_yellow_train['tip_amount'].values

# %% FEATURE SETS Definition

# --- NEW: OLD FEATURE SET (everything from "full" without time-related columns) ---
X_green_old = data_green_train[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

X_yellow_old = data_yellow_train[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

# --- 1) FULL FEATURE SET WITH ADDITIONAL TIME COLUMNS ---
X_green = data_green_train[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount',
    # Added time-related columns:
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

X_yellow = data_yellow_train[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount',
    # Added time-related columns:
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

# Feature set excluding location (unchanged)
X_no_location_green = data_green_train[[
    'fare_amount', 'trip_distance', 'payment_type',
    'passenger_count', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

X_no_location_yellow = data_yellow_train[[
    'fare_amount', 'trip_distance', 'payment_type',
    'passenger_count', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

# Minimal feature set (unchanged)
X_minimal_green = data_green_train[['fare_amount', 'trip_distance']].values
X_minimal_yellow = data_yellow_train[['fare_amount', 'trip_distance']].values

# --- 3) NEW TIME-RELATED FEATURE SET ---
X_time_green = data_green_train[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

X_time_yellow = data_yellow_train[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

# Define all feature sets (order matters: "old" is the first to be trained)
feature_sets = {
    "old": (X_green_old, X_yellow_old),
    "full": (X_green, X_yellow),
    "no_location": (X_no_location_green, X_no_location_yellow),
    "minimal": (X_minimal_green, X_minimal_yellow),
    "time_features": (X_time_green, X_time_yellow)
}

# Define feature sets for test data using the same columns as the training data

# "old" test set (without time-related features)
X_green_test_old = data_green_test[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

X_yellow_test_old = data_yellow_test[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

# "full" test set with time-related columns
X_green_test_full = data_green_test[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount', 'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

X_yellow_test_full = data_yellow_test[[
    'fare_amount', 'trip_distance', 'payment_type', 'passenger_count',
    'PULocationID', 'DOLocationID', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount', 'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

X_no_location_green_test = data_green_test[[
    'fare_amount', 'trip_distance', 'payment_type',
    'passenger_count', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

X_no_location_yellow_test = data_yellow_test[[
    'fare_amount', 'trip_distance', 'payment_type',
    'passenger_count', 'RatecodeID', 'congestion_surcharge',
    'tolls_amount'
]].values

X_minimal_green_test = data_green_test[['fare_amount', 'trip_distance']].values
X_minimal_yellow_test = data_yellow_test[['fare_amount', 'trip_distance']].values

# "time_features" test set (only time-related columns)
X_time_green_test = data_green_test[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values
X_time_yellow_test = data_yellow_test[[
    'pickup_hour', 'dropoff_hour', 'pickup_dayofweek'
]].values

# Corresponding test data feature sets dictionary (with "old" first)
feature_sets_test = {
    "old": (X_green_test_old, X_yellow_test_old),
    "full": (X_green_test_full, X_yellow_test_full),
    "no_location": (X_no_location_green_test, X_no_location_yellow_test),
    "minimal": (X_minimal_green_test, X_minimal_yellow_test),
    "time_features": (X_time_green_test, X_time_yellow_test)
}

# Loop over every feature set and train models
for FEATURE_SET, (X_feat_green, X_feat_yellow) in feature_sets.items():

    # Use the provided train and test data feature sets
    X_train_green = np.nan_to_num(X_feat_green)
    y_train_green = np.nan_to_num(y_green)
    X_test_green = np.nan_to_num(feature_sets_test[FEATURE_SET][0])
    y_test_green = np.nan_to_num(data_green_test['tip_amount'].values)

    X_train_yellow = np.nan_to_num(X_feat_yellow)
    y_train_yellow = np.nan_to_num(y_yellow)
    X_test_yellow = np.nan_to_num(feature_sets_test[FEATURE_SET][1])
    y_test_yellow = np.nan_to_num(data_yellow_test['tip_amount'].values)

    # Reshape targets
    y_train_green = y_train_green.reshape(-1, 1)
    y_train_yellow = y_train_yellow.reshape(-1, 1)

    # Train the model for Green Taxi data
    print("Training Green Taxi Model...")

    nn_green = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        alpha=0.0001,
        early_stopping=True,
        max_iter=1000,
        random_state=0
    )
    nn_green.fit(X_train_green, y_train_green.ravel())
    y_pred_green = nn_green.predict(X_test_green)
    print("Green Taxi Model Training Complete!")

    # Train the model for Yellow Taxi data
    print("Training Yellow Taxi Model...")

    nn_yellow = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        alpha=0.0001,
        early_stopping=True,
        max_iter=1000,
        random_state=0
    )
    nn_yellow.fit(X_train_yellow, y_train_yellow.ravel())
    y_pred_yellow = nn_yellow.predict(X_test_yellow)
    print("Yellow Taxi Model Training Complete!")

    # Save models
    print("Saving models...")
    with open(os.path.join(model_dir, f"nn_green_{FEATURE_SET}.pkl"), "wb") as f:
        pickle.dump(nn_green, f)
    with open(os.path.join(model_dir, f"nn_yellow_{FEATURE_SET}.pkl"), "wb") as f:
        pickle.dump(nn_yellow, f)
    print("Models saved successfully.")

    # Evaluate the models on test data
    print("Evaluating Models...")
    y_pred_green_log = nn_green.predict(X_test_green)
    y_pred_green = np.expm1(y_pred_green_log)
    y_pred_yellow_log = nn_yellow.predict(X_test_yellow)
    y_pred_yellow = np.expm1(y_pred_yellow_log)

    test_loss_green = np.mean((y_pred_green - y_test_green) ** 2)
    test_loss_yellow = np.mean((y_pred_yellow - y_test_yellow) ** 2)

    print(f"\nGreen Taxi Model - Test Loss (MSE): {test_loss_green:.6f}")
    print(f"Yellow Taxi Model - Test Loss (MSE): {test_loss_yellow:.6f}")
    print(f"Completed training for feature set: {FEATURE_SET}\n")

print("Script completed successfully.")