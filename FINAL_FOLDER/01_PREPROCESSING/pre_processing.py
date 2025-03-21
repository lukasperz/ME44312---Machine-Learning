import os
import pandas as pd
import pyarrow.parquet as pq

# Directory containing the parquet files
directory = "/Users/lukas/TU Delft (Macintosh HD)/ML_for_Transport_and_Multi_Machine_Systems/Assignment/Data"

# Output filenames for green & yellow taxi data
green_parquet_file = os.path.join(directory, "green_taxi_data.parquet")
yellow_parquet_file = os.path.join(directory, "yellow_taxi_data.parquet")

# 1. Build lists of all relevant Parquet files
green_files = []
yellow_files = []

for filename in os.listdir(directory):
    # Only process .parquet files
    if filename.endswith(".parquet"):
        # Filter by prefix to separate green and yellow taxi data
        if filename.startswith("green"):
            green_files.append(os.path.join(directory, filename))
        elif filename.startswith("yellow"):
            yellow_files.append(os.path.join(directory, filename))

# 2. Read & combine all green taxi files
green_dfs = []
for file_path in green_files:
    df = pq.read_table(file_path).to_pandas()
    # Filter rows that pass the conditions for valid taxi trips
    df_filtered = df[
        (df['trip_distance'] > 0) &
        (df['fare_amount'] > 0) &
        (df['passenger_count'] > 0) &
        (df['tip_amount'] >= 0)
    ]
    
    # Remove outliers from 'fare_amount' using the IQR method
    Q1 = df_filtered['fare_amount'].quantile(0.15)
    Q3 = df_filtered['fare_amount'].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df_filtered[
        (df_filtered['fare_amount'] >= lower_bound) &
        (df_filtered['fare_amount'] <= upper_bound)
    ]

    # Remove outliers from 'tip_amount' using the IQR method
    Q1_tip = df_filtered['tip_amount'].quantile(0.15)
    Q3_tip = df_filtered['tip_amount'].quantile(0.85)
    IQR_tip = Q3_tip - Q1_tip
    lower_bound_tip = Q1_tip - 1.5 * IQR_tip
    upper_bound_tip = Q3_tip + 1.5 * IQR_tip
    df_filtered = df_filtered[
        (df_filtered['tip_amount'] >= lower_bound_tip) &
        (df_filtered['tip_amount'] <= upper_bound_tip)
    ]

    # Sample 7500 rows if available
    if len(df_filtered) > 7500:
        df_filtered = df_filtered.sample(n=7500, random_state=42)

    # -- NEW: Extract datetime features --
    if 'tpep_pickup_datetime' in df_filtered.columns and 'tpep_dropoff_datetime' in df_filtered.columns:
        df_filtered['tpep_pickup_datetime'] = pd.to_datetime(df_filtered['tpep_pickup_datetime'], errors='coerce')
        df_filtered['tpep_dropoff_datetime'] = pd.to_datetime(df_filtered['tpep_dropoff_datetime'], errors='coerce')
        
        # Extract numeric features
        df_filtered['pickup_hour'] = df_filtered['tpep_pickup_datetime'].dt.hour
        df_filtered['pickup_dayofweek'] = df_filtered['tpep_pickup_datetime'].dt.dayofweek
        
        df_filtered['dropoff_hour'] = df_filtered['tpep_dropoff_datetime'].dt.hour
        df_filtered['dropoff_dayofweek'] = df_filtered['tpep_dropoff_datetime'].dt.dayofweek
    
    green_dfs.append(df_filtered)

if green_dfs:
    # Combine all green taxi DataFrames
    combined_green = pd.concat(green_dfs, ignore_index=True)
    combined_green.to_parquet(green_parquet_file, engine="pyarrow")
    print(f"Combined {len(green_files)} green files into {green_parquet_file}")
    
    # Preprocessing for tip prediction on green taxi data
    df = combined_green.copy()  # Work on a copy
    df_filtered = df  # Updated line to remove second filtering step
    
    # Label encoding for categorical features
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    categorical_features = ['payment_type', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID']
    LE = LabelEncoder()
    for col in categorical_features:
        if col in df_filtered.columns:
            df_filtered[col] = LE.fit_transform(df_filtered[col])
    
    # Check if df_filtered is empty
    if df_filtered.empty:
        print("No valid rows for green taxi data after filtering. Skipping scaling and train/test split.")
    else:
        # -- NEW: Include our new datetime-derived features in numerical_features
        numerical_features = [
            'fare_amount', 'trip_distance', 'passenger_count',
            'pickup_hour', 'pickup_dayofweek', 'dropoff_hour', 'dropoff_dayofweek',
        ]
        
        # Scale numerical features
        scaler = StandardScaler()
        # Only scale columns that actually exist
        for col in numerical_features:
            if col not in df_filtered.columns:
                df_filtered[col] = 0  # or drop it, but here we just set to 0 if missing
        
        df_filtered[numerical_features] = scaler.fit_transform(df_filtered[numerical_features])
        
        # Split the data into train (80%) and test (20%)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
        
        # Save train and test sets
        base_dir = os.path.dirname(os.path.abspath(__file__))
        green_train_file = os.path.join(base_dir, "green_taxi_data_train.parquet")
        green_test_file = os.path.join(base_dir, "green_taxi_data_test.parquet")
        
        train_df.to_parquet(green_train_file, engine="pyarrow")
        test_df.to_parquet(green_test_file, engine="pyarrow")
        
        print(f"Green taxi data split into train and test sets saved to {green_train_file} and {green_test_file}")
else:
    print("No green Parquet files found.")

# 3. Read & combine all yellow taxi files
yellow_dfs = []
for file_path in yellow_files:
    df = pq.read_table(file_path).to_pandas()
    # Filter rows that pass the conditions for valid taxi trips
    df_filtered = df[
        (df['trip_distance'] > 0) &
        (df['fare_amount'] > 0) &
        (df['passenger_count'] > 0) &
        (df['tip_amount'] >= 0)
    ]
    
    # Remove outliers from 'fare_amount'
    Q1 = df_filtered['fare_amount'].quantile(0.15)
    Q3 = df_filtered['fare_amount'].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df_filtered[
        (df_filtered['fare_amount'] >= lower_bound) &
        (df_filtered['fare_amount'] <= upper_bound)
    ]

    # Remove outliers from 'tip_amount'
    Q1_tip = df_filtered['tip_amount'].quantile(0.15)
    Q3_tip = df_filtered['tip_amount'].quantile(0.85)
    IQR_tip = Q3_tip - Q1_tip
    lower_bound_tip = Q1_tip - 1.5 * IQR_tip
    upper_bound_tip = Q3_tip + 1.5 * IQR_tip
    df_filtered = df_filtered[
        (df_filtered['tip_amount'] >= lower_bound_tip) &
        (df_filtered['tip_amount'] <= upper_bound_tip)
    ]

    # Sample 7500 rows if available
    if len(df_filtered) > 7500:
        df_filtered = df_filtered.sample(n=7500, random_state=42)
    
    # -- NEW: Extract datetime features --
    if 'tpep_pickup_datetime' in df_filtered.columns and 'tpep_dropoff_datetime' in df_filtered.columns:
        df_filtered['tpep_pickup_datetime'] = pd.to_datetime(df_filtered['tpep_pickup_datetime'], errors='coerce')
        df_filtered['tpep_dropoff_datetime'] = pd.to_datetime(df_filtered['tpep_dropoff_datetime'], errors='coerce')
        
        # Extract numeric features
        df_filtered['pickup_hour'] = df_filtered['tpep_pickup_datetime'].dt.hour
        df_filtered['pickup_dayofweek'] = df_filtered['tpep_pickup_datetime'].dt.dayofweek
        
        df_filtered['dropoff_hour'] = df_filtered['tpep_dropoff_datetime'].dt.hour
        df_filtered['dropoff_dayofweek'] = df_filtered['tpep_dropoff_datetime'].dt.dayofweek

    yellow_dfs.append(df_filtered)

if yellow_dfs:
    # Combine all yellow taxi DataFrames
    combined_yellow = pd.concat(yellow_dfs, ignore_index=True)
    combined_yellow.to_parquet(yellow_parquet_file, engine="pyarrow")
    print(f"Combined {len(yellow_files)} yellow files into {yellow_parquet_file}")
    
    # Preprocessing for tip prediction on yellow taxi data
    df = combined_yellow.copy()
    df_filtered = df  # Updated line to remove second filtering step
    
    # Label encoding for categorical features
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    categorical_features = ['payment_type', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID']
    LE = LabelEncoder()
    for col in categorical_features:
        if col in df_filtered.columns:
            df_filtered[col] = LE.fit_transform(df_filtered[col])
    
    # Check if df_filtered is empty
    if df_filtered.empty:
        print("No valid rows for yellow taxi data after filtering. Skipping scaling and train/test split.")
    else:
        # -- NEW: Include our new datetime-derived features in numerical_features
        numerical_features = [
            'fare_amount', 'trip_distance', 'passenger_count',
            'pickup_hour', 'pickup_dayofweek', 'dropoff_hour', 'dropoff_dayofweek'
        ]
        
        scaler = StandardScaler()
        for col in numerical_features:
            if col not in df_filtered.columns:
                df_filtered[col] = 0
        
        df_filtered[numerical_features] = scaler.fit_transform(df_filtered[numerical_features])
        
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yellow_train_file = os.path.join(base_dir, "yellow_taxi_data_train.parquet")
        yellow_test_file = os.path.join(base_dir, "yellow_taxi_data_test.parquet")
        
        train_df.to_parquet(yellow_train_file, engine="pyarrow")
        test_df.to_parquet(yellow_test_file, engine="pyarrow")
        
        print(f"Yellow taxi data split into train and test sets saved to {yellow_train_file} and {yellow_test_file}")
else:
    print("No yellow Parquet files found.")