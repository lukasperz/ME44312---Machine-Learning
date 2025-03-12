import os
import pandas as pd
import pyarrow.parquet as pq

# Directory containing the parquet files
directory = "/Users/lukas/TU Delft (Macintosh HD)/ML_for_Transport_and_Multi_Machine_Systems/Assignment/Data"

# Output filenames
training_green_file = os.path.join(directory, "training_green_taxi_data.parquet")
training_yellow_file = os.path.join(directory, "training_yellow_taxi_data.parquet")
validation_green_file = os.path.join(directory, "validation_green_taxi_data.parquet")
validation_yellow_file = os.path.join(directory, "validation_yellow_taxi_data.parquet")

# Lists to store sampled data for each category
sampled_training_green = []
sampled_training_yellow = []
sampled_validation_green = []
sampled_validation_yellow = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # We only want .parquet files for green or yellow
    if (filename.startswith("green") or filename.startswith("yellow")) and filename.endswith(".parquet"):
        file_path = os.path.join(directory, filename)


        # --- Extract the year from the filename ---
        # Typical format: "green_tripdata_2018-01.parquet"
        # We can split on "_" and then split on "-"
        # Example: ["green", "tripdata", "2018-01.parquet"] -> "2018"
        year_str = filename.split('_')[2].split('-')[0]
        year = int(year_str)

        # Read the parquet file
        df = pq.read_table(file_path).to_pandas()

        # Filter rows based on the conditions:
        # Trip distance > 0, Fare amount > 0, Passenger count > 0, Tip amount >= 0
        df_filtered = df[
            (df['trip_distance'] > 0) &
            (df['fare_amount'] > 0) &
            (df['passenger_count'] > 0) &
            (df['tip_amount'] >= 0)
        ]

        # Sample 10,000 random rows (or fewer if the file has fewer rows)
        if len(df_filtered) > 0:
            sampled_df = df_filtered.sample(n=min(16500, len(df_filtered)), random_state=42)
        else:
            # Skip empty dataframes
            continue

        # --- Decide which list to append to based on taxi color & year ---
        if filename.startswith("green"):
            if year >= 2020:
                sampled_training_green.append(sampled_df)
            else:
                # year <= 2018 goes to validation
                sampled_validation_green.append(sampled_df)
        elif filename.startswith("yellow"):
            if year >= 2020:
                sampled_training_yellow.append(sampled_df)
            else:
                # year <= 2018 goes to validation
                sampled_validation_yellow.append(sampled_df)

# --- Concatenate and save each subset if not empty ---
def save_if_not_empty(df_list, output_path):
    """Helper to concatenate and save if there's data."""
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_parquet(output_path, engine='pyarrow')
        print(f"Saved {len(final_df)} rows to {output_path}")
    else:
        print(f"No data to save for {output_path}")

save_if_not_empty(sampled_training_green, training_green_file)
save_if_not_empty(sampled_training_yellow, training_yellow_file)
save_if_not_empty(sampled_validation_green, validation_green_file)
save_if_not_empty(sampled_validation_yellow, validation_yellow_file)
