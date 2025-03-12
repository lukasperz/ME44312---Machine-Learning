import os
import pandas as pd
import pyarrow.parquet as pq

# ---------------- READ ME ----------------
# This code reads the parquet files in the folder and filters 
# null or physically unmeaningfull values
# and then reduces outliers by applying a 3-sigma reduction.
# The results are then saved as .CSV files

# Directory containing the parquet files
directory = "/Users/lukas/TU Delft (Macintosh HD)/ML_for_Transport_and_Multi_Machine_Systems/Assignment/Data"

# Output filenames for green taxi data
green_parquet_file = os.path.join(directory, "green_taxi_data.parquet")

# Output filenames for yellow taxi data
yellow_parquet_file = os.path.join(directory, "yellow_taxi_data.parquet")

# Lists to store sampled data for each category
sampled_green = []
sampled_yellow = []

def three_sigma_filter(df, columns):
    """
    Removes rows in `df` where values in each of the specified `columns`
    lie outside the mean ± 3*std for that column.
    """
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        upper_bound = mean_val + 3 * std_val
        lower_bound = mean_val - 3 * std_val
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Process only .parquet files for green or yellow taxis
    if (filename.startswith("green") or filename.startswith("yellow")) and filename.endswith(".parquet"):
        file_path = os.path.join(directory, filename)
        
        # Read the parquet file
        df = pq.read_table(file_path).to_pandas()

        # 1) Basic filtering
        df_filtered = df[
            (df['trip_distance'] > 0) &
            (df['fare_amount'] > 0) &
            (df['passenger_count'] > 0) &
            (df['tip_amount'] >= 0)
        ]

        # 2) 3-sigma outlier removal for specified columns
        columns_to_filter = ['trip_distance', 'fare_amount', 'passenger_count', 'tip_amount']
        df_filtered = three_sigma_filter(df_filtered, columns_to_filter)

        # 3) Sample 10,000 random rows (or fewer if the file has fewer rows)
        if len(df_filtered) > 0:
            sampled_df = df_filtered.sample(n=min(8000, len(df_filtered)), random_state=42)
        else:
            continue  # Skip if nothing left after filtering

        # 4) Append to the corresponding list based on file type (green or yellow)
        if filename.startswith("green"):
            sampled_green.append(sampled_df)
        else:  # filename.startswith("yellow")
            sampled_yellow.append(sampled_df)

# --- Concatenate and save each subset as Parquet if not empty ---
def save_if_not_empty(df_list, parquet_path):
    """Concatenate, then save the dataframe as a Parquet file."""
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_parquet(parquet_path, engine='pyarrow')
        print(f"Saved {len(final_df)} rows to {parquet_path}")
    else:
        print(f"No data to save for {parquet_path}")

save_if_not_empty(sampled_green, green_parquet_file)
save_if_not_empty(sampled_yellow, yellow_parquet_file)