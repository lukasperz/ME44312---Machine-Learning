import os
import pandas as pd
import pyarrow.parquet as pq

# Directory containing the parquet files
directory = "/Users/lukas/TU Delft (Macintosh HD)/ML_for_Transport_and_Multi_Machine_Systems/Assignment/Data"  # Change this to your actual directory

# Output file
output_file = os.path.join(directory, "sampled_yellow_data.parquet")

# List to store sampled data
sampled_data = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith("yellow") and filename.endswith(".parquet"):
        file_path = os.path.join(directory, filename)

        # Read the parquet file
        df = pq.read_table(file_path).to_pandas()

        # Sample 10,000 random rows (or less if the file has fewer rows)
        sampled_df = df.sample(n=min(10000, len(df)), random_state=42)

        # Append to the list
        sampled_data.append(sampled_df)

# Concatenate all sampled data
if sampled_data:
    final_df = pd.concat(sampled_data, ignore_index=True)

    # Save the combined data to a single parquet file
    final_df.to_parquet(output_file, engine='pyarrow')
    print(f"Saved {len(final_df)} rows to {output_file}")
else:
    print("No files found or no data sampled.")