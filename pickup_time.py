import pandas as pd
from datetime import datetime, timezone

# Read the Parquet file
df = pd.read_parquet('green_taxi_data') #adapt for yellow as well

# Check if the column "tpep_dropoff_datetime" exists
if 'tpep_dropoff_datetime' in df.columns:
    # Convert the "tpep_dropoff_datetime" column to UTC datetime
    # Handle potential NaN or invalid values
    df['dropoff_datetime'] = df['tpep_dropoff_datetime'].apply(
        lambda x: datetime.fromtimestamp(x / 1_000_000, timezone.utc) if isinstance(x, (int, float)) else x
    )

    # Print the resulting datetimes
    for dt in df['dropoff_datetime']:
        if pd.notnull(dt):  # Check to avoid printing NaN values
            print("Drop-off Date & Time (UTC):", dt)
        else:
            print("Invalid or missing timestamp")
else:
    print("Column 'tpep_dropoff_datetime' not found in the data.")
