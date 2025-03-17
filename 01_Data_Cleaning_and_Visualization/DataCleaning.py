import pandas as pd
import os

# Load the data


# Define the path to the main directory where the .parquet files are stored
main_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Load the data
file_path =  pd.read_parquet(os.path.join(main_folder_path, "sampled_yellow_data.parquet"))
df = pd.read_parquet(file_path)

# Remove duplicate rows
df = df.drop_duplicates()

# Drop columns with too many missing values (more than 50%)
missing_limit = len(df) * 0.5
df = df.dropna(thresh=missing_limit, axis=1)

# Fill missing values with median (for numbers) or mode (for text)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Standardize text columns (trim spaces, lowercase)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()

# Convert date columns (if any) to proper datetime format
date_cols = [col for col in df.columns if 'date' in col or 'time' in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Save cleaned data
cleaned_file_path = "/mnt/data/cleaned_green_data.parquet"
df.to_parquet(cleaned_file_path, index=False)

print("Cleaning complete. Saved cleaned data to:", cleaned_file_path)