import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the path to the main directory where the .parquet files are stored
main_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Load the data
yellow_data = pd.read_parquet(os.path.join(main_folder_path, "yellow_taxi_data.parquet"))
green_data = pd.read_parquet(os.path.join(main_folder_path, "green_taxi_data.parquet"))

# Add taxi type column
yellow_data["taxi_type"] = "yellow"
green_data["taxi_type"] = "green"

# Combine datasets
data = pd.concat([yellow_data, green_data])

# Keep only payment_type 1 (Card) and 2 (Cash)
data = data[data["payment_type"].isin([1, 2])]

# Keep only positive fare amounts
data = data[(data["fare_amount"] > 0) & (data["tip_amount"] >= 0)]

color_map = {"yellow": "gold", "green": "green"}
plot_titles = {
    ("yellow", 2): "Yellow Taxi - Cash", ("green", 2): "Green Taxi - Cash",
    ("yellow", 1): "Yellow Taxi - Card", ("green", 1): "Green Taxi - Card"}

# Generate subplots for each taxi type and payment method
for (taxi, payment), group in data.groupby(["taxi_type", "payment_type"]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(plot_titles[(taxi, payment)], fontsize=14)

    # Scatter plots
    scatter_plots = [
        ("fare_amount", "Tip Amount vs Fare Amount"),
        ("trip_distance", "Tip Amount vs Trip Distance"),
        ("passenger_count", "Tip Amount vs Passenger Count"),
        ("trip_distance", "Trip Distance vs Fare Amount")
    ]

    for ax, (x_col, title) in zip(axes.flat, scatter_plots):
        y_col = "tip_amount" if "Tip" in title else "fare_amount"
        ax.scatter(group[x_col], group[y_col], c=color_map[taxi], marker="o", edgecolors="black", alpha=0.6)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_title(title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()