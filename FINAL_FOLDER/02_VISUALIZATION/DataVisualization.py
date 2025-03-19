import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def view_data(main_folder_path, filename1, color1, filename2, color2):
    data_folder_path = os.path.join(main_folder_path, "00_DATA")

    # Load data for the two taxi types
    color_data1 = pd.read_parquet(os.path.join(data_folder_path, filename1))
    color_data2 = pd.read_parquet(os.path.join(data_folder_path, filename2))

    # Add the taxi type information to each dataset
    color_data1["taxi_type"] = color1
    color_data2["taxi_type"] = color2

    # Combine both datasets
    data = pd.concat([color_data1, color_data2])

    # Keep only positive fare amounts & positive fare and tip amounts
    data = data[data["payment_type"].isin([1, 2])]
    data = data[(data["fare_amount"] > 0) & (data["tip_amount"] >= 0)]

    scatter_plots = [
        ("fare_amount", "Tip Amount vs Fare Amount"),
        ("trip_distance", "Tip Amount vs Trip Distance"),
        ("passenger_count", "Tip Amount vs Passenger Count"),
        ("trip_distance", "Trip Distance vs Fare Amount")
    ]

    for taxi_type, color in [(color1, color1), (color2, color2)]:
        for payment_type, payment_label in [(1, "Card Payment"), (2, "Cash Payment")]:
            taxi_data = data[(data["taxi_type"] == taxi_type) & (data["payment_type"] == payment_type)]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            title = f"{taxi_type.capitalize()} Taxi - {payment_label}"
            fig.suptitle(title, fontsize=14)

            for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
                y_col = "tip_amount" if "Tip" in plot_title else "fare_amount"
                ax.scatter(taxi_data[x_col], taxi_data[y_col], c=color, marker="o", edgecolors="black", alpha=0.6)
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(y_col.replace("_", " ").title())
                ax.set_title(plot_title)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for titles and labels
            plt.show()


# Define the main folder path as the parent directory of the script location
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
view_data(main_folder_path, "yellow_taxi_data_NO_SCALE.parquet", "Yellow", "green_taxi_data_NO_SCALE.parquet", "Green")