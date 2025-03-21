import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MaxNLocator


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
        ("fare_amount", " "),
        ("trip_distance", " "),
    ]

    # Combined overlay plot for all taxi data (Yellow & Green, Cash & Card)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    title = " "
    fig.suptitle(title, fontsize=14)

    for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
        # Compute tip percentage for Yellow taxi data
        yellow_data = data[data["taxi_type"] == "Yellow"].copy()
        yellow_data["tip_percentage"] = 100.0 * yellow_data["tip_amount"] / (yellow_data["fare_amount"] + yellow_data["extra"] + yellow_data["mta_tax"] + yellow_data["tolls_amount"])

        # Compute tip percentage for Green taxi data
        green_data = data[data["taxi_type"] == "Green"].copy()
        green_data["tip_percentage"] = 100.0 * green_data["tip_amount"] / (green_data["fare_amount"] + green_data["extra"] + green_data["mta_tax"] + green_data["tolls_amount"])

        # Scatter plot using the selected x-axis and tip percentage as y-axis
        ax.scatter(
            yellow_data[x_col],
            yellow_data["tip_percentage"],
            color="yellow",
            alpha=0.5,
            s=20,
            marker='o',
            label="Yellow Taxi"
        )
        ax.scatter(
            green_data[x_col],
            green_data["tip_percentage"],
            color="green",
            alpha=0.5,
            s=20,
            marker='o',
            label="Green Taxi"
        )
        
        # Add legend
        ax.legend()

        # Set x-axis label depending on x_col
        if x_col == "fare_amount":
            ax.set_xlabel("Fare Amount [$]", fontsize=14)
        elif x_col == "trip_distance":
            ax.set_xlabel("Trip Distance [Miles]", fontsize=14)
        else:
            ax.set_xlabel(x_col.title(), fontsize=14)

        # Set y-axis label as Tip Percentage
        ax.set_ylabel("Tip Percentage [%]", fontsize=14)
        ax.set_title(plot_title, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_passenger_vs_relative_tip(main_folder_path, filename1, color1, filename2, color2):
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

    # Compute relative tip amount as tip_amount divided by the sum of fare, extra, mta_tax, and tolls_amount
    yellow_data = data[data["taxi_type"] == "Yellow"].copy()
    yellow_data["relative_tip"] = yellow_data["tip_amount"] / (yellow_data["fare_amount"] + yellow_data["extra"] + yellow_data["mta_tax"] + yellow_data["tolls_amount"])

    green_data = data[data["taxi_type"] == "Green"].copy()
    green_data["relative_tip"] = green_data["tip_amount"] / (green_data["fare_amount"] + green_data["extra"] + green_data["mta_tax"] + green_data["tolls_amount"])

    # Create the scatter plot for Passenger Count vs Relative Tip Amount
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        yellow_data["passenger_count"],
        yellow_data["tip_amount"],
        color="yellow",
        alpha=0.5,
        s=20,
        marker='o',
        label="Yellow Taxi"
    )
    ax.scatter(
        green_data["passenger_count"],
        green_data["tip_amount"],
        color="green",
        alpha=0.5,
        s=20,
        marker='o',
        label="Green Taxi"
    )

    # Add legend, labels, and title
    ax.legend()
    ax.set_xlabel("Passenger Count", fontsize=14)
    ax.set_ylabel("Tip Amount [$]", fontsize=14)
    ax.set_title(" ", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


# Define the main folder path as the parent directory of the script location
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
view_data(main_folder_path, "yellow_taxi_data_NO_SCALE.parquet", "Yellow", "green_taxi_data_NO_SCALE.parquet", "Green")
plot_passenger_vs_relative_tip(main_folder_path, "yellow_taxi_data_NO_SCALE.parquet", "Yellow", "green_taxi_data_NO_SCALE.parquet", "Green")