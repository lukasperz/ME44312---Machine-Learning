import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def view_data(main_folder_path, filename1, color1, filename2, color2):
    main_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Load data for the two taxi types
    color_data1 = pd.read_parquet(os.path.join(main_folder_path, filename1))
    color_data2 = pd.read_parquet(os.path.join(main_folder_path, filename2))

    # Add the taxi type information to each dataset
    color_data1["taxi_type"] = color1
    color_data2["taxi_type"] = color2

    # Combine both datasets
    data = pd.concat([color_data1, color_data2])

    # Keep only positive fare amounts & positive fare and tip amounts
    data = data[data["payment_type"].isin([1, 2])]
    data = data[(data["fare_amount"] > 0) & (data["tip_amount"] >= 0)]

    color_map = {color1: color1, "green": "green"}

    # Scatter plot setup
    scatter_plots = [
        ("fare_amount", "Tip Amount vs Fare Amount"),
        ("trip_distance", "Tip Amount vs Trip Distance"),
        ("passenger_count", "Tip Amount vs Passenger Count"),
        ("trip_distance", "Trip Distance vs Fare Amount")
    ]


    # Color 1 - Card Payment
    taxi_data11 = data[(data["taxi_type"] == color1) & (data["payment_type"] == 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = color1.capitalize() + " Taxi - Card Payment"
    fig.suptitle(title, fontsize=14)

    for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
        y_col = "tip_amount" if "Tip" in plot_title else "fare_amount"

        # Dynamic color based on the taxi color type (color1)
        ax.scatter(taxi_data11[x_col], taxi_data11[y_col], c=color1, marker="o", edgecolors="black", alpha=0.6)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_title(plot_title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for titles and labels
    plt.show()


    # Color 1 - Cash Payment
    taxi_data12 = data[(data["taxi_type"] == color1) & (data["payment_type"] == 2)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = color1.capitalize() + " Taxi - Cash Payment"
    fig.suptitle(title, fontsize=14)

    for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
        y_col = "tip_amount" if "Tip" in plot_title else "fare_amount"

        # Dynamic color based on the taxi color type (color1)
        ax.scatter(taxi_data12[x_col], taxi_data12[y_col], c=color1, marker="o", edgecolors="black", alpha=0.6)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_title(plot_title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for titles and labels
    plt.show()


    # Color 2 - Card Payment
    taxi_data21 = data[(data["taxi_type"] == color2) & (data["payment_type"] == 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = color1.capitalize() + " Taxi - Card Payment"
    fig.suptitle(title, fontsize=14)

    for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
        y_col = "tip_amount" if "Tip" in plot_title else "fare_amount"

        # Dynamic color based on the taxi color type (color1)
        ax.scatter(taxi_data21[x_col], taxi_data21[y_col], c=color2, marker="o", edgecolors="black", alpha=0.6)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_title(plot_title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for titles and labels
    plt.show()


    # Color 1 - Cash Payment
    taxi_data22 = data[(data["taxi_type"] == color2) & (data["payment_type"] == 2)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = color1.capitalize() + " Taxi - Cash Payment"
    fig.suptitle(title, fontsize=14)

    for ax, (x_col, plot_title) in zip(axes.flat, scatter_plots):
        y_col = "tip_amount" if "Tip" in plot_title else "fare_amount"

        # Dynamic color based on the taxi color type (color1)
        ax.scatter(taxi_data22[x_col], taxi_data22[y_col], c=color2, marker="o", edgecolors="black", alpha=0.6)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_title(plot_title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for titles and labels
    plt.show()




view_data(os.path.abspath(os.path.join(os.getcwd(), '..')), "yellow_taxi_data.parquet", "Yellow",  "green_taxi_data.parquet",  "Green")