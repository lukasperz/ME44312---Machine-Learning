import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prepare_data(main_folder_path, filename1, color1, filename2, color2):
    """Load data, filter necessary columns, and prepare for visualization."""
    data_folder_path = os.path.join(main_folder_path, "00_DATA")
    color_data1 = pd.read_parquet(os.path.join(data_folder_path, filename1))
    color_data2 = pd.read_parquet(os.path.join(data_folder_path, filename2))

    color_data1["taxi_type"] = color1
    color_data2["taxi_type"] = color2

    data = pd.concat([color_data1, color_data2])

    # Keep only valid fare and tip amounts
    data = data[data["payment_type"].isin([1, 2])]
    data = data[(data["fare_amount"] > 0) & (data["tip_amount"] >= 0)]
    
    # Rename taxi types to full names
    data["taxi_type"] = data["taxi_type"].replace({
         "Yellow": "Yellow Taxi",
         "Green": "Green Taxi"
    })

    return data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(data, x_col="fare_amount"):
    x_threshold = data[x_col].quantile(1)
    y_threshold = data["tip_amount"].quantile(1)
    filtered_data = data[(data[x_col] <= x_threshold) & (data["tip_amount"] <= y_threshold)]
    
    sns.set_style("whitegrid")
    sns.set_context("paper")  
    
    plt.figure(figsize=(5, 4))  
    ax = sns.scatterplot(
        data=filtered_data,
        x=x_col,
        y="tip_amount",
        hue="taxi_type",
        hue_order=["Yellow Taxi", "Green Taxi"],  
        palette={"Yellow Taxi": "gold", "Green Taxi": "green"},
        alpha=0.6,
        edgecolor='none'
    )
    ax.legend(title="")

    plt.title("")
    plt.xticks([1, 2], ["Card", "Cash"])
    plt.xlim(0.5, 2.5)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel("Tip Amount [$]")
    plt.tight_layout()
    sns.despine()
    plt.show()

def main():
    main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data = load_and_prepare_data(
        main_folder_path, 
        "yellow_taxi_data_NO_SCALE.parquet", "Yellow",
        "green_taxi_data_NO_SCALE.parquet", "Green"
    )
    # Choose the column to plot against tip_amount. Change "fare_amount" to any other column as desired.
    selected_column = "payment_type"
    plot_scatter(data, x_col="payment_type")

if __name__ == "__main__":
    main()