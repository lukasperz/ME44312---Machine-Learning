import pandas as pd
import geopandas as gpd
import folium
import os
from shapely.lib import length

# === 1️⃣ Define the Main Folder Path ===
# Adjusted to explicitly include "FINAL_FOLDER"
main_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'FINAL_FOLDER'))
data_folder_path = os.path.join(main_folder_path, "00_DATA")

# === 2️⃣ Load the Parquet Files ===
green_taxi = pd.read_parquet(os.path.join(data_folder_path, "green_taxi_data_NO_SCALE.parquet"))
yellow_taxi = pd.read_parquet(os.path.join(data_folder_path, "yellow_taxi_data_NO_SCALE.parquet"))

# === 3️⃣ Load NYC Taxi Zone Shapefile ===
shapefile_path = os.path.join(main_folder_path, "02_VISUALIZATION", "Taxi Zones", "taxi_zones.shp")
taxi_zones = gpd.read_file(shapefile_path)

# ✅ Compute accurate centroids
taxi_zones = taxi_zones.to_crs(epsg=2263)  # Projected CRS for NYC
taxi_zones["centroid"] = taxi_zones.geometry.centroid
taxi_zones = taxi_zones.set_geometry("centroid").to_crs(epsg=4326)
taxi_zones["latitude"] = taxi_zones.geometry.y
taxi_zones["longitude"] = taxi_zones.geometry.x
taxi_zones = taxi_zones.drop(columns=["geometry", "centroid"])

# === 4️⃣ Merge Lat/Lon with Taxi Data ===
# Merge pickup locations
green_taxi = green_taxi.merge(taxi_zones, left_on="PULocationID", right_on="LocationID").rename(
    columns={"latitude": "latitude_pickup", "longitude": "longitude_pickup"})
yellow_taxi = yellow_taxi.merge(taxi_zones, left_on="PULocationID", right_on="LocationID").rename(
    columns={"latitude": "latitude_pickup", "longitude": "longitude_pickup"})

# Merge dropoff locations
green_taxi = green_taxi.merge(taxi_zones, left_on="DOLocationID", right_on="LocationID").rename(
    columns={"latitude": "latitude_dropoff", "longitude": "longitude_dropoff"})
yellow_taxi = yellow_taxi.merge(taxi_zones, left_on="DOLocationID", right_on="LocationID").rename(
    columns={"latitude": "latitude_dropoff", "longitude": "longitude_dropoff"})

# Drop missing values
green_taxi = green_taxi.dropna(subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])
yellow_taxi = yellow_taxi.dropna(
    subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])


# === 5️⃣ Create Function to Add Points to the Map ===
def add_points(df, lat_col, lon_col, map_obj, color):
    """
    Adds points to the map with a specified color.
    """
    coords = df[[lat_col, lon_col]].dropna().values.tolist()

    for coord in coords:
        folium.CircleMarker(
            location=coord,
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(map_obj)


# === 6️⃣ Sample a fixed number of rides & Ensure Matching Pairs ===
sample_size = min(5000, len(green_taxi), len(yellow_taxi))  # Choose a safe sample size
green_sample = green_taxi.sample(n=sample_size, random_state=42)
yellow_sample = yellow_taxi.sample(n=sample_size, random_state=42)


# === 7️⃣ Create Maps and Save as HTML Files ===
def create_and_save_map(df, task, color, lat_col, lon_col, file_name):
    """
    Creates a folium map for either pickups or dropoffs of a specific taxi color and saves the map.
    """
    # Create base map
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    # Add points for pickups or dropoffs
    add_points(df, lat_col, lon_col, nyc_map, color)

    # Save the map to an HTML file
    output_folder = os.path.join(main_folder_path, "02_VISUALIZATION", "Map Visualization")
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
    output_path = os.path.join(output_folder, f"nyc_taxi_map_{color}_{task}.html")

    nyc_map.save(output_path)
    print(f"\n✅ Map saved as '{output_path}'. Open it in your browser to view.")


# Add pickups and dropoffs for GREEN taxi
create_and_save_map(green_sample, "PU", "green", "latitude_pickup", "longitude_pickup", "green_PU")
create_and_save_map(green_sample, "DO", "green", "latitude_dropoff", "longitude_dropoff", "green_DO")

# Add pickups and dropoffs for YELLOW taxi
create_and_save_map(yellow_sample, "PU", "yellow", "latitude_pickup", "longitude_pickup", "yellow_PU")
create_and_save_map(yellow_sample, "DO", "yellow", "latitude_dropoff", "longitude_dropoff", "yellow_DO")
