import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
import os

# === 1️⃣ Set Folder Paths ===
final_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Move up to FINAL_FOLDER
data_folder_path = os.path.join(final_folder_path, "00_DATA")  # Updated location for parquet files
shapefile_folder_path = os.path.join(final_folder_path, "02_VISUALIZATION", "Taxi Zones")  # Updated path
output_folder = os.path.join(final_folder_path, "02_VISUALIZATION", "Map Visualization")  # Ensure correct output location

# === 2️⃣ Load the Parquet Files ===
green_taxi = pd.read_parquet(os.path.join(data_folder_path, "green_taxi_data_NO_SCALE.parquet"))
yellow_taxi = pd.read_parquet(os.path.join(data_folder_path, "yellow_taxi_data_NO_SCALE.parquet"))
print(green_taxi)

# === 3️⃣ Load NYC Taxi Zone Shapefile ===
shapefile_path = os.path.join(shapefile_folder_path, "taxi_zones.shp")
taxi_zones = gpd.read_file(shapefile_path)

# ✅ Convert to WGS84 (Lat/Lon)
taxi_zones = taxi_zones.to_crs(epsg=4326)

# Compute centroids
taxi_zones["latitude"] = taxi_zones.geometry.centroid.y
taxi_zones["longitude"] = taxi_zones.geometry.centroid.x

# Debugging: Print first few converted coordinates
print("🔍 Sample Taxi Zone Coordinates (WGS84):")
print(taxi_zones[["LocationID", "latitude", "longitude"]].head())

# Drop the geometry column
taxi_zones = taxi_zones.drop(columns=["geometry"])

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

# ✅ Debugging - Print merged taxi data
print("\n🔍 Sample Green Taxi Data (Lat/Lon):")
print(green_taxi[["PULocationID", "latitude_pickup", "longitude_pickup"]].head())

print("\n🔍 Sample Yellow Taxi Data (Lat/Lon):")
print(yellow_taxi[["PULocationID", "latitude_pickup", "longitude_pickup"]].head())

# Drop missing values
green_taxi = green_taxi.dropna(subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])
yellow_taxi = yellow_taxi.dropna(subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])

# === 5️⃣ Create NYC Base Map ===
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# ✅ Add a test marker at Times Square
folium.Marker([40.7580, -73.9855], popup="Times Square", icon=folium.Icon(color="red")).add_to(nyc_map)

# === 6️⃣ Add Pickup and Dropoff Points to the Map ===

# Sample only 10,000 points for better performance
green_sample = green_taxi.sample(n=min(10000, len(green_taxi)), random_state=42)
yellow_sample = yellow_taxi.sample(n=min(10000, len(yellow_taxi)), random_state=42)


def add_clustered_points(df, lat_col, lon_col, map_obj, label):
    """
    Adds clustered points to the map using FastMarkerCluster.
    """
    coords = df[[lat_col, lon_col]].dropna().values.tolist()

    # ✅ Debugging - Print first 5 coordinates
    print(f"\n🔍 Adding points for {label}:")
    print(coords[:5])  # Print first 5 coordinate pairs

    # Add points to the map
    FastMarkerCluster(data=coords).add_to(map_obj)
    print(f"✅ Added {len(coords)} points for {label}.")


# Add GREEN taxi pickups and dropoffs with clustering
add_clustered_points(green_sample, "latitude_pickup", "longitude_pickup", nyc_map, "Green Pickup")
add_clustered_points(green_sample, "latitude_dropoff", "longitude_dropoff", nyc_map, "Green Dropoff")

# Add YELLOW taxi pickups and dropoffs with clustering
add_clustered_points(yellow_sample, "latitude_pickup", "longitude_pickup", nyc_map, "Yellow Pickup")
add_clustered_points(yellow_sample, "latitude_dropoff", "longitude_dropoff", nyc_map, "Yellow Dropoff")

# === 7️⃣ Save and Show Map ===
os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
output_path = os.path.join(output_folder, "nyc_taxi_cluster_map.html")

nyc_map.save(output_path)
print(f"\n✅ Map saved as '{output_path}'. Open it in your browser to view.")
