import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import FastMarkerCluster
import os

# === 1️⃣ Load the Parquet Files ===
main_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
green_taxi = pd.read_parquet(os.path.join(main_folder_path, "green_taxi_data.parquet"))
yellow_taxi = pd.read_parquet(os.path.join(main_folder_path, "yellow_taxi_data.parquet"))
print(green_taxi)

# === 2️⃣ Load NYC Taxi Zone Shapefile ===
shapefile_path = os.path.join(main_folder_path, "01_Data_Cleaning_and_Visualization", "Taxi Zones", "taxi_zones.shp")
taxi_zones = gpd.read_file(shapefile_path)

# ✅ Step 1: Convert to WGS84 (Lat/Lon) before extracting coordinates
taxi_zones = taxi_zones.to_crs(epsg=4326)

# Compute centroids
taxi_zones["latitude"] = taxi_zones.geometry.centroid.y
taxi_zones["longitude"] = taxi_zones.geometry.centroid.x

# Debugging: Print first few converted coordinates
print("🔍 Step 1 - Sample Taxi Zone Coordinates (WGS84):")
print(taxi_zones[["LocationID", "latitude", "longitude"]].head())

# Drop the geometry column
taxi_zones = taxi_zones.drop(columns=["geometry"])

# === 3️⃣ Merge Lat/Lon with Taxi Data ===
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

# ✅ Step 2: Debugging - Print merged taxi data
print("\n🔍 Step 2 - Sample Green Taxi Data (Lat/Lon):")
print(green_taxi[["PULocationID", "latitude_pickup", "longitude_pickup"]].head())

print("\n🔍 Step 2 - Sample Yellow Taxi Data (Lat/Lon):")
print(yellow_taxi[["PULocationID", "latitude_pickup", "longitude_pickup"]].head())

# Drop missing values
green_taxi = green_taxi.dropna(subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])
yellow_taxi = yellow_taxi.dropna(
    subset=["latitude_pickup", "longitude_pickup", "latitude_dropoff", "longitude_dropoff"])

# === 4️⃣ Create NYC Base Map ===
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# ✅ Step 3: Add a test marker at Times Square to check if map renders
folium.Marker([40.7580, -73.9855], popup="Times Square", icon=folium.Icon(color="red")).add_to(nyc_map)

# === 5️⃣ Add Pickup and Dropoff Points to the Map ===

# Sample only 10,000 points for better performance
green_sample = green_taxi.sample(n=min(10000, len(green_taxi)), random_state=42)
yellow_sample = yellow_taxi.sample(n=min(10000, len(yellow_taxi)), random_state=42)


def add_clustered_points(df, lat_col, lon_col, map_obj, label):
    """
    Adds clustered points to the map using FastMarkerCluster.
    """
    coords = df[[lat_col, lon_col]].dropna().values.tolist()

    # ✅ Step 4: Debugging - Print first 5 coordinates
    print(f"\n🔍 Step 4 - Adding points for {label}:")
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

# === 6️⃣ Save and Show Map ===
output_path = os.path.join(main_folder_path, "01_Data_Cleaning_and_Visualization", "Map Visualization", "nyc_taxi_cluster_map.html")
nyc_map.save(output_path)
print(f"\n✅ Step 5 - Map saved as '{output_path}'. Open it in your browser to view.")