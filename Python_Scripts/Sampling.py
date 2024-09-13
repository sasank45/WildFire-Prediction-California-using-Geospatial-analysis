# %%
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import rasterio


#%%
# Function to generate random points within the bounds of a given GeoDataFrame
def generate_random_points_within_gdf(gdf, num_points):
    minx, miny, maxx, maxy = gdf.total_bounds
    count = 0
    points = []

    while count < num_points:
        point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if gdf.contains(point).any():
            points.append(point)
            count += 1
    return gpd.GeoDataFrame(geometry=points, crs=gdf.crs)


# Load the naturalearth_lowres dataset which contains a map of the USA
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
usa = world[world.name == "United States of America"]

# Generate 100,000 random points within the USA boundary
num_samples = 100000
sample_points_gdf = generate_random_points_within_gdf(usa, num_samples)

# Load the raster to get its CRS
raster_path = "/home/ubuntu/WildFire_Capstone/Output/ppt/Time_Series_Features/ppt__abs_energy__2013.tif"
with rasterio.open(raster_path) as raster:
    raster_crs = raster.crs

# Reproject the sample points to match the raster's CRS
sample_points_gdf = sample_points_gdf.to_crs(raster_crs)
sample_points_gdf.to_file("sampling_points.geojson", driver="GeoJSON")
# %%

sample_points_gdf.explore()


# %%
import geowombat as gw
import glob
import os

# Define the file paths for the directories
ppt_folder = "/home/ubuntu/WildFire_Capstone/Output/ppt/Time_Series_Features"
fire_perimeter_folder = "/home/ubuntu/WildFire_Capstone/Output/Fire_Perimeter"
tmax_folder = "/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features"

# Use glob to get lists of file paths
ppt_paths = glob.glob(os.path.join(ppt_folder, "ppt_*.tif"))
fire_perimeter_paths = glob.glob(
    os.path.join(fire_perimeter_folder, "rasterized_*.tif")
)
tmax_paths = glob.glob(
    os.path.join(tmax_folder, "tmax_*.tif")
)
print(ppt_paths)
ppt_names = [os.path.basename(ppt_path).split(".")[0] for ppt_path in ppt_paths]
fire_names = [os.path.basename(path).split(".")[0] for path in fire_perimeter_paths]
tmax_names = [os.path.basename(tmax_path).split(".")[0] for tmax_path in tmax_paths]

# Define the path for the reference image
ref_image_path = "/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter/example_1Km.tif"

# Define the path for your sample points, this should be a GeoJSON, shapefile, etc.
sample_points_path = "sampling_points.geojson"

# %%
# Process the rasters
#Empty list to store DataFrames
data_list = []

with gw.config.update(ref_image=ref_image_path):
    # Assuming you want to stack all ppt_paths and fire_perimeter_paths
    # and extract data for each raster
    with gw.open(
        list(ppt_paths + fire_perimeter_paths + tmax_paths),
        stack_dim="band",
        resample="nearest",
    ) as src:
        # Assuming src.gw.extract is a method and you have a list of sample points
        # Replace 'your_band' with the band name or index you wish to extract
        data = gw.extract(
            src, sample_points_path, band_names=list(ppt_names + fire_names + tmax_paths)
        )
        print(data)
        data_list.append(data)
# %%
# Now, assuming the variable 'data_list' is a list of DataFrames obtained from the previous operations
# You can concatenate them into a single DataFrame
import pandas as pd

data_df = pd.concat(data_list, ignore_index=True)

data_df.to_csv("/home/ubuntu/WildFire_Capstone/Output/extracted_Final_data.csv", index=False)

# %%
ddf = pd.read_csv("/home/ubuntu/WildFire_Capstone/Output/extracted_Final_data.csv")

ddf.head()
# %%

list(ddf.columns)
# %%
