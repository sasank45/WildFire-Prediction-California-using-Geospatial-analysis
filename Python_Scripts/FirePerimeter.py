#%%
#Importing Libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import re
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from matplotlib import pyplot as plt
from rasterio.plot import show

#%%
# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)


# Load your GeoJSON file
gdf = gpd.read_file('/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter/WFIGS_Interagency_Fire_Perimeters.geojson')

 #%%
#Testing
#print(gdf.head())  # Print the first few rows of the DataFrame
#print(gdf.columns)
#list(gdf.columns)
#print(gdf['attr_FireOutDateTime'].isna().sum())
#print(gdf['attr_FireDiscoveryDateTime'].isna().sum())

#%%

#Output Directory
output_directory = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter'


# Assuming gdf is your GeoDataFrame and 'attr_FireDiscoveryDateTime' is the column
# Convert 'attr_FireDiscoveryDateTime' to datetime format if it's not already
gdf['attr_FireDiscoveryDateTime'] = pd.to_datetime(gdf['attr_FireDiscoveryDateTime'])


# %%
# Extract the year from the datetime column
gdf['year'] = gdf['attr_FireDiscoveryDateTime'].dt.year


# Get a list of unique years
unique_years = gdf['year'].unique()

# Filter and save GeoJSON for each year
for year in unique_years:
    # Skip if year is NaN
    if pd.isna(year):
        continue
    
    # Filter the GeoDataFrame for the current year
    gdf_year = gdf[gdf['year'] == year]
    
    # Define the output file name based on the year
    output_filename = os.path.join(output_directory, f'fire_perimeter_{int(year)}.geojson')
    
    # Save the subset to a GeoJSON file
    gdf_year.to_file(output_filename, driver='GeoJSON')

#%%
#Raterize Function
def rasterize_geojson(geojson_path, raster_template_path, output_raster_path):
    vector = gpd.read_file(geojson_path)
    geom = [shapes for shapes in vector.geometry]
    raster = rasterio.open(raster_template_path)

    vector['id'] = range(len(vector))
    geom = [shapes for shapes in vector.geometry]
    rasterized = features.rasterize(geom,
                                out_shape = raster.shape,
                                fill = 0,  # value where there are no fires
                                out = None,
                                transform = raster.transform,
                                all_touched = True,
                                default_value = 1,  # value assigned when there is a fire
                                dtype = np.int16)
    
    with rasterio.open(
            output_raster_path, "w",
            driver="GTiff",
            crs=raster.crs,
            transform=raster.transform,
            dtype=rasterio.uint8,
            count=1,
            width=raster.width,
            height=raster.height) as dst:
        dst.write(rasterized, indexes=1)

# Define your directories
geojson_dir = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter'
raster_template_path = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter/example_1Km.tif'
output_dir = '/home/ubuntu/WildFire_Capstone/Output/Fire_Perimeter'
os.makedirs(output_dir, exist_ok=True)

# Regular expression pattern to match filenames ending with _{year}.geojson
pattern = re.compile(r'.*_(\d{4})\.geojson$')

for geojson_file in os.listdir(geojson_dir):
    if pattern.match(geojson_file):
        year = pattern.search(geojson_file).group(1)
        geojson_path = os.path.join(geojson_dir, geojson_file)
        output_raster_path = os.path.join(output_dir, f"rasterized_{year}.tif")
        
        rasterize_geojson(geojson_path, raster_template_path, output_raster_path)





#%%
#Backup
""" def rasterize_geojson(geojson_path, raster_template_path, output_raster_path):
    vector = gpd.read_file(geojson_path)
    geom = [shapes for shapes in vector.geometry]
    raster = rasterio.open(raster_template_path)

    vector['id'] = range(len(vector))
    geom_value = ((geom, value) for geom, value in zip(vector.geometry, vector['id']))

    rasterized = features.rasterize(geom_value,
                                    out_shape=raster.shape,
                                    transform=raster.transform,
                                    all_touched=True,
                                    fill=-5,  # background value
                                    merge_alg=MergeAlg.replace,
                                    dtype=np.int16)
    
    with rasterio.open(
            output_raster_path, "w",
            driver="GTiff",
            crs=raster.crs,
            transform=raster.transform,
            dtype=rasterio.uint8,
            count=1,
            width=raster.width,
            height=raster.height) as dst:
        dst.write(rasterized, indexes=1)

# Define your directories
geojson_dir = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter'
raster_template_path = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter/example_1Km.tif'
output_dir = '/home/ubuntu/WildFire_Capstone/Output/Fire_Perimeter'
os.makedirs(output_dir, exist_ok=True)

# Regular expression pattern to match filenames ending with _{year}.geojson
pattern = re.compile(r'.*_(\d{4})\.geojson$')

for geojson_file in os.listdir(geojson_dir):
    if pattern.match(geojson_file):
        year = pattern.search(geojson_file).group(1)
        geojson_path = os.path.join(geojson_dir, geojson_file)
        output_raster_path = os.path.join(output_dir, f"rasterized_{year}.tif")
        
        rasterize_geojson(geojson_path, raster_template_path, output_raster_path) """



#%%
# Directory directory containing your GeoJSON files
directory_path = '/home/ubuntu/WildFire_Capstone/Data/Fire_Perimeter/'

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    # Check if it's a GeoJSON file
    if os.path.isfile(file_path) and filename.endswith('.geojson'):
        # Load the GeoJSON file into a GeoDataFrame
        gdf = gpd.read_file(file_path)
        # Count the number of features (records) in the GeoDataFrame
        num_records = len(gdf)
        print(f'{filename} has {num_records} records')

# %%
