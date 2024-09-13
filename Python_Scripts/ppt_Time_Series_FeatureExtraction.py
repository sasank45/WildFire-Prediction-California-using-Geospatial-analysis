#%%
import rasterio as rio
from glob import glob
import numpy as np
import os
import geowombat as gw
from xr_fresh.extractors import extract_features

#%%
# Define the range of years to process
years = [2023]

# Base directory for the ppt files
base_ppt_dir = "/home/ubuntu/WildFire_Capstone/Data/ppt"

#output directory for .tif files
output_ppt_dir = "/home/ubuntu/WildFire_Capstone/Output/ppt"

#%%
# Loop over each year
for year in years:
    print(f"Processing year: {year}")
    
    # Create year-specific directories if they don't exist
    year_dir = os.path.join(base_ppt_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    bil_files = glob(os.path.join(year_dir, "*.bil"))
    print(bil_files)
    
    # Process .bil files
    for bil in bil_files:
        with rio.open(bil) as src:
            profile = src.profile
            arr = src.read()
            arr = np.flipud(arr)  # Flip the array
            arr = np.squeeze(arr)  # Remove singleton dimensions
            print(profile)
            with rio.open(bil.replace(".bil", ".tif"), "w", **profile) as dst:
                dst.write(arr, 1)  # Write the array to the first band

    # Find all .tif files for the current year
    files = sorted(glob(os.path.join(year_dir, "*.tif")))

    # Feature list remains the same for all years
    feature_list = {
        "abs_energy": [{}],
        "maximum": [{}],
        "mean": [{}],
        "minimum": [{}],
        "mean_abs_change": [{}],
        "standard_deviation": [{}],
        "sum_values": [{}],
        # "linear_trend": [{"attr": "slope"}],  # Optionally included
        "median": [{}],
        "count_above_mean": [{}],
        "count_below_mean": [{}],
        "variance": [{}],
        "skewness": [{}],
    }

    outpath = os.path.join(output_ppt_dir, "Time_Series_Features", str(year))
    # Ensure output directory exists
    os.makedirs(outpath, exist_ok=True)

    # Use gw.config.update() as a context manager with the first .tif file of the current year as a reference
    if files:  # Check if there are any .tif files to process
        with gw.config.update(ref_image=files[0]):
            with gw.open(files, band_names=["ppt"]) as ds:
                print(ds)
                
                # Assuming 'time' or similar dimensions exist and need rechunking.
                # This example simply demonstrates the correct syntax; actual implementation may vary.
                ds_rechunked = ds.chunk({'time': -1})  # Adjust as needed for your dataset
                
                features = extract_features(
                    xr_data=ds_rechunked,
                    band="ppt",
                    feature_dict=feature_list,
                    na_rm=True,
                    persist=True,
                    filepath=outpath,
                    postfix=f"_{year}",
                )
    else:
        print(f"No .tif files found for year {year}. Skipping feature extraction.")

# %%
