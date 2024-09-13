#!/bin/bash

# Base URL for data
BASE_URL="https://ftp.prism.oregonstate.edu/monthly/"

# Base directory for storing ppt and tmax data
BASE_DATA_DIR="/home/ubuntu/WildFire_Capstone/Data"

# Directories for ppt and tmax within the 2023 folder
declare -A DIRS=(
    [ppt]="${BASE_DATA_DIR}/ppt/2023"
    [tmax]="${BASE_DATA_DIR}/tmax/2023"
)

# Month for the year 2023, specifically September
MONTH=9  # Use a number without leading zero

download_and_unzip() {
    local data_type=$1
    local month=$2
    local target_dir=${DIRS[$data_type]}
    # Ensure the month is formatted with two digits
    local year_month=$(printf "2023%02d" $month)
    local file_name="PRISM_${data_type}_provisional_4kmM3_${year_month}_bil.zip"
    local file_url="${BASE_URL}${data_type}/2023/${file_name}"
    local zip_file="${target_dir}/${file_name}"

    # Ensure the target directory exists
    mkdir -p "$target_dir"

    echo "Attempting to download ${file_name}..."
    if wget -q "$file_url" -O "$zip_file"; then
        echo "Downloaded ${file_name} successfully."
        # Optionally, unzip the file
        echo "Unzipping ${file_name}..."
        if unzip -q -o "$zip_file" -d "$target_dir"; then
            echo "Unzipped ${file_name} successfully."
        else
            echo "Failed to unzip ${file_name}."
        fi
        # Optionally, remove the zip file after extraction
        rm "$zip_file"
    else
        echo "Failed to download ${file_name}. Please check the URL and file availability."
    fi
}

for data_type in "${!DIRS[@]}"
do
    download_and_unzip "$data_type" $MONTH
done

echo "Attempt to download missed data completed."
