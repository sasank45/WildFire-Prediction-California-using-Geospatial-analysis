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

# Months for the year 2023, from September to December
MONTHS=(09 10 11 12)

download_and_unzip() {
    local data_type=$1
    local year_month=$2
    local target_dir=${DIRS[$data_type]}
    local file_name="PRISM_${data_type}_provisional_4kmM3_${year_month}_bil.zip"
    local file_url="${BASE_URL}${data_type}/2023/${file_name}"
    local zip_file="${target_dir}/${file_name}"

    # Ensure the target directory exists
    mkdir -p "$target_dir"

    echo "Downloading ${file_name}..."
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
        echo "Failed to download ${file_name}."
    fi
}

for data_type in "${!DIRS[@]}"
do
    echo "Processing $data_type..."
    for month in "${MONTHS[@]}"
    do
        download_and_unzip "$data_type" "2023$(printf '%02d' $month)"
    done
done

echo "All processing completed."
