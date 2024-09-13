#!/bin/bash

# Base URL for data
BASE_URL="https://ftp.prism.oregonstate.edu/monthly/"

# Directories to download from with their respective target directories
declare -A DIRS=(
    [ppt]="/home/ubuntu/WildFire_Capstone/Data/ppt"
    [tmax]="/home/ubuntu/WildFire_Capstone/Data/tmax"
)

# Years to process
YEARS=(2022 2023)

download_and_unzip() {
    local dir_key=$1
    local year=$2
    local month=$3
    # Ensure target directory exists
    local target_dir="${DIRS[$dir_key]}/${year}"
    mkdir -p "$target_dir"
    # Format the month to two digits
    printf -v month_formatted "%02d" $month
    local file_name="PRISM_${dir_key}_stable_4kmM3_${year}${month_formatted}_bil.zip"
    local file_url="${BASE_URL}${dir_key}/${year}/${file_name}"
    local zip_file="${target_dir}/${file_name}"
    # Download and unzip
    if wget -q "$file_url" -O "$zip_file"; then
        echo "Downloaded ${file_name} to ${target_dir}"
        unzip -q -o "$zip_file" -d "$target_dir"
        echo "Extracted: ${file_name} in ${target_dir}"
        rm "$zip_file"
    else
        echo "File for $year, month $month in $dir_key not found, skipping..."
        echo "${year}-${month}" >> "${target_dir}/${dir_key}_missing_files.txt"
    fi
}

for dir_key in "${!DIRS[@]}"; do
    echo "Processing directory: $dir_key"
    for year in "${YEARS[@]}"; do
        for month in {1..12}; do
            download_and_unzip "$dir_key" "$year" "$month"
        done
    done
done

echo "Processing completed."
