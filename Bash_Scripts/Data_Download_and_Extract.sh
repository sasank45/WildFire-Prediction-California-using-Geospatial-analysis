#!/bin/bash

# Base URL
BASE_URL="https://ftp.prism.oregonstate.edu/monthly/"

# Directories to download from and their target directories
declare -A DIRS=(
    [ppt]="/home/ubuntu/WildFire_Capstone/Data/ppt"
    [tmax]="/home/ubuntu/WildFire_Capstone/Data/tmax"
)

# Years to process
YEARS=({2013..2023})

download_and_unzip() {
    local dir_key=$1
    local year=$2
    local base_target_dir=${DIRS[$dir_key]} # The base directory for ppt or tmax
    local year_target_dir="${base_target_dir}/${year}" # Append the year to create a specific directory for each year
    mkdir -p "$year_target_dir" # Ensure the directory exists
    local file_url="${BASE_URL}${dir_key}/${year}/PRISM_${dir_key}_stable_4kmM3_${year}_all_bil.zip"
    local zip_file="${year_target_dir}/${dir_key}_${year}_all_bil.zip"
    if wget -q "$file_url" -O "$zip_file"; then
        echo "Downloaded ${dir_key}_${year}_all_bil.zip to $year_target_dir"
        unzip -q -o "$zip_file" -d "$year_target_dir"
        rm "$zip_file"
    else
        echo "File for $year in $dir_key not found, skipping..."
        echo "$year" >> "${base_target_dir}/${dir_key}_missing_years.txt"
    fi
}

for dir_key in "${!DIRS[@]}"; do
    echo "Processing directory: $dir_key"
    for year in "${YEARS[@]}"; do
        download_and_unzip "$dir_key" "$year"
    done
done

echo "Processing completed."
