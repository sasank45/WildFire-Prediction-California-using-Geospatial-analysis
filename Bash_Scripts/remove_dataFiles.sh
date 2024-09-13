#!/bin/bash

# Define the directory to search in
search_dir="/home/ubuntu/POC/ppt"

# Verify that the directory exists
if [ ! -d "$search_dir" ]; then
    echo "The directory $search_dir does not exist."
    exit 1
fi

# Define the specific file extensions to search for and delete
extensions=(".tif" ".tif.aux.xml")

# Loop through each file extension
for ext in "${extensions[@]}"; do
    echo "Searching for files ending with $ext in $search_dir"
    # Find files ending with the current extension
    files_found=$(find "$search_dir" -type f -name "*$ext")
    if [ -z "$files_found" ]; then
        echo "No files found ending with $ext."
    else
        echo "The following files will be deleted:"
        echo "$files_found"
        # Ask for confirmation before deletion
        read -p "Are you sure you want to delete these files? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "$files_found" | while read file; do
                echo "Removing $file"
                rm -v "$file"
            done
        else
            echo "Deletion cancelled."
        fi
    fi
done

echo "Process completed."
