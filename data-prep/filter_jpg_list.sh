#!/bin/bash

# Check if the right number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <txt_file> <directory>"
    exit 1
fi

txt_file=$1
directory=$2

# Check if the txt file exists
if [ ! -f "$txt_file" ]; then
    echo "The file $txt_file does not exist."
    exit 1
fi

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "The directory $directory does not exist."
    exit 1
fi

# Find all jpg files in the directory (recursively) and store them in an array
mapfile -t jpg_files < <(find "$directory" -type f -name "*.jpg")

# Create a temporary file to store matched lines
temp_file=$(mktemp)

# Loop through each line in the txt file
while IFS= read -r line; do
    match_string=$(echo "$line" | cut -d'/' -f3- | sed 's/\./_/g')
    match_found=false
    # Check if any jpg file in the directory contains the line as a substring
    for jpg_file in "${jpg_files[@]}"; do
        if [[ "$jpg_file" == *"$match_string"* ]]; then
            match_found=true
            break
        fi
    done
    # If a match was found, add the line to the temp file
    if $match_found; then
        echo "$line" >> "$temp_file"
    fi
done < "$txt_file"

# Replace the original file with the temp file
mv "$temp_file" "$txt_file"

echo "Filtering complete. Unmatched .jpg names have been removed from $txt_file."
