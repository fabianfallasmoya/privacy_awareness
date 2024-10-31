#!/bin/bash

# Directory to start from
start_dir="WIDER_val_PA/images/"

# Find all directories recursively
find "$start_dir" -type d | while read -r dir; do
  # Find all jpg files in the directory, sort them by name, and store in an array
  jpg_files=($(find "$dir" -maxdepth 1 -type f -iname "*.jpg" | sort))

  # Check if the array has more than 5 elements
  if [ ${#jpg_files[@]} -gt 5 ]; then
    # Delete all but the first 5 files
    for ((i=5; i<${#jpg_files[@]}; i++)); do
      echo "Deleting: ${jpg_files[$i]}"
      rm "${jpg_files[$i]}"
    done
  fi
done

