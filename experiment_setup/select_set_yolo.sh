#!/bin/bash

# This script takes the complete custom dataset and deletes the items that are not present in the test_sets.txt (product of the generate_sets.sh)

# Example, run from root:
# ./experiment_setup/select_set_yolo.sh ./experiment_setup/test_sets.txt 1

# Args
output_file="$1"
row_number="$2"

dataset_name="WIDERFACE_PA_yolov8"
dataset_path="/home/jcordero/work/OD/datasets/customPA/${dataset_name}"
path_bk="/home/jcordero/work/OD/datasets/customPA_bk/${dataset_name}"

path_a="${dataset_path}/valid/images"
path_b="${dataset_path}/valid/labels"
path_cache="${dataset_path}/valid/labels.cache"

path_a_orig="${path_bk}/valid/images"
path_b_orig="${path_bk}/valid/labels"

# Check arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <output_file> <row_number>"
    exit 1
fi

# Check file and directory existence
if [[ ! -f "$output_file" ]]; then
    echo "Error: Output file '$output_file' does not exist."
    exit 1
fi
if [[ ! -d "$path_a" || ! -d "$path_b" ]]; then
    echo "Error: One or both of the images or labels directories do not exist."
    exit 1
fi
if [[ ! -d "$path_bk" ]]; then
    echo "Error: Backup directory '$path_bk' does not exist."
    exit 1
fi

# Restore the directory to the initial state
# Clean cache if it exists
if [[ -f "$path_cache" ]]; then
    echo "Cleaning cache file"
    rm "$path_cache"
fi
# Restore the images and labels from the backup
cp -r "${path_a_orig}" "${dataset_path}/valid/"
cp -r "${path_b_orig}" "${dataset_path}/valid/"

# Trick to quickly reset to complete list of images and labels
if [[ "$row_number" -eq 0 ]]; then
    exit 1
fi

# Extract line from file
line=$(awk -v row="$row_number" 'NR==row' "$output_file")
if [[ -z "$line" ]]; then
    echo "Error: No line found for row $row_number"
    exit 1
fi

# Get the number combination (remove the first "set number")
read -a combo <<< "$line"
combo=("${combo[@]:1}")  # Remove first element (set number)

# Convert combo to associative array for quick lookup
declare -A combo_map
for num in "${combo[@]}"; do
    combo_map[$num]=1
done

# Function to filter directory files
filter_and_delete() {
    local dir_path="$1"
    local extension="$2"
    local count=1

    # Sort files naturally
    mapfile -t files < <(find "$dir_path" -maxdepth 1 -type f -iname "*.$extension" | sort -V)

    for file in "${files[@]}"; do
        if [[ -z "${combo_map[$count]}" ]]; then
            echo "Deleting: $file"
            rm -f "$file"
        fi
        ((count++))
    done
}

# Process both directories
filter_and_delete "$path_a" "jpg"
filter_and_delete "$path_b" "txt"

echo "Done processing set $row_number."
