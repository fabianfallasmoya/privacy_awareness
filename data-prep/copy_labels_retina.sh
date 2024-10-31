#!/bin/bash

# Check for the required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <jpg_list_file> <directory_A>"
    exit 1
fi

jpg_list_file=$1
directory_A=$2
output_file="wider_face_val_bbx_gt.txt"

# Check if the jpg list file exists
if [ ! -f "$jpg_list_file" ]; then
    echo "The file $jpg_list_file does not exist."
    exit 1
fi

# Check if directory A exists
if [ ! -d "$directory_A" ]; then
    echo "The directory $directory_A does not exist."
    exit 1
fi

# Empty the output file if it exists, or create it
> "$output_file"

# Function to convert YOLO format to bounding box coordinates and round them to integers
yolobbox2bbox() {
    local x="$1"
    local y="$2"
    local w="$3"
    local h="$4"
    local x1 y1 x2 y2
    x1=$(printf "%.0f" "$(echo "$x - $w / 2" | bc -l)")
    y1=$(printf "%.0f" "$(echo "$y - $h / 2" | bc -l)")
    x2=$(printf "%.0f" "$(echo "$x + $w / 2" | bc -l)")
    y2=$(printf "%.0f" "$(echo "$y + $h / 2" | bc -l)")
    echo "$x1 $y1 $x2 $y2"
}

# Function to convert YOLO format to Coco bounding box coordinates and round them to integers
yolobbox2coco() {
    local x_center="$1"
    local y_center="$2"
    local width="$3"
    local height="$4"
    
    # Calculate x_min and y_min, rounding to the nearest integer
    x_min=$(printf "%.0f" "$(echo "$x_center - $width / 2" | bc -l)")
    y_min=$(printf "%.0f" "$(echo "$y_center - $height / 2" | bc -l)")
    
    # Round width and height to the nearest integer as well
    width=$(printf "%.0f" "$width")
    height=$(printf "%.0f" "$height")
    
    # Output the COCO-format bounding box
    echo "$x_min $y_min $width $height"
}

# Read each line in the jpg list file
while IFS= read -r line; do
    match_string=$(echo "$line" | cut -d'/' -f3- | sed 's/\./_/g')
    new_file_name=$(echo "$line" | cut -d'/' -f2-)

    # Find the best-matching .txt file in directory A using fzf for fuzzy matching (requires fzf to be installed)
    best_match=$(find "$directory_A" -type f -name "*.txt" | fzf --query="$match_string" --select-1 --exit-0)
    
    # Check if a matching file was found
    if [ -n "$best_match" ]; then
        # Get the line count of the best-matching file
        count=$(wc -l < "$best_match")
        
        # Append the current line (jpg filename) and the count to the output file
        echo "$new_file_name" >> "$output_file"
        echo "$count" >> "$output_file"

        # Read each line in the best-matching txt file
        while IFS= read -r bbox_line; do
            # Split the line into fields, extract bounding box coordinates, and ignore the first field
            read -r _ x y w h pa <<< "$bbox_line"
            
            # Multiply x, y, w, h by 640
            x=$(printf "%.0f" "$(echo "$x * 640" | bc -l)")
            y=$(printf "%.0f" "$(echo "$y * 640" | bc -l)")
            w=$(printf "%.0f" "$(echo "$w * 640" | bc -l)")
            h=$(printf "%.0f" "$(echo "$h * 640" | bc -l)")
            #bbox_coordinates=$(echo "$x $y $w $h")
            
            # Convert YOLO bbox format to (x1, y1, x2, y2)
            bbox_coordinates=$(yolobbox2coco "$x" "$y" "$w" "$h")
            
            # Append bbox coordinates and the sixth number to the output file
            echo "$bbox_coordinates $pa" >> "$output_file"
        done < "$best_match"
        
        echo "Processed $line with match $best_match, line count: $count"
    else
        echo "No match found for $line in $directory_A."
    fi
done < "$jpg_list_file"

echo "--/" >> "$output_file"

echo "Operation completed. Results saved to $output_file."
