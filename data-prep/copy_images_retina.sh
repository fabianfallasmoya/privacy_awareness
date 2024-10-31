#!/bin/bash

# Check for the required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <txt_file> <directory_A> <directory_B>"
    exit 1
fi

txt_file=$1
directory_A=$2
directory_B=$3

# Check if the txt file exists
if [ ! -f "$txt_file" ]; then
    echo "The file $txt_file does not exist."
    exit 1
fi

# Check if directory A exists
if [ ! -d "$directory_A" ]; then
    echo "The directory $directory_A does not exist."
    exit 1
fi

# Check if directory B exists
if [ ! -d "$directory_B" ]; then
    echo "The directory $directory_B does not exist."
    exit 1
fi

# Read each line in the txt file
while IFS= read -r line; do

    # Extract the second substring after splitting by slashes
    target_subdir=$(echo "$line" | cut -d'/' -f2)

    match_string=$(echo "$line" | cut -d'/' -f3- | sed 's/\./_/g')
    new_file_name=$(echo "$line" | cut -d'/' -f3-)
    
    # Find the best-matching .jpg file in directory A using fzf for fuzzy matching (requires fzf to be installed)
    best_match=$(find "$directory_A" -type f -name "*jpg" | fzf --query="$match_string" --select-1 --exit-0)
    
    # Check if a matching file was found
    if [ -n "$best_match" ]; then
        # Determine the destination directory in B
        destination_dir="$directory_B/$target_subdir"
        
        # Create the destination subdirectory if it doesn't exist
        #mkdir -p "$destination_dir"
        
        # Copy the best match to the destination directory
        cp "$best_match" "$destination_dir/$new_file_name"
        
        echo "Copied $best_match to $destination_dir/"
    else
        echo "No match found for $line in $directory_A."
    fi
done < "$txt_file"

echo "Operation completed."
