#!/bin/bash

# Args
output_file="$1"
row_number="$2"
file_a="../retina-face/data/widerface/val/wider_val.txt"
file_b="../retina-face/data/widerface/val/wider_face_val_bbx_gt.txt"

file_a_orig="../retina-face/data/widerface/val_bk/wider_val.txt"
file_b_orig="../retina-face/data/widerface/val_bk/wider_face_val_bbx_gt.txt"

# Temp files
temp_a="filtered_a.tmp"
temp_b="filtered_b.tmp"

# Check input
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <output_file> <row_number>"
    exit 1
fi

if [[ ! -f "$output_file" ]]; then
    echo "Error: Output file '$output_file' does not exist."
    exit 1
fi

if [[ ! -f "$file_a" || ! -f "$file_b" ]]; then
    echo "Error: One or both of the target files do not exist."
    exit 1
fi

if [[ ! -f "$file_a_orig" || ! -f "$file_b_orig" ]]; then
    echo "Error: One or both of the original files do not exist."
    exit 1
fi

# Restore the files
cp "${file_a_orig}" "${file_a}"
cp "${file_b_orig}" "${file_b}"

# Trick to quickly restore the files
if [[ "$row_number" -eq 0 ]]; then
    exit 1
fi

# --- Extract the combination set line ---
set_line=$(awk -v row="$row_number" 'NR == row' "$output_file")
if [[ -z "$set_line" ]]; then
    echo "Error: Could not find row $row_number in $output_file"
    exit 1
fi

echo "Using set: $set_line"

# Get just the numbers (skip the set number)
read -a numbers <<< "$set_line"
numbers=("${numbers[@]:1}")

# --- Build a list for awk ---
number_list=$(IFS=,; echo "${numbers[*]}")

# --- Filter File A ---
awk -v list="$number_list" '
    BEGIN {
        split(list, keep, ",")
        for (i in keep) line_map[keep[i]] = 1
    }
    {
        if (line_map[FNR]) print
    }
' "$file_a" > "$temp_a"

if [[ ! -s "$temp_a" ]]; then
    echo "Warning: Filtered file A is empty. Aborting."
    exit 1
fi

# --- Step 2: Extract section headers from File B ---
grep -- "--" "$file_b" > __headers.tmp

# --- Step 3: Keep only headers that are substrings of any line in File A ---
# Create a list of matching headers
> __keep_headers.tmp
while IFS= read -r header; do
    if grep -Fq "$header" "$temp_a"; then
        echo "$header" >> __keep_headers.tmp
    fi
done < __headers.tmp

# Always keep the --/ section
echo "--/" >> __keep_headers.tmp

# --- Step 4: Filter File B based on those headers ---
awk '
    BEGIN {
        while ((getline line < "__keep_headers.tmp") > 0) keep_headers[line] = 1
        keep = 0
    }
    /^.*--.*$/ {
        keep = ($0 in keep_headers)
    }
    {
        if (keep) print
    }
' "$file_b" > "$temp_b"

# Finalize
mv "$temp_a" "$file_a"
mv "$temp_b" "$file_b"
rm -f __headers.tmp __keep_headers.tmp

echo "Done. Filtered $file_a and $file_b based on set #$row_number"
