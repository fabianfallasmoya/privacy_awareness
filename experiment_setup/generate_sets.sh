#!/bin/bash

# Configuration
output_file="test_sets.txt"
total_sets=100
numbers_per_set=100
max_number=275

# Clear output file
> "$output_file"

for ((i=1; i<=total_sets; i++)); do
    # Generate unique random numbers
    combo=($(shuf -i 1-$max_number -n $numbers_per_set | sort -n))
    
    # Join numbers into a single line
    combo_str=$(printf "%s " "${combo[@]}")
    
    # Write to file: "SetNumber Num1 Num2 ..."
    echo "$i ${combo_str}" >> "$output_file"
done

echo "File '$output_file' created with $total_sets combinations."