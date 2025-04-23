#!/bin/bash

# Paths and setup
output_file="test_sets.txt"
csv_output="results.csv"
python_script="python3 evaluation.py"

# Header for CSV
echo "TestSetID,mAP,Precision,Recall" > "$csv_output"

# Loop through each test set ID
while IFS= read -r line; do
    id=$(echo "$line" | awk '{print $1}')
    echo "Processing test set ID: $id"

    # Step 1: Prepare files
    cd ../experiment_setup/ || exit 1
    ./select_set_retina.sh "$output_file" "$id"
    if [[ $? -ne 0 ]]; then
        echo "Error processing test set $id. Skipping."
        cd - || exit 1
        continue
    fi
    cd - || exit 1

    # Step 2: Run Python model and capture output
    rm -r widerface_evaluate/widerface_txt/*
    python3 test_widerface.py --trained_model Resnet50_Final.pth --network resnet50 -s
    cd widerface_evaluate || exit 1
    output=$($python_script 2>&1)
    cd .. || exit 1

    # Step 3: Extract metrics using grep/sed
    val_ap=$(echo "$output" | grep "Val AP:" | sed 's/.*Val AP:[[:space:]]*//')
    precision=$(echo "$output" | grep "Precision:" | sed 's/.*Precision:[[:space:]]*//')
    recall=$(echo "$output" | grep "Recall:" | sed 's/.*Recall:[[:space:]]*//')

    # Step 4: Append to CSV
    echo "$id,$val_ap,$precision,$recall" >> "$csv_output"

done < "../experiment_setup/$output_file"

echo "✅ All test sets processed. Results saved to $csv_output"
