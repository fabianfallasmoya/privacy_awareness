#!/bin/bash

# Paths and setup
output_file="test_sets.txt"
csv_output="results.csv"
python_script="python3 run_val.py"

# Header for CSV
echo "TestSetID,mAP,Precision,Recall" > "$csv_output"

# Loop through each test set ID
while IFS= read -r line; do
    id=$(echo "$line" | awk '{print $1}')
    echo "Processing test set ID: $id"

    # Step 1: Prepare the dataset
    cd ../experiment_setup/ || exit 1
    ./select_set_yolo.sh "$output_file" "$id"
    if [[ $? -ne 0 ]]; then
        echo "Error processing test set $id. Skipping."
        cd - || exit 1
        continue
    fi
    cd - || exit 1

    # Step 2: Run the model and capture output
    output=$($python_script 2>&1)

    # Step 3: Extract the metrics from the line starting with "all"
    metrics_line=$(echo "$output" | grep -E '^\s+all\s+[0-9]')
    if [[ -z "$metrics_line" ]]; then
        echo "Warning: Metrics not found for test set $id. Skipping."
        continue
    fi

    # Parse the values
    precision=$(echo "$metrics_line" | awk '{print $5}')
    recall=$(echo "$metrics_line" | awk '{print $6}')
    map50=$(echo "$metrics_line" | awk '{print $7}')

    # Step 4: Append to CSV
    echo "$id,$map50,$precision,$recall" >> "$csv_output"

done < "../experiment_setup/$output_file"

echo "✅ All test sets processed. Results saved to $csv_output"
