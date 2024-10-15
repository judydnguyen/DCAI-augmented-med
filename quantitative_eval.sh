#!/bin/bash

# Define the paths
REAL_FOLDER_PATH="custom_covid_dataset/train"
FAKE_FOLDER_PATH="custom_covid_dataset/train_synthetic_proto/2000"

# Define the suffixes
REAL_FOLDER_SUFFIX=("covid" "normal" "pneumonia_bac" "pneumonia_vir")

# Loop through each suffix and calculate the FID score
for suffix in "${REAL_FOLDER_SUFFIX[@]}"
do
    REAL_PATH="$REAL_FOLDER_PATH/$suffix"
    FAKE_PATH="$FAKE_FOLDER_PATH/$suffix"

    echo "Calculating FID score for: $suffix"
    
    # Execute the FID calculation
    python -m pytorch_fid "$REAL_PATH" "$FAKE_PATH"
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo "FID score for $suffix calculated successfully."
    else
        echo "Error calculating FID score for $suffix."
    fi
done
