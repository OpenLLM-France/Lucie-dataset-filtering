#!/bin/bash

root_output_folder=$1
root_input_folder=$2


# Array of dataset names
datasets=("eurovoc_es")
# Loop through each dataset and execute sbatch command
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    input_folder="${root_input_folder}/${dataset}"
    output_folder="${root_output_folder}/${dataset}"
    for file in "$input_folder"/*.parquet; do
        filename=$(basename "$file")
        if [ ! -f "$output_folder/$filename" ]; then
            echo "Processing $filename"
            sbatch job_jz_parquet_wise_es.sh "${dataset}/${filename}"
        else
            echo "$filename already processed"
        fi
    done
done
