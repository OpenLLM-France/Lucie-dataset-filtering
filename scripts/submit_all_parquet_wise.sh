#!/bin/bash

root_output_folder=$1
root_input_folder=$2


# Array of dataset names
datasets=("discours_publics_parquet" "gallica_mono_parquet" "gallica_presse_html_parquet" "gallica_presse_txt_parquet" "hal_parquet" "legi_fr_parquet" "open_edition_parquet" "other_fr_parquet" "persee_parquet" "theses_parquet")

# Loop through each dataset and execute sbatch command
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"
    input_folder="${root_input_folder}/${dataset}"
    output_folder="${root_output_folder}/${dataset}"
    for file in "$input_folder"/*.parquet; do
        filename=$(basename "$file")
        if [ ! -f "$output_folder/$filename" ]; then
            echo "Processing $filename"
            sbatch job_jz_parquet_wise.sh "${dataset}/${filename}"
        else
            echo "$filename already processed"
        fi
    done
done
