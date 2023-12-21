#!/bin/bash

# This script is used to launch the processing of the LEGI data

DIR_ASSETS=/data/openllm/assets
DIR_DATASET=/scratch/openllm/data/LEGI
DIR_OUTPUT=/scratch/openllm/data/LEGI_processed

eval "$(conda shell.bash hook)"

conda activate /scratch/envs/conda/redpajama

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/exact_deduplication.py \
--listings ${DIR_OUTPUT}/listings/listings.txt \
--input_base_uri file://${DIR_OUTPUT} \
--output_dir ${DIR_OUTPUT}/duplicates \
--parallel_readers 4 \
--batch_size 8 \
--seed 42 \
--capacity 100000000 \
--error_rate 0.01 \
--custom-dataset
