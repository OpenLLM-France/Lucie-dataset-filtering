#!/bin/bash

# This script is used to launch the processing of the LEGI data

# Load conda

eval "$(conda shell.bash hook)"

conda activate /scratch/envs/conda/redpajama

# 'url' possible pour mapping_fields

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/worker.py \
--path_dataset /scratch/openllm/data/LEGI \
--path_output /scratch/openllm/data/LEGI_processed \
--path_fasttext_model /scratch/openllm/data/lid.176.bin \
--path_perplexity_models /data/openllm/ccnet_models/cc_net/data/lm_sp \
--path_words_filter /scratch/openllm/data/ldnoobw \
--path_domain_filter /scratch/openllm/data/ut1 \
--path_cut_offs /scratch/openllm/data/cut_offs.json \
--mapping_fields '{"raw_content": "text", "date_download": "date"}' \
--default_fields '{"source_domain": "legifrance.gouv.fr"}' \
--fields_to_keep '["id"]' \
--language 'fr' \
--n_processes 32
