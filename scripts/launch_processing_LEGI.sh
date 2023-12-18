#!/bin/bash

# This script is used to launch the processing of the LEGI data

# Load conda

eval "$(conda shell.bash hook)"

conda activate /scratch/envs/conda/redpajama

    # parser.add_argument(
    #     "--path_dataset",
    #     type=str,
    #     help="Path to the dataset to process",
    # )
    # parser.add_argument(
    #     "--path_output",
    #     type=str,
    #     help="Path to the output folder",
    # )
    # parser.add_argument(
    #     "--path_fasttext_model",
    #     type=str,
    #     help="Path to the fasttext model",
    # )
    # parser.add_argument(
    #     "--path_perplexity_models",
    #     type=str,
    #     help="Path to the perplexity models",
    # )
    # parser.add_argument(
    #     "--path_words_filter",
    #     type=str,
    #     help="Path to the words filter",
    # )
    # parser.add_argument(
    #     "--path_domain_filter",
    #     type=str,
    #     help="Path to the domain filter",
    # )
    # parser.add_argument(
    #     "--path_cut_offs",
    #     type=str,
    #     help="Path to the cut offs",
    # )
    # parser.add_argument(
    #     "--mapping_fields",
    #     type=str,
    #     help="Mapping fields",
    # )
    # parser.add_argument(
    #     "--default_fields",
    #     type=str,
    #     help="Default fields",
    # )
    # parser.add_argument(
    #     "--fields_to_keep",
    #     type=str,
    #     help="Fields to keep",
    # )
    # parser.add_argument(
    #     "--language",
    #     type=str,
    #     help="Language",
    # )
    # parser.add_argument(
    #     "--n_processes",
    #     type=int,
    #     help="Number of processes",
    # )

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
