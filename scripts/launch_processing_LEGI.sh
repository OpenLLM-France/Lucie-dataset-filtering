#!/bin/bash

# This script is used to launch the processing of the LEGI data

eval "$(conda shell.bash hook)"

NAME_ENV="/scratch/envs/conda/redpajama"
conda activate $NAME_ENV # or source [...]/bin/activate if using venv

# mapping_fields can take the following values:
# - 'url' : url of the page
# - 'title' : title of the page
# - 'raw_content' : text of the page
# - 'date_download' : download date of the page
# - 'source_domain' : domain name of the page

# default_fields allows to set default value for fields of all pages, example:
# - 'source_domain' : for instance here the domain name (i.e. legifrance.gouv.fr)

DIR_ASSETS=/data/assets
DIR_DATASET=/scratch/openllm/data/LEGI
DIR_OUTPUT=/scratch/openllm/data/LEGI_processed

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/worker.py \
--dir_dataset $DIR_DATASET \
--dir_output $DIR_OUTPUT \
--path_fasttext_model ${DIR_ASSETS}/fasttext/lid.176.bin \
--dir_perplexity_models ${DIR_ASSETS}/ccnet_models \
--dir_words_filter ${DIR_ASSETS}/ldnoobw \
--dir_domain_filter ${DIR_ASSETS}/ut1 \
--path_cut_offs ${DIR_ASSETS}/cut_offs.json \
--mapping_fields '{"raw_content": "text", "date_download": "date"}' \
--default_fields '{"source_domain": "legifrance.gouv.fr"}' \
--fields_to_keep '["id"]' \
--language 'fr' \
--n_processes 32 \
--flush-freq 1000 \
--minhash-similarities "[1.0, 0.9, 0.8, 0.7]" \
--size-shard -1 \