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
DIR_DATASET=/data/original_gallica
DIR_OUTPUT=/data/processed_gallica

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/worker_parquet.py \
--dir-input $DIR_DATASET \
--dir-output $DIR_OUTPUT \
--path_fasttext_model ${DIR_ASSETS}/fasttext/lid.176.bin \
--dir_perplexity_models ${DIR_ASSETS}/ccnet_models \
--language 'fr' \
--n-processes 32 \
--flush-freq 1000
