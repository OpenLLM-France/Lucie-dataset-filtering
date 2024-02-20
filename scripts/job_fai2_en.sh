#!/bin/bash
#SBATCH --job-name=Preprocess
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH -c 72
#SBATCH --mem=180G
#SBATCH --exclusive
#SBATCH --partition=allcpu

eval "$(conda shell.bash hook)"

conda activate redpajama

folder_dataset=$1

DIR_ASSETS=${CODE_DIR}/assets
DIR_DATASET=/home/data/edufraisse/DATA/LUCIE/${folder_dataset}
DIR_OUTPUT=/scratch/edufraisse/perplexity_corpus_open_llm
mkdir -p $DIR_OUTPUT

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/worker_parquet.py \
--dir-input $DIR_DATASET \
--dir-output $DIR_OUTPUT \
--final-dir-output /home/data/edufraisse/DATA/LUCIE/perplexity_corpus_open_llm \
--path_fasttext_model ${DIR_ASSETS}/fasttext/lid.176.bin \
--dir_perplexity_models ${DIR_ASSETS}/ccnet_models \
--language 'fr' \
--n-processes 72 \
--flush-freq 1000

rsync -avh --progress $DIR_OUTPUT/ /home/data/edufraisse/DATA/LUCIE/perplexity_corpus_open_llm/ && rm -rf $DIR_OUTPUT