#!/bin/bash
#SBATCH --account=ksy@cpu
#SBATCH --job-name=Preprocess
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH -c 40
#SBATCH --hint=nomultithread
#SBATCH --partition=cpu_p1

source ~/.bashrc

module load python/3.9.12

eval "$(conda shell.bash hook)"

conda activate python-3.9.12

folder_dataset=$1

DIR_ASSETS=${CODE_DIR}/assets
DIR_DATASET=/gpfswork/rech/qgz/commun/data/corpus_openllm/${folder_dataset}
DIR_OUTPUT=/gpfswork/rech/ksy/uyc63jm/DATA/corpus_openllm_processed

python ${CODE_DIR}/Bloom-ng-dataset-processing/src/blmrdata/utils/redpajama/worker_parquet.py \
--dir-input $DIR_DATASET \
--dir-output $DIR_OUTPUT \
--path_fasttext_model ${DIR_ASSETS}/fasttext/lid.176.bin \
--dir_perplexity_models ${DIR_ASSETS}/ccnet_models \
--language 'fr' \
--n-processes 38 \
--flush-freq 1000
