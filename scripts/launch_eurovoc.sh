#!/bin/bash

langs=('de' 'it' 'en' 'es')

for lang in "${langs[@]}"; do
    ./submit_all_parquet_wise_${lang}.sh /gpfswork/rech/qgz/commun/data/corpus_openllm /gpfswork/rech/ksy/uyc63jm/DATA/corpus_openllm_processed
done
