# Data Pre-processing

Code to process English and French datasets using ccnet and gopher, refineweb and redpajamaV2 metrics.


## Note: todo

- Near-deduplication is not yet implemented, but minhashes are computed.

# Process

## Installation

Installation involves a modified version of the RedPajama packaged code, which is automatically cloned during installation and available [here](https://github.com/EvanDufraisse/RedPajamaV2-Utils.git)
requirements: python 3.8+

```bash
cd Lucie-dataset-filtering
pip install -e ./
```

## Download assets

See function `download_ccnet_models` in [the documentation about CCNet Perplexity](./src/blmrdata/utils/ccnet/README.md) to download ccnet models programmatically.

You can also download ccnet models using:
```
cd assets/ccnet_models/
wget http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).arpa.bin
wget http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).sp.model
```
where `$(lang)` is the language code (e.g. `en` for English).

The fasttext language identification model can be downloaded from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin and placed in `assets/fasttext/lid.176.bin`.

## Processing Steps

This assumes that your dataset is a compressed jsonl file (e.g. a jsonl.gz) (i.e. one json per line).

In `./scripts' you will find the scripts to process the data for LEGI, replace variables with the correct values for you.

### 1 - Computing the metrics
The first script `launch_processing_LEGI` will compute the perplexity of all documents using the CCNET models. Currently only FR and EN are supported, but in practice many CCNET models are available. It will also compute the minhashes of the documents, exact hashes of the documents for exact deduplication, and the Gopher, RefinedWeb, and RedpajamaV2 metrics. The metrics from the paper `Importance Resampling (Xie et al.)' are not implemented because they rely on models that are not available.

The file called is `Lucie-dataset-filtering/src/blmrdata/utils/redpajama/worker.py`, and the arguments are the following:


```bash

python ${CODE_DIR}/Lucie-dataset-filtering/src/blmrdata/utils/redpajama/worker.py \
--dir_dataset $DIR_DATASET \ # Path to the dataset folder
--dir_output $DIR_OUTPUT \ # Path to the output folder
--path_fasttext_model ${DIR_ASSETS}/fasttext/lid.176.bin \ # Path to the fasttext model
--dir_perplexity_models ${DIR_ASSETS}/ccnet_models \ # Path to the ccnet models for perplexity
--dir_words_filter ${DIR_ASSETS}/ldnoobw \ # Path to the words filter for bad language detection
--dir_domain_filter ${DIR_ASSETS}/ut1 \ # Path to the domain filter for bad domain detection
--path_cut_offs ${DIR_ASSETS}/cut_offs.json \ # Path to the cut offs found by the ccnet approach on RedPajamaV2
--mapping_fields '{"raw_content": "text", "date_download": "date"}' \ # Mapping between the fields of the source dataset and the fields used in redpajama (see script for more details)
--default_fields '{"source_domain": "legifrance.gouv.fr"}' \ # Default fields to add to the dataset if not present in the source dataset
--fields_to_keep '["id"]' \ # Fields not present in redpajama but to keep in the dataset
--language 'fr' \ # Language of the dataset
--n_processes 32 \ # Number of processes to use (duplicates the whole pipeline n_processes times)
--flush-freq 1000 \ # Frequency of flushing the results to disk
--minhash-similarities "[1.0, 0.9, 0.8, 0.7]" \ # Similarities to compute for minhashes
--size-shard -1 \ # Size of the shards to use, -1 means no sharding

```

In the output folder you will find the following folders:

- `documents`: contain the text, metadata and some metrics but only a subset as in RedPajamaV2
- `listings`: contain the list of files to process for the next steps
- `minhashes`: contain the minhashes of the documents
- `quality_signals`: contain the quality signals of the documents

Documents are in the same order in each corresponding bucket output between `quality_signals` and `documents`, but not necessarily in the same order as the input dataset (due to  ultiprocessing).

As in RedpajamaV2, the output documents are separated into three buckets (tail, middle, head) according to their perplexity scores.

The unique ID to identify a document between different folders is the `id` field.

See example output of LEGI dataset in `output_legi.tar.gz`.

### 2 - Exact deduplication

The second script `launch_exact_dedup_LEGI` will compute the quasi-exact deduplication (bloom-filter) of the dataset. It will use the exact hashes computed in the previous step.

The output is a parquet file in `${DIR_OUTPUT}/duplicates`.

```bash

python ${CODE_DIR}/Lucie-dataset-filtering/src/blmrdata/utils/redpajama/exact_deduplication.py \
--listings ${DIR_OUTPUT}/listings/listings.txt # List files to process, built in previous step\
--input_base_uri file://${DIR_OUTPUT} \ # Output dir of the previous step
--output_dir ${DIR_OUTPUT}/duplicates \ # Output folder for the duplicates .parquet
--parallel_readers 4 \ # Number of parallel readers
--batch_size 8 \ # Batch size
--seed 42 \ # Seed for the bloom filter
--capacity 100000000 \ # Estimated number of documents to process (higher reduce collisions but increase memory usage)
--error_rate 0.01 \ # Target error rate (lower reduce collisions but increase memory usage)
--custom-dataset \ # If the dataset is not redpajamaV2
```

### 3 - Filtering

You can filter by implementing your own filter functions based on quality signals.
An example of the Gopher filter is provided in the description of the RedPajamaV2 dataset on huggingface. https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2.

## License

[MIT](https://choosealicense.com/licenses/mit/)
