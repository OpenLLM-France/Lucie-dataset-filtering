# -*- coding: utf-8 -*-
""" Worker class to process any compressed jsonl dataset based input

@Author: Evan Dufraisse
@Date: Mon Dec 18 2023
@Contact: e[dot]dufraisse[at]gmail[dot]com
@License: MIT License
"""
import os
from xopen import xopen
import gzip, json
from tqdm.auto import tqdm
import multiprocessing as mp
from redpajama.core.quality_signals.content import register_content_callables
from redpajama.core.quality_signals.lines import register_lines_callables
from redpajama.core.quality_signals.natural_language import (
    register_natural_language_callables,
)
from redpajama.core.quality_signals.repetitions import register_repetitions_callables
from pathlib import Path
from loguru import logger
from blmrdata.utils.redpajama.metrics import FastLanguageIdentification
import gc
import sys
from blmrdata.utils.ccnet.perplexity import Perplexity
from blmrdata.utils.redpajama.chunk_text import chunk_text_simple, chunk_text_smart
from redpajama.core.document import Document
from redpajama.dedupe.minhash import MinHash

from redpajama.core.worker import _ccnet_bucket_to_int, ParquetBatchWriter
import hashlib
import pyarrow as pa
import argparse
from ast import literal_eval
import time
import re
import pyarrow.parquet as pq
import pdb
import pandas as pd
import gc
from typing import Union, List

_BYTE_ORDER = sys.byteorder


class DatasetProcessor(object):
    def __init__(
        self,
        path_fasttext_model: Path,
        path_perplexity_models: Path,
        text_chunk_strategy: Union[str, int] = "simple", # 10_000,
        n_processes: int = 1,
        regex_pattern: str = ".*.parquet",
        flush_freq: int = 1000,
    ):
        self.path_fasttext_model = path_fasttext_model
        self.path_perplexity_models = path_perplexity_models
        self.text_chunk_strategy = text_chunk_strategy
        self.n_processes = n_processes  # Num processing workers
        self.regex_pattern = regex_pattern
        self.flush_freq = flush_freq

    @staticmethod
    def get_files_of_interest(root_path: Path, regex_pattern: str = ".*.jsonl.gz"):
        """Get all files of interest in a given directory

        Args:
            root_path (str): path to the root directory
            regex_pattern (str, optional): regex pattern to match files of interest. Defaults to "*.jsonl.gz".

        Returns:
            list: list of files of interest
        """
        files_of_interest = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if re.match(regex_pattern, file):
                    files_of_interest.append(os.path.join(root, file))
        return sorted(files_of_interest)

    def process_dataset(
        self, dir_input: Path, dir_output: Path, final_dir_output: Path, language: Union[List[str], str]
    ):
        """Process a dataset and output a new one with the perplexity score"""

        if str(dir_input).endswith(".parquet"):
            files_of_interest = [dir_input]
        else:
            files_of_interest = self.get_files_of_interest(
                dir_input, self.regex_pattern
            )
        print(f"Found {len(files_of_interest)} files of interest", flush=True)
        if final_dir_output == "":
            final_dir_output = dir_output
        for idx_file, file in enumerate(files_of_interest):
            print(f"Processing file {file} {idx_file}/{len(files_of_interest)-1}")
            dirname_file = os.path.dirname(file).split("/")[-1]
            if not (os.path.exists(os.path.join(final_dir_output, dirname_file))):
                os.makedirs(os.path.join(final_dir_output, dirname_file), exist_ok=True)
            output_parquet = os.path.join(
                dir_output, dirname_file, os.path.basename(file)
            )
            final_output_parquet = os.path.join(
                final_dir_output, dirname_file, os.path.basename(file)
            )
            name_files_already_processed = set(
                [
                    f
                    for f in os.listdir(os.path.dirname(final_output_parquet))
                    if f.endswith(".parquet")
                ]
            )
            if os.path.basename(output_parquet) in name_files_already_processed:
                logger.info(
                    f"Skipping {os.path.basename(output_parquet)}, already processed"
                )
                continue

            parquet_file = pq.read_table(file)
            original_schema = parquet_file.schema
            original_fields = {field.name: field for field in original_schema}

            merged_schema = pa.schema(
                list(original_fields.values())
                + [
                    pa.field("chunk_start", pa.list_(pa.int32())),
                    pa.field("chunk_end", pa.list_(pa.int32())),
                    pa.field("ccnet_avg_log_prob", pa.list_(pa.float32())),
                    pa.field("ccnet_length", pa.list_(pa.int32())),
                    pa.field("ccnet_language_score", pa.list_(pa.float32())),
                    pa.field("fasttext_language", pa.list_(pa.string())),
                    pa.field("idx_row", pa.int32()),
                ]
            )

            del parquet_file

            gc.collect()
            manager = mp.Manager()
            writer_queue = manager.Queue()
            # Start writer process
            writer_process = mp.Process(
                target=writer,
                args=(writer_queue, output_parquet, merged_schema, self.flush_freq),
            )
            writer_process.start()
            processing_queues = [mp.Queue() for _ in range(self.n_processes)]
            worker_processes = [
                mp.Process(
                    target=worker,
                    args=(
                        processing_queues[i],
                        writer_queue,
                        self.path_fasttext_model,
                        self.path_perplexity_models,
                        self.text_chunk_strategy,
                        language,
                    ),
                )
                for i in range(self.n_processes)
            ]
            for p in worker_processes:
                p.start()

            gc.collect()

            for idx_row, row in enumerate(
                pq.read_table(file).to_pandas().itertuples(index=False)
            ):
                row = row._asdict()
                # argmin to get the queue with the smallest size
                smallest_queue_index = min(
                    range(len(processing_queues)),
                    key=lambda i: processing_queues[i].qsize(),
                )
                processing_queues[smallest_queue_index].put(("Process", row, idx_row))
            while (
                processing_queues[smallest_queue_index].qsize() > 10000
                and writer_queue.qsize() > 10000
            ):
                time.sleep(1)
            # processing_queues[smallest_queue_index].put(
            #     ("Close", None, uri_id, idx_line)
            # )

            for q in processing_queues:
                q.put(("STOP", None, None))

            for p in worker_processes:
                p.join()

            writer_queue.put(("STOP", None, None))
            writer_process.join()
        return


def open_quality_signals_jsonl_gz(uri_id, output_folder):
    return open_jsonl_gz(
        uri_id=uri_id,
        subfolder="quality_signals",
        output_folder=output_folder,
        extension=".signals.json.gz",
    )


def open_documents_jsonl_gz(uri_id, output_folder):
    return open_jsonl_gz(
        uri_id=uri_id,
        subfolder="documents",
        output_folder=output_folder,
        extension=".json.gz",
    )


def open_jsonl_gz(uri_id, subfolder, output_folder, extension=".json.gz"):
    path_gz = os.path.join(
        output_folder, subfolder, uri_id.split(".")[0].strip("/") + extension
    )
    dirname = os.path.dirname(path_gz)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return gzip.open(
        path_gz,
        "wt",
    )


def open_minhash_parquet(uri_id, output_folder, minhash_schema):
    output_fp = os.path.join(
        output_folder,
        "minhash",
        uri_id.split(".")[0].strip("/") + ".minhash.parquet",
    )
    dirname = os.path.dirname(output_fp)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return ParquetBatchWriter(
        output_fp=output_fp,
        schema=minhash_schema,
    )


def get_bucket_output_filename(uri_id, bucket):
    splitted_uri = uri_id.split(".")
    main_name = splitted_uri[0]
    new_name = main_name + "_" + bucket
    return new_name + "." + ".".join(splitted_uri[1:])


def get_all_buckets_output_filename(uri_id):
    all_bucket_outputs = {}
    for bucket in ["tail", "middle", "head"]:
        all_bucket_outputs[bucket] = get_bucket_output_filename(uri_id, bucket)
    return all_bucket_outputs


def writer(writer_queue, output_parquet, schema, flush_freq=1000):
    """Writer process, writes minhash to parquet file and quality signals to json.gz respecting format of orginal redpajama dataset"""

    output_folder = os.path.dirname(output_parquet)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    schema_parquet = schema
    _writer = ParquetBatchWriter(output_parquet, schema_parquet)
    output_jsonl_gz = output_parquet.split(".")[0] + ".signals.jsonl.gz"
    _signal_writer = gzip.open(output_jsonl_gz, "wt")
    signals_list = []
    while True:
        value = writer_queue.get()
        if value is None:
            time.sleep(1)
            continue
        else:
            action, record, signals = value
        if action == "STOP":
            _writer.close()
            for signal in signals_list:
                _signal_writer.write(signal)
            signals_list = []
            _signal_writer.close()
            return
        elif action == "Close":
            pass
        else:

            _writer.update_batch(record)
            signals_list.append(json.dumps(signals) + "\n")
            if _writer.counter % flush_freq == 0:
                _writer.write_batch()
                for signal in signals_list:
                    _signal_writer.write(signal)
                signals_list = []

def worker(
    processing_queue,
    writer_queue,
    fasttext_model_path,
    cc_net_model_path,
    text_chunk_strategy,
    languages,
):
    if isinstance(languages, str):
        languages = [languages]
    assert isinstance(languages, list) and len(languages) > 0, "Languages must be a non-empty list"
    fasttext_model = FastLanguageIdentification(fasttext_model_path)
    perplexity_models =dict((language, Perplexity(cc_net_model_path, language)) for language in languages)
    default_language = languages[0]
    rp2_callables = []
    # rp2_callables += register_content_callables(
    #     language=language,
    #     bad_urls_dir=domain_filter_path,
    #     bad_words_dir=words_filter_path,
    # )
    rp2_callables += register_lines_callables()
    rp2_callables += register_natural_language_callables()
    rp2_callables += register_repetitions_callables()
    # min_hasher = MinHash(
    #     similarity_thresholds=minhash_similarities,
    #     ngram_size=minhash_ngram_size,
    #     num_permutations=minhash_num_permutations,
    #     seed=seed,
    # )

    if isinstance(text_chunk_strategy, int):
        get_text_chunks = lambda x: chunk_text_simple(x, text_chunk_strategy)
    elif text_chunk_strategy == "smart":
        get_text_chunks = chunk_text_smart
    elif text_chunk_strategy == "simple":
        get_text_chunks = chunk_text_simple
    else:
        raise ValueError(f"Text chunk strategy '{text_chunk_strategy}' ({type(text_chunk_strategy)}) not supported")

    while True:
        value = processing_queue.get()
        if value is None:
            time.sleep(1)
            continue
        else:
            action, row, idx_row = value

        if action == "STOP":
            return
        elif action == "Close":
            writer_queue.put(("Close", None, None))
        else:
            if "text" in row:
                text_field = "text"
            elif "complete_text" in row:
                text_field = "complete_text"
            else:
                raise ValueError("No text field found in the row")

            # Chunk the text
            text = row[text_field]
            chunks, _ = get_text_chunks(text)
            all_text_splits = [text[start:end] for (start, end) in chunks]
            row["chunk_start"] = [start for (start, _) in chunks]
            row["chunk_end"] = [end for (_, end) in chunks]

            # Run language identification
            results = [
                fasttext_model.predict_lang(chunk)
                for chunk in all_text_splits
            ]
            row["ccnet_language_score"] = [round(r[1], 4) for r in results]
            row["fasttext_language"] = [r[0] for r in results]

            default_perplexity_model = perplexity_models[default_language]
            if len(perplexity_models) == 1:
                perplexities = [
                    default_perplexity_model(chunk)
                    for chunk in all_text_splits
                ]
            else:
                perplexities = [
                    perplexity_models.get(language, default_perplexity_model)(chunk)
                    for chunk, language in zip(all_text_splits, row["fasttext_language"])
                ]

            row["ccnet_avg_log_prob"] = [logprob for logprob, _ in perplexities]
            row["ccnet_length"] = [length for _, length in perplexities]


            documents = [
                Document(
                    content=chunk,
                    domain="domain",
                    precompute_ngrams=True,
                    precompute_hash_features=False,
                    dsir_buckets=1,
                )
                for chunk in all_text_splits
            ]
            signals = {}
            for func in rp2_callables:
                signals[func.field_name] = [func(doc) for doc in documents]
            signals["idx_row"] = idx_row
            row["idx_row"] = idx_row
            writer_queue.put(("Process", row, signals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir-input",
        type=str,
        help="Path to the parquet file to process",
    )
    parser.add_argument(
        "--final-dir-output",
        type=str,
        help="Path to final parquet directory",
        default="",
    )
    parser.add_argument(
        "--regex-pattern",
        type=str,
        default=".*.parquet",
        help="Regex pattern to match files of interest",
    )
    parser.add_argument(
        "--dir-output",
        type=str,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--path_fasttext_model",
        type=str,
        help="Path to the fasttext model lid.176.bin file",
    )
    parser.add_argument(
        "--dir_perplexity_models",
        type=str,
        help="Root directory to the perplexity models",
    )
    parser.add_argument(
        "--text_chunk_strategy",
        type=str,
        help="Text chunk strategy ('smart', 'simple' or int)",
        default="simple",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        help="Number of processes",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language(s) of the dataset ('en', 'fr', ...). If it can be multi-lingual, the first language is the default when the detected languages is not in the list",
        nargs="+",
    )
    parser.add_argument(
        "--flush-freq",
        type=int,
        default=1000,
        help="Frequency of flushing the writer queue",
    )

    print("Parsing args", flush=True)

    args = parser.parse_args()

    try:
        args.text_chunk_strategy = int(args.text_chunk_strategy)
    except: pass

    dataset_processor = DatasetProcessor(
        path_fasttext_model=args.path_fasttext_model,
        path_perplexity_models=args.dir_perplexity_models,
        text_chunk_strategy=args.text_chunk_strategy,
        n_processes=args.n_processes,
        flush_freq=args.flush_freq,
    )

    print("Processing dataset", flush=True)

    dataset_processor.process_dataset(
        dir_input=args.dir_input,
        dir_output=args.dir_output,
        final_dir_output=args.final_dir_output,
        language=args.language,
    )
