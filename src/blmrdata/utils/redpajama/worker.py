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
from redpajama.core.document import Document
from redpajama.dedupe.minhash import MinHash

from redpajama.core.worker import _ccnet_bucket_to_int, ParquetBatchWriter
import hashlib
import pyarrow as pa
import argparse
from ast import literal_eval
import time
import re

_BYTE_ORDER = sys.byteorder


class DatasetProcessor(object):
    def __init__(
        self,
        path_fasttext_model: Path,
        path_perplexity_models: Path,
        path_words_filter: Path,
        path_domain_filter: Path,
        path_cut_offs: Path,
        mapping_fields: str,
        default_fields: str,
        fields_to_keep: str,
        n_processes: int = 1,
    ):
        self.path_fasttext_model = path_fasttext_model
        self.path_perplexity_models = path_perplexity_models
        self.path_words_filter = path_words_filter
        self.path_domain_filter = path_domain_filter
        self.path_cut_offs = path_cut_offs
        self.n_processes = n_processes  # Num processing workers
        self.mapping_fields = literal_eval(mapping_fields)
        self.default_fields = literal_eval(default_fields)
        self.fields_to_keep = literal_eval(fields_to_keep)

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
        return files_of_interest

    def process_dataset(self, path_dataset: Path, path_output: Path, language: str):
        """Process a dataset and output a new one with the perplexity score"""

        assert language in ["fr", "en"], f"Language {language} not supported"

        files_of_interest = sorted(DatasetProcessor.get_files_of_interest(path_dataset))
        print(f"Processing {len(files_of_interest)} files")
        manager = mp.Manager()
        writer_queue = manager.Queue()
        # Start writer process
        minhash_similarities = [1.0, 0.9, 0.8, 0.7]
        writer_process = mp.Process(
            target=writer, args=(writer_queue, path_output, minhash_similarities)
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
                    self.path_words_filter,
                    self.path_domain_filter,
                    self.path_cut_offs,
                    language,
                    self.mapping_fields,
                    self.default_fields,
                    self.fields_to_keep,
                    minhash_similarities,
                ),
            )
            for i in range(self.n_processes)
        ]
        for p in worker_processes:
            p.start()

        gc.collect()

        for file in tqdm(files_of_interest):
            uri_id = file.split(path_dataset)[1]
            with xopen(file, "rt") as fin:
                for idx_line, line in tqdm(enumerate(fin)):
                    doc = json.loads(line)
                    #argmin to get the queue with the smallest size
                    smallest_queue_index = min(range(len(processing_queues)), key=lambda i: processing_queues[i].qsize())
                    processing_queues[smallest_queue_index].put(
                        ("Process", doc, uri_id, idx_line)
                    )
                processing_queues[smallest_queue_index].put(("Close", None, uri_id, idx_line))
                while processing_queues[smallest_queue_index].qsize() > 1000 and writer_queue.qsize() > 10000:
                    time.sleep(1)
        for q in processing_queues:
            q.put(("STOP", None, None, None))
        writer_queue.put(("STOP", None, None))

        for p in worker_processes:
            p.join()
        writer_process.join()
        return


def open_jsonl_gz(uri_id, output_folder):
    path_gz = os.path.join(output_folder, "data", uri_id.split(".")[0].strip("/") + ".jsonl.gz")
    return gzip.open(
        path_gz,
        "wt",
    )


def open_parquet(uri_id, output_folder, minhash_schema):
    return ParquetBatchWriter(
        output_fp=os.path.join(
            output_folder, "minhashes", uri_id.split(".")[0].strip("/") + ".parquet"
        ),
        schema=minhash_schema,
    )


def writer(writer_queue, output_folder, minhash_similarities=[1.0, 0.9, 0.8, 0.7]):
    os.makedirs(os.path.join(output_folder, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "minhashes"), exist_ok=True)
    opened_parquet = {}
    opened_jsonl_gz = {}
    minhash_schema = pa.schema(
        [
            ("shard_id", pa.string()),
            ("id", pa.string()),
            ("id_int", pa.uint64()),
            *[
                ("signature_sim{s}".format(s=s), pa.list_(pa.binary()))
                for s in minhash_similarities
            ],
        ]
    )
    counter = 0
    while True:
        value = writer_queue.get()
        if value is None:
            time.sleep(1)
            continue
        else:
            action, record, minhashes = value

        if action == "STOP":
            for uri_id in opened_parquet:
                opened_parquet[uri_id].write_batch()
                opened_parquet[uri_id].close()
            for uri_id in opened_jsonl_gz:
                opened_jsonl_gz[uri_id].close()
            return
        elif action == "Close":
            uri_id = record
            if uri_id in opened_parquet:
                opened_parquet[uri_id].write_batch()
                opened_parquet[uri_id].close()
                counter = 0
                del opened_parquet[uri_id]
            if uri_id in opened_jsonl_gz:
                opened_jsonl_gz[uri_id].close()
                del opened_jsonl_gz[uri_id]
        else:
            uri_id = record["uri_id"]
            if uri_id not in opened_parquet:
                opened_parquet[uri_id] = open_parquet(uri_id, output_folder, minhash_schema)
            if uri_id not in opened_jsonl_gz:
                opened_jsonl_gz[uri_id] = open_jsonl_gz(uri_id, output_folder)
            opened_parquet[uri_id].update_batch(
                obj={
                    "shard_id": uri_id,
                    "id_int": record["doc_id_int"],
                    "id": record["doc_id"],
                    **minhashes,
                }
            )
            if counter % 1000 == 0:
                opened_parquet[uri_id].write_batch()
                counter = 0
            opened_jsonl_gz[uri_id].write(json.dumps(record) + "\n")


def compute_sha1_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def worker(
    processing_queue,
    writer_queue,
    fasttext_model_path,
    cc_net_model_path,
    words_filter_path,
    domain_filter_path,
    cut_offs_path,
    language,
    mapping_fields,
    default_fields,
    fields_to_keep,
    minhash_similarities=[1.0, 0.9, 0.8, 0.7],
    minhash_ngram_size=13,
    minhash_num_permutations=128,
    seed=42,
):
    mandatory_fields = ["raw_content", "source_domain"]
    for m in mandatory_fields:
        if (
            m not in default_fields
            and m not in mapping_fields
            and m not in fields_to_keep
        ):
            assert f"Missing mandatory field {m}"
    fasttext_model = FastLanguageIdentification(fasttext_model_path)
    perplexity_model = Perplexity(cc_net_model_path, language)
    cut_offs = json.load(open(cut_offs_path))[language]
    rp2_callables = []
    rp2_callables += register_content_callables(
        language=language,
        bad_urls_dir=domain_filter_path,
        bad_words_dir=words_filter_path,
    )
    rp2_callables += register_lines_callables()
    rp2_callables += register_natural_language_callables()
    rp2_callables += register_repetitions_callables()
    min_hasher = MinHash(
        similarity_thresholds=minhash_similarities,
        ngram_size=minhash_ngram_size,
        num_permutations=minhash_num_permutations,
        seed=seed,
    )
    while True:
        value = processing_queue.get()
        if value is None:
            time.sleep(1)
            continue
        else:
            action, doc, uri_id, idx = value

        if action == "STOP":
            return
        elif action == "Close":
            writer_queue.put(("Close", uri_id, None))
        else:
            final_record = {"quality_signals": {}, "other": {}, "meta": {}}
            final_record["raw_content"] = doc[mapping_fields["raw_content"]]
            if "source_domain" in mapping_fields:
                final_record["meta"]["source_domain"] = doc[
                    mapping_fields["source_domain"]
                ]
            else:
                final_record["meta"]["source_domain"] = default_fields["source_domain"]

            if "url" in mapping_fields:
                final_record["meta"]["url"] = doc[mapping_fields["url"]]
            else:
                final_record["meta"]["url"] = None

            if "date_download" in mapping_fields:
                final_record["meta"]["date_download"] = doc[
                    mapping_fields["date_download"]
                ]
            else:
                final_record["meta"]["date_download"] = None

            final_record["meta"]["digest"] = compute_sha1_hash(
                final_record["raw_content"]
            )

            final_record["meta"]["language"] = language

            # Compute quality signals

            document_length = len(final_record["raw_content"])
            final_record["quality_signals"]["ccnet_length"] = [
                [0, document_length, document_length]
            ]
            final_record["quality_signals"]["ccnet_nlines"] = [
                [
                    0,
                    document_length,
                    len(final_record["raw_content"].split("\n")),
                ]
            ]
            detected_language, language_score = fasttext_model.predict_lang(
                final_record["raw_content"]
            )
            if detected_language != language:
                continue
            final_record["quality_signals"]["ccnet_language_score"] = [
                [0, document_length, round(language_score, 4)]
            ]

            final_record["quality_signals"]["ccnet_perplexity"] = [
                [0, document_length, perplexity_model(final_record["raw_content"])]
            ]

            if (
                final_record["quality_signals"]["ccnet_perplexity"][0][-1]
                > cut_offs["tail_middle"]
            ):
                final_record["quality_signals"]["bucket"] = "tail"
            elif (
                final_record["quality_signals"]["ccnet_perplexity"][0][-1]
                > cut_offs["middle_head"]
            ):
                final_record["quality_signals"]["bucket"] = "middle"
            else:
                final_record["quality_signals"]["bucket"] = "head"

            dsir_buckets = _ccnet_bucket_to_int(
                final_record["quality_signals"]["bucket"]
            )

            final_record["quality_signals"]["ccnet_bucket"] = [
                [0, document_length, dsir_buckets]
            ]

            document = Document(
                content=final_record["raw_content"],
                domain=final_record["meta"]["source_domain"],
                precompute_ngrams=True,
                precompute_hash_features=True,
                dsir_buckets=int(dsir_buckets) + 1,
            )

            for func in rp2_callables:
                final_record["quality_signals"][func.field_name] = func(document)

            minhashes = min_hasher.compute_banded_signatures(
                tokens=document.normalized_words
            )

            doc_id = f"{uri_id}/{idx}"
            doc_id_int = int.from_bytes(
                hashlib.sha1(doc_id.encode("utf-8")).digest()[:8],  # take 8 bytes
                byteorder=_BYTE_ORDER,
                signed=False,
            )
            final_record["uri_id"] = uri_id
            final_record["idx"] = idx
            final_record["doc_id"] = doc_id
            final_record["doc_id_int"] = doc_id_int
            for field in fields_to_keep:
                final_record["other"][field] = doc[field]
            writer_queue.put(("Process", final_record, minhashes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to the dataset to process",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--path_fasttext_model",
        type=str,
        help="Path to the fasttext model",
    )
    parser.add_argument(
        "--path_perplexity_models",
        type=str,
        help="Path to the perplexity models",
    )
    parser.add_argument(
        "--path_words_filter",
        type=str,
        help="Path to the words filter",
    )
    parser.add_argument(
        "--path_domain_filter",
        type=str,
        help="Path to the domain filter",
    )
    parser.add_argument(
        "--path_cut_offs",
        type=str,
        help="Path to the cut offs",
    )
    parser.add_argument(
        "--mapping_fields",
        type=str,
        help="Mapping fields",
    )
    parser.add_argument(
        "--default_fields",
        type=str,
        help="Default fields",
    )
    parser.add_argument(
        "--fields_to_keep",
        type=str,
        help="Fields to keep",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        help="Number of processes",
    )
    args = parser.parse_args()

    dataset_processor = DatasetProcessor(
        path_fasttext_model=args.path_fasttext_model,
        path_perplexity_models=args.path_perplexity_models,
        path_words_filter=args.path_words_filter,
        path_domain_filter=args.path_domain_filter,
        path_cut_offs=args.path_cut_offs,
        mapping_fields=args.mapping_fields,
        default_fields=args.default_fields,
        fields_to_keep=args.fields_to_keep,
        n_processes=args.n_processes,
    )

    dataset_processor.process_dataset(
        path_dataset=args.path_dataset,
        path_output=args.path_output,
        language=args.language,
    )
