# -*- coding: utf-8 -*-
""" Utils functions for redpajama metrics

@Author: Evan Dufraisse
@Date: Mon Dec 18 2023
@Contact: e[dot]dufraisse[at]gmail[dot]com
@License: MIT License
"""

import os
import json
from tqdm.auto import tqdm


def create_blacklist_ut1_domain_to_category(
    path_ut1,
    list_to_filter=[
        "adult",
        "phishing",
        "dating",
        "gambling",
        "filehosting",
        "ddos",
        "agressif",
        "chat",
        "mixed_adult",
        "arjel",
    ],
):
    """
    Create a blacklist of domains to filter based on UT1 categories in the RefinedWeb dataset paper
    Link to UT1: https://dsi.ut-capitole.fr/blacklists/
    """
    all_folders = [os.path.join(path_ut1, f) for f in list_to_filter]
    domain_to_category_id = {}
    for folder in tqdm(all_folders):
        with open(os.path.join(folder, "domains")) as fin:
            for line in fin:
                domain_to_category_id[line.strip()] = os.path.basename(folder)
    with open(os.path.join(path_ut1, "domain_to_category_id.json"), "w") as fout:
        json.dump(domain_to_category_id, fout)


class BatchWriter:
    def __init__(self, writer, max_size=1024):
        self.writer = writer
        self.max_size = max_size
        self.batch = []
        self.batch_id = 0

    def write(self, item):
        self.batch.append(item)
        if len(self.batch) >= self.max_size:
            self.flush()

    def flush(self):
        for item in self.batch:
            self.writer.write(item)
        self.batch = []
        self.batch_id += 1

    def __len__(self):
        return len(self.batch)

    def close(self):
        self.flush()
        self.writer.close()
