# -*- coding: utf-8 -*-
""" Dowload and use CCNET models for perplexity estimation

@Author: Evan Dufraisse
@Date: Wed Nov 29 2023
@Contact: e[dot]dufraisse[at]gmail[dot]com
@License: MIT License
"""
import os
from loguru import logger
import typing
from tqdm.auto import tqdm
from pathlib import Path
from blmrdata.utils.ccnet.ccnet_utilities import text_normalizer
from blmrdata.utils import download

def download_perplexity_models(languages: typing.List , output_folder: str):
    """Download perplexity models for supplied languages

    Args:
        languages (typing.List): List of languages to download
        output_folder (str): Folder to download the models
    """
    from tqdm.auto import tqdm
    import requests
    import tarfile
    import shutil

    os.makedirs(output_folder, exist_ok=True)

    for lang in tqdm(languages, desc="Downloading models"):
        url_model = f"http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin"
        url_sp = f"http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).sp.model"

        logger.info(f"Downloading model for {lang}")
        download(url_model, os.path.join(output_folder, f"{lang}.arpa.bin"))
        download(url_sp, os.path.join(output_folder, f"{lang}.sp.model"))

import kenlm
from blmrdata.utils.ccnet.ccnet_utilities.sentencepiece import SentencePiece
class Perplexity:
    """Class to compute perplexity of a document"""

    def __init__(self, model_download_folder: str, language: str):
        """Initialize the class

        Args:
            model_folder (str): Folder containing the models
            languages (typing.List): List of languages to use
        """
        self.model_folder = model_download_folder
        self.language = language
        self.sp = SentencePiece(
            model=Path(f"{model_download_folder}/{language}.sp.model"),
            field="text",
            output_field="tokenized",
            normalize=True,
        )
        self.lm = kenlm.Model(f"{model_download_folder}/{language}.arpa.bin", kenlm.Config())

    @staticmethod
    def pp(log_score, length):
        return 10.0 ** (-log_score / length)


    def __call__(self, text: str, normalize=False) -> dict:
        """Compute the perplexity of a text

        Args:
            text (str): Text to compute the perplexity of

        Returns:
            dict: Dict containing the text and the perplexity
        """
        tokenized = self.sp(text)["tokenized"]
        lines = tokenized.split("\n")
        for line in lines:
            if normalize:
                line = text_normalizer.normalize(line)
            log_score = self.lm.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
            perplexity = round(self.pp(doc_log_score, doc_length), 1)

        return perplexity
        
