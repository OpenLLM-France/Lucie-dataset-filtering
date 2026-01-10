# -*- coding: utf-8 -*-
"""Dowload and use CCNET models for perplexity estimation

@Date: Wed Nov 29 2023
@Contact: contact@openllm-france.fr
@License: MIT License
"""

import os
from pathlib import Path
from typing import List, Tuple

import kenlm
from loguru import logger
from tqdm.auto import tqdm

from blmrdata.utils import download
from blmrdata.utils.ccnet.ccnet_utilities import text_normalizer
from blmrdata.utils.ccnet.ccnet_utilities.sentencepiece import SentencePiece


def download_perplexity_models(languages: List[str], output_folder: str) -> None:
    """Download perplexity models for the specified languages.

    Args:
        languages (List[str]): List of language codes to download (e.g., ['fr', 'en'])
        output_folder (str): Destination folder for downloaded models

    Raises:
        requests.RequestException: If an error occurs during download
    """
    os.makedirs(output_folder, exist_ok=True)

    for lang in tqdm(languages, desc="Downloading models"):
        download_language_models(lang, output_folder)


def download_language_models(language: str, output_folder: str) -> Tuple[Path, Path]:
    """Download ARPA and SentencePiece models for a given language.

    Args:
        language (str): Language code (e.g., 'fr', 'en')
        output_folder (str): destination folder

    Returns:
        Tuple[Path, Path]: A tuple containing the paths to the downloaded models
    """
    BASE_MODEL_URL = "http://dl.fbaipublicfiles.com/cc_net/lm"
    MODEL_EXTENSIONS = {"arpa": ".arpa.bin", "sentencepiece": ".sp.model"}

    logger.info(f"Downloading model for {language}")

    arpa_url = f"{BASE_MODEL_URL}/{language}{MODEL_EXTENSIONS['arpa']}"
    arpa_path = Path(
        os.path.join(output_folder, f"{language}{MODEL_EXTENSIONS['arpa']}")
    )
    download(arpa_url, str(arpa_path))

    sp_url = f"{BASE_MODEL_URL}/{language}{MODEL_EXTENSIONS['sentencepiece']}"
    sp_path = Path(
        os.path.join(output_folder, f"{language}{MODEL_EXTENSIONS['sentencepiece']}")
    )
    download(sp_url, str(sp_path))

    return arpa_path, sp_path


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
        self.lm = kenlm.Model(
            f"{model_download_folder}/{language}.arpa.bin", kenlm.Config()
        )

    def pp(self, text: str, **kwargs):
        """Compute the perplexity of a text"""
        (avg_neg_log_likelihood, length) = self.__call__(text, **kwargs)
        return 10.0**-avg_neg_log_likelihood

    def __call__(self, text: str, normalize=False) -> dict:
        """Compute the average negative-log likelihood of a text

        Args:
            text (str): Text to compute the perplexity of
            normalize (bool): Normalize the text before computing the perplexity

        Returns:
            a tuple (average negative log proba, length)
        """
        tokenized = self.sp(text)["tokenized"]
        lines = tokenized.split("\n")
        doc_log_score = 0
        doc_length = 0
        for line in lines:
            if normalize:
                line = text_normalizer.normalize(line)
            log_score = self.lm.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length

        return (-doc_log_score / doc_length, doc_length)
