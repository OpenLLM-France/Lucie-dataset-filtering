# -*- coding: utf-8 -*-
"""Dowload and use CCNET models for perplexity estimation

@Date: Fri Jan 9 2026
@Contact: contact@openllm-france.fr
@License: MIT License
"""

import os
from pathlib import Path
from typing import IO, List, Tuple

import kenlm
import requests
from loguru import logger
from tqdm.auto import tqdm

from blmrdata.utils.ccnet.ccnet_utilities import text_normalizer
from blmrdata.utils.ccnet.ccnet_utilities.sentencepiece import SentencePiece

def download(url: str, fname: str) -> IO[bytes]:
    """
    Download a file from the given URL and save it to the given filename.

    Args:
        url (str): The URL of the file to download.
        fname (str): The filename to save the downloaded file as.

    Returns:
        IO[bytes]: The file object of the downloaded file.
    """
    session = requests.Session()
    with session.get(url, stream=True) as resp:
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024 * 16):
                size = file.write(data)
                bar.update(size)
    return file

def download_perplexity_models(languages: List[str], output_folder: Path) -> None:
    """Download perplexity models for the specified languages.

    Args:
        languages (List[str]): List of language codes to download (e.g., ['fr', 'en'])
        output_folder (Path): Destination folder for downloaded models

    Raises:
        requests.RequestException: If an error occurs during download
    """
    os.makedirs(output_folder, exist_ok=True)

    for lang in tqdm(languages, desc="Downloading models"):
        download_language_models(lang, output_folder)


def download_language_models(language: str, output_folder: Path) -> Tuple[Path, Path]:
    """Download ARPA and SentencePiece models for a given language.

    Args:
        language (str): Language code (e.g., 'fr', 'en')
        output_folder (Path): destination folder

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
    if not arpa_path.exists():
        download(arpa_url, str(arpa_path))

    sp_url = f"{BASE_MODEL_URL}/{language}{MODEL_EXTENSIONS['sentencepiece']}"
    sp_path = Path(
        os.path.join(output_folder, f"{language}{MODEL_EXTENSIONS['sentencepiece']}")
    )
    if not sp_path.exists():
        download(sp_url, str(sp_path))

    return arpa_path, sp_path


class Perplexity:
    """Class to compute perplexity of a document"""

    def __init__(self, model_download_folder: Path, language: str) -> None:
        """Initialize the perplexity calculator.

        Args:
            model_download_folder: Path to the folder containing the downloaded models
            language: str, language code of the model to use (e.g., 'fr', 'en')

        Raises:
            FileNotFoundError: If the model files do not exist
        """
        self.model_folder = Path(model_download_folder)
        self.language = language

        # Initialize SentencePiece tokenizer
        sp_model_path = self.model_folder / f"{language}.sp.model"
        self.sp = SentencePiece(
            model=sp_model_path,
            field="text",
            output_field="tokenized",
            normalize=True,
        )

        # Initialize KenLM language model
        lm_model_path = self.model_folder / f"{language}.arpa.bin"
        self.lm = kenlm.Model(str(lm_model_path))

    def pp(self, text: str, normalize: bool = False) -> float:
        """Compute the perplexity of a text

        Args:
            text (str): Text to compute the perplexity of
            normalize (bool): Normalize the text before computing the perplexity

        Returns:
            float: Perplexity value (10^(-average_log_prob))
        """
        avg_neg_log_likelihood, _ = self.__call__(text, normalize=normalize)
        return 10.0**-avg_neg_log_likelihood

    def __call__(self, text: str, normalize: bool = False) -> Tuple[float, int]:
        """Compute the average negative-log likelihood of a text

        Args:
            text (str): Text to compute the perplexity of
            normalize (bool): Normalize the text before computing the perplexity

        Returns:
            Tuple[float, int]: A tuple containing the average negative log probability and the length of the text
        """
        tokenized: str = self.sp(text)["tokenized"]
        lines: List[str] = tokenized.split("\n")
        doc_log_score: float = 0
        doc_length: int = 0

        for line in lines:
            if not line.strip():  # Skip empty lines
                continue

            if normalize:
                line: str = text_normalizer.normalize(line)

            log_score: float = self.lm.score(line)
            length: int = len(line.split()) + 1  # +1 for end-of-line token
            doc_log_score += log_score
            doc_length += length

        if doc_length == 0:
            return (0.0, 0)

        return (-doc_log_score / doc_length, doc_length)
