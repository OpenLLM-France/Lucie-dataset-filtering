# -*- coding: utf-8 -*-
"""Download resources to compile the dataset

@Date: Tue May 23 2023
@Contact: contact@openllm-france.fr
@License: MIT License
"""

from pathlib import Path
from typing import List

import click
from loguru import logger

from blmrdata.utils.ccnet.perplexity import download_perplexity_models


@click.group(name="download")
def cli_download():
    """Download resources to compile the dataset"""
    pass


@cli_download.command(
    name="perplexity-models",
    help="Download perplexity models for supplied languages",
    context_settings={"show_default": True},
)
@click.option(
    "--languages",
    type=click.Choice(["en", "fr", "de", "es", "it", "nl", "pt", "ru", "sv", "tr"]),
    default=("en", "fr"),
    multiple=True,
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False),
    default="data/perplexity_models",
)
def cli_download_perplexity(
    languages: List[str],
    output_folder: Path,
) -> None:
    """Download perplexity models for supplied languages"""

    logger.info(f"Downloading perplexity models for languages: {languages}")
    download_perplexity_models(languages, output_folder)
