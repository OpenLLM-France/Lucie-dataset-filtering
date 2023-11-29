# -*- coding: utf-8 -*-
""" Example of a command line interface for a group of commands

@Author: Evan Dufraisse
@Date: Tue May 23 2023
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2023 CEA - LASTI
"""
import click
from loguru import logger

from tqdm.auto import tqdm


@click.group(name="download")
def cli_download():
    """Download resources to compile the dataset"""
    pass


@cli_download.command(
    name="perplexity-models", help="Download perplexity models for supplied languages", context_settings={"show_default": True}
)
@click.option(
    "--languages",
    type=click.Choice(["en", "fr", "de", "es", "it", "nl", "pt", "ru", "sv", "tr"]),
    default=("en","fr"),
    multiple=True,
)
@click.option(
    "--output-folder",
    type=click.Path(exists=False),
    default="data/perplexity_models",
)
def cli_download_perplexity(languages, output_folder):
    """Download perplexity models for supplied languages"""
    from blmrdata.utils.ccnet.perplexity import download_perplexity_models

    logger.info(f"Downloading perplexity models for languages: {languages}")
    download_perplexity_models(languages, output_folder)