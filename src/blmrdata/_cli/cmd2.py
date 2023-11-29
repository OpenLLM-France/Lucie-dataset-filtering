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


@click.group()
def cli_group2():
    pass


@cli_group2.command(
    name="cmd2", help="DESCRIPTION OF COMMAND", context_settings={"show_default": True}
)
@click.option(
    "--input_folder_ner",
    "-i",
    type=click.Path(exists=True),
)
@click.option(
    "--output_folder_sentences",
    "-o",
    type=click.Path(exists=False),
)
def cli_group2_cmd2(input_folder_ner, output_folder_sentences):
    """DESCRIPTION OF COMMAND"""
    print(
        "cmd2 executed with input_folder_ner: {} and output_folder_sentences: {}".format(
            input_folder_ner, output_folder_sentences
        )
    )
