import sys
import click
from loguru import logger
from blmrdata._cli.utils.download import cli_download


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


cli.add_command(cli_download)

if __name__ == "__main__":
    cli()
