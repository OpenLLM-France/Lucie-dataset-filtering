import sys
import click
from loguru import logger
from blmrdata._cli.cmd_group1.group1 import cli_group1
from blmrdata._cli.cmd2 import cli_group2_cmd2


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


cli.add_command(cli_group1)
cli.add_command(cli_group2_cmd2)


if __name__ == "__main__":
    cli()
