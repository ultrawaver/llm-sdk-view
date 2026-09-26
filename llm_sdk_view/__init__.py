from llm import hookimpl

from .cli import sdk_view


@hookimpl
def register_commands(cli):
    cli.add_command(sdk_view)
