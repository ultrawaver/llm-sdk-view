from llm import hookimpl

from .cli import native_chat


@hookimpl
def register_commands(cli):
    cli.add_command(native_chat)
