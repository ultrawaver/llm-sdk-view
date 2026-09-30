import click


@click.command(name="native-chat")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--reload", is_flag=True, help="Reload when Python files change")
def native_chat(host: str, port: int, reload: bool) -> None:
    "Start the local Native API Chat web application."
    # Imported here so that loading the plugin for any `llm` command does not
    # pay the cost of importing an ASGI server that will not be used.
    import uvicorn

    click.echo(f"Native API Chat: http://{host}:{port}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        click.echo("Warning: the server is exposed beyond the local machine", err=True)
    uvicorn.run("native_api_chat.app:app", host=host, port=port, reload=reload)


def standalone() -> None:
    native_chat.main(standalone_mode=True)
