"""History lives in llm's own database, not in a second one.

Everything a turn is made of mostly already has a home upstream: the message
chain, the turn's timings, usage and provider response all go through
``llm.logs.LogStore``, which is llm's documented read/write API for its SQLite
schema and the same object its own CLI builds on. Nothing here calls the CLI,
nothing here reimplements its writes, and nothing here builds a parallel
conversation or response table.

What llm does not store - the request this app actually built, and the code
the right pane rendered from it - goes in one narrow sidecar table keyed by
the upstream turn id. It adds a table for its own data instead of inventing
meaning for someone else's columns.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import llm
from llm.logs import LogStore
from llm.migrations import migrate
from sqlite_utils import Database

from .records import SOURCE, ResponseView, TurnRecord, conversation_name

# A test needs a database of its own; nothing else should ever set this.
DATABASE_ENV = "LLM_SDK_VIEW_LOGS_DB"

SIDECAR_TABLE = "llm_sdk_view_turns"


class PersistenceError(RuntimeError):
    """A completed turn that could not be written.

    The call succeeded and was billed, so this must surface as itself rather
    than as a graceful "saved" - the difference between the two is history.
    """


def database_path() -> Path:
    """Where llm keeps its logs; the file this app adds rows to."""
    override = os.environ.get(DATABASE_ENV)
    if override:
        return Path(override)
    # llm.cli owns the canonical path; user_dir() plus llm's own filename is
    # the fallback if that helper ever moves.
    try:
        from llm.cli import logs_db_path

        return Path(logs_db_path())
    except Exception:  # pragma: no cover - only if llm drops that helper
        return Path(llm.user_dir()) / "logs.db"


def connect(path: Path | None = None) -> Database:
    """Open llm's database with its migrations and our sidecar applied."""
    target = path or database_path()
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    db = Database(sqlite3.connect(str(target)))
    db.execute("PRAGMA foreign_keys = ON")
    # Upstream migrations first: the sidecar points at their tables.
    migrate(db)
    _ensure_sidecar(db)
    db.conn.commit()
    return db


def _ensure_sidecar(db: Database) -> None:
    if db[SIDECAR_TABLE].exists():
        return
    db[SIDECAR_TABLE].create(
        {
            "turn_id": str,
            "thread_id": str,
            "user_input": str,
            "effective_options": str,
            "request_kwargs": str,
            "rendered_code": str,
            "response_json": str,
            "context_json": str,
            "source": str,
            "datetime_utc": str,
        },
        pk="turn_id",
        foreign_keys=(("turn_id", "turns", "id"),),
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# --- writing ---------------------------------------------------------------


def save_turn(response: Any, record: TurnRecord, db: Database | None = None) -> str:
    """Record one finished turn atomically. Returns the upstream turn id.

    One transaction for both writes, so a sidecar that cannot be written
    leaves no orphaned upstream turn, and the caller hears about it instead
    of being told the conversation was saved.
    """
    database = db or connect()
    store = LogStore(database)
    # The thread is created here, with our own name, so the first user message
    # sets the title: no model call, and no unexpected ellipsis in the middle.
    try:
        # Everything inside one transaction, including creating the thread: a
        # failed save must not leave a conversation pointing at nothing.
        with database.atomic():
            store.ensure_thread(
                record.conversation_id, name=conversation_name(record.user_input)
            )
            turn_id = store.log(response, thread_id=record.conversation_id)
            # One clock: whatever instant llm stamped the turn with is the
            # instant every surface reports, rather than a second timestamp
            # taken a few milliseconds earlier on the way out.
            when = database["turns"].get(turn_id)["datetime_utc"]
            database[SIDECAR_TABLE].insert(
                {
                    "turn_id": turn_id,
                    "thread_id": record.conversation_id,
                    "user_input": record.user_input,
                    "effective_options": _json(record.options),
                    "request_kwargs": _json(record.request_kwargs),
                    "rendered_code": record.rendered_code,
                    "response_json": _json(record.response.as_dict()),
                    "context_json": _json(record.context),
                    "source": SOURCE,
                    "datetime_utc": when or record.timestamp,
                },
                replace=True,
            )
    except Exception as error:  # noqa: BLE001 - re-raised with context below
        raise PersistenceError(
            f"the turn completed but could not be saved: {error}"
        ) from error
    return turn_id


# --- reading ----------------------------------------------------------------


def list_conversations(db: Database | None = None, limit: int = 50) -> list[dict]:
    """Conversations newest first, each with what a list has to show."""
    database = db or connect()
    rows = list(
        database.query(
            """
            select threads.id as id,
                   threads.name as name,
                   max(turns.datetime_utc) as last_turn,
                   count(turns.id) as turns
            from threads
            left join turns on turns.thread_id = threads.id
            group by threads.id
            having turns > 0
            order by last_turn desc
            limit ?
            """,
            [limit],
        )
    )
    return [
        {
            "id": row["id"],
            "name": row["name"] or "Untitled conversation",
            "last_turn": row["last_turn"],
            "turns": row["turns"],
        }
        for row in rows
    ]


def load_conversation(thread_id: str, db: Database | None = None) -> dict | None:
    """One conversation with every turn's request and response.

    Returns None for an unknown id, so a stale entry in the list cannot pass
    itself off as an empty conversation.
    """
    database = db or connect()
    thread = list(database["threads"].rows_where("id = ?", [thread_id]))
    if not thread:
        return None
    turns = []
    for row in database.query(
        """
        select turns.id as turn_id,
               turns.model as model,
               turns.datetime_utc as datetime_utc,
               side.user_input as user_input,
               side.effective_options as effective_options,
               side.request_kwargs as request_kwargs,
               side.rendered_code as rendered_code,
               side.response_json as response_json,
               side.context_json as context_json
        from turns
        left join {sidecar} side on side.turn_id = turns.id
        where turns.thread_id = ?
        order by turns.datetime_utc, turns.id
        """.replace("{sidecar}", SIDECAR_TABLE),
        [thread_id],
    ):
        turns.append(
            TurnRecord(
                conversation_id=thread_id,
                turn_id=row["turn_id"],
                user_input=row["user_input"] or "",
                options=json.loads(row["effective_options"] or "{}"),
                request_kwargs=json.loads(row["request_kwargs"] or "{}"),
                rendered_code=row["rendered_code"] or "",
                response=ResponseView.from_dict(json.loads(row["response_json"] or "{}")),
                context=json.loads(row["context_json"] or "{}"),
                timestamp=row["datetime_utc"] or "",
            )
        )
    return {
        "id": thread[0]["id"],
        "name": thread[0]["name"] or "Untitled conversation",
        "model": turns[0].response.model if turns else None,
        "turns": turns,
    }


def rename_conversation(thread_id: str, name: str, db: Database | None = None) -> dict:
    """Rename one stored conversation. Returns the id and the name kept.

    The name is the user's own label on llm's own ``threads`` row, so the
    sidebar, the chat title and ``llm logs`` all read the same string. An
    unknown id is a KeyError, an empty name a ValueError: neither may be
    reported as a rename that happened.
    """
    cleaned = " ".join((name or "").split())
    if not cleaned:
        raise ValueError("a conversation name must not be empty")
    database = db or connect()
    if not database["threads"].count_where("id = ?", [thread_id]):
        raise KeyError(thread_id)
    database["threads"].update(thread_id, {"name": cleaned})
    return {"id": thread_id, "name": cleaned}


def thread_messages(thread_id: str, db: Database | None = None) -> list[Any]:
    """The stored message chain, exactly as it was written.

    This is what lets a restored conversation carry the original content
    blocks rather than a history rebuilt from chat text: reasoning
    signatures, citations and provider metadata are part of the message.
    """
    return LogStore(db or connect()).thread_messages(thread_id)
