"""SQLite storage for conversations, messages, feedback and request traces.

Traces hold timings, counts and versions only (no document text or prompts);
the question and answer live in the messages table because the chat history
needs them.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload TEXT,
    request_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    request_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    total_ms REAL,
    error TEXT,
    config TEXT,
    spans TEXT NOT NULL
);
"""
TITLE_CHARS = 60


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id():
    return uuid.uuid4().hex


class ChatStore:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _write(self, sql, params=()):
        with self.lock, self.db:
            return self.db.execute(sql, params)

    def _read(self, sql, params=()):
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    def create_conversation(self, first_message):
        title = " ".join(first_message.split())
        if len(title) > TITLE_CHARS:
            title = title[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"
        conversation_id, stamp = new_id(), now()
        self._write(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, stamp, stamp),
        )
        return conversation_id

    def conversation(self, conversation_id):
        rows = self._read("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        return dict(rows[0]) if rows else None

    def conversations(self, limit=100):
        return [dict(row) for row in self._read(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        )]

    def delete_conversation(self, conversation_id):
        return self._write("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def add_message(self, conversation_id, role, content, payload=None, request_id=None, message_id=None):
        message_id, stamp = message_id or new_id(), now()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO messages (id, conversation_id, role, content, payload, request_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(payload) if payload else None, request_id, stamp),
            )
            self.db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (stamp, conversation_id))
        return message_id, stamp

    def messages(self, conversation_id):
        rows = self._read(
            "SELECT m.*, f.rating FROM messages m LEFT JOIN feedback f ON f.message_id = m.id"
            " WHERE m.conversation_id = ? ORDER BY m.created_at, m.rowid",
            (conversation_id,),
        )
        messages = []
        for row in rows:
            message = dict(row)
            message["payload"] = json.loads(message["payload"]) if message["payload"] else None
            if message["payload"] is not None:
                message["payload"]["feedback"] = message["rating"]
            messages.append(message)
        return messages

    def history(self, conversation_id):
        """[(user message, assistant answer)] for follow-up rewriting, oldest first."""
        turns = []
        for message in self.messages(conversation_id):
            if message["role"] == "user":
                turns.append([message["content"], None])
            elif turns:
                turns[-1][1] = message["content"]
        return [tuple(turn) for turn in turns]

    def message_exists(self, message_id, role=None):
        sql = "SELECT 1 FROM messages WHERE id = ?" + (" AND role = ?" if role else "")
        return bool(self._read(sql, (message_id, role) if role else (message_id,)))

    def set_feedback(self, message_id, rating, comment=None):
        self._write(
            "INSERT INTO feedback (message_id, rating, comment, created_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(message_id) DO UPDATE SET rating = excluded.rating, comment = excluded.comment,"
            " created_at = excluded.created_at",
            (message_id, rating, comment, now()),
        )

    def save_trace(self, trace, conversation_id, status, config, error=None):
        data = trace.to_dict()
        self._write(
            "INSERT OR REPLACE INTO traces (request_id, conversation_id, created_at, status, total_ms, error, config, spans)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trace.request_id, conversation_id, now(), status, data["total_ms"], error, json.dumps(config),
             json.dumps(data["spans"])),
        )

    def trace(self, request_id):
        rows = self._read("SELECT * FROM traces WHERE request_id = ?", (request_id,))
        if not rows:
            return None
        trace = dict(rows[0])
        trace["config"] = json.loads(trace["config"]) if trace["config"] else None
        trace["spans"] = json.loads(trace["spans"])
        return trace

    def close(self):
        self.db.close()
