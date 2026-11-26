"""Durable local workspace records; independent of model and training artifacts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class WorkspaceStore:
    """Small SQLite document store with atomic updates and no network dependency."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS documents (kind TEXT NOT NULL, id TEXT NOT NULL, "
                         "body TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, "
                         "PRIMARY KEY(kind,id))")
            conn.execute("CREATE INDEX IF NOT EXISTS documents_updated ON documents(kind,updated)")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def _put(self, kind, record):
        with self._lock, self._connection() as conn:
            conn.execute("INSERT INTO documents(kind,id,body,created,updated) VALUES(?,?,?,?,?) "
                         "ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body,updated=excluded.updated",
                         (kind, record["id"], self._encode(record), record["created"], record["updated"]))
        return record["id"]

    def _get(self, kind, identifier):
        if not isinstance(identifier, str) or len(identifier) > 128:
            return None
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT body FROM documents WHERE kind=? AND id=?",
                               (kind, identifier)).fetchone()
        return json.loads(row[0]) if row else None

    def _list(self, kind, limit=50):
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT body FROM documents WHERE kind=? ORDER BY updated DESC LIMIT ?",
                                (kind, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _update(self, kind, identifier, change):
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT body FROM documents WHERE kind=? AND id=?",
                               (kind, identifier)).fetchone()
            if not row:
                raise ValueError("Workspace record not found")
            record = json.loads(row[0])
            change(record)
            record["updated"] = time.time()
            conn.execute("UPDATE documents SET body=?,updated=? WHERE kind=? AND id=?",
                         (self._encode(record), record["updated"], kind, identifier))
        return record

    def put_analysis(self, result, analysis_id=None):
        if not isinstance(result, dict):
            raise ValueError("Analysis must be an object")
        identifier = analysis_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
            raise ValueError("Invalid analysis ID")
        now = time.time()
        old = self._get("analysis", identifier)
        record = {"id": identifier, "created": old["created"] if old else now,
                  "updated": now, "result": result}
        self._put("analysis", record)
        return identifier

    def get_analysis(self, analysis_id):
        record = self._get("analysis", analysis_id)
        if record is None:
            return None
        return {**record["result"], "analysis_id": record["id"], "created": record["created"]}

    def list_analyses(self, limit=50):
        return [{"id": r["id"], "analysis_id": r["id"], "created": r["created"],
                 "updated": r["updated"], "target": r["result"].get("target", ""),
                 "solved": bool(r["result"].get("solved")),
                 "route_count": len(r["result"].get("routes", [])),
                 "model_calls": r["result"].get("model_calls", 0),
                 "mode": r["result"].get("mode")}
                for r in self._list("analysis", limit)]

    def create_session(self, title="New conversation"):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
            raise ValueError("Conversation title must contain 1 to 120 characters")
        now = time.time()
        record = {"id": uuid.uuid4().hex, "title": title.strip(), "created": now,
                  "updated": now, "messages": [], "tool_calls": []}
        self._put("session", record)
        return record

    def get_session(self, session_id):
        return self._get("session", session_id)

    def session_for_analysis(self, analysis_id):
        if not isinstance(analysis_id, str) or not 1 <= len(analysis_id) <= 128:
            return None
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT body FROM documents WHERE kind='session' AND body LIKE ? ORDER BY updated DESC",
                                ('%"analysis_id":"' + analysis_id + '"%',)).fetchall()
        for row in rows:
            session = json.loads(row[0])
            if any(t.get("name") in ("plan_routes", "plan_fixed_core_case", "evaluate_routes") and t.get("status") == "complete"
                   and t.get("result", {}).get("analysis_id") == analysis_id for t in session.get("tool_calls", [])):
                return session["id"]
            if any(m.get("role") == "user" and m.get("context", {}).get("analysis_id") == analysis_id
                   for m in session.get("messages", [])):
                return session["id"]
        return None

    def list_sessions(self, limit=50):
        return [{"id": s["id"], "title": s["title"], "created": s["created"],
                 "updated": s["updated"], "message_count": len(s["messages"]),
                 "last_message": next((m["content"][:180] for m in reversed(s["messages"])
                                       if m["role"] in ("user", "assistant") and m["content"]), "")}
                for s in self._list("session", limit)]

    def append_message(self, session_id, role, content, **metadata):
        if role not in ("user", "assistant", "tool") or not isinstance(content, str):
            raise ValueError("Invalid conversation message")
        message = {**metadata, "id": uuid.uuid4().hex, "role": role,
                   "content": content, "created": time.time()}
        self._update("session", session_id, lambda s: s["messages"].append(message))
        return message

    def append_tool_trace(self, session_id, trace):
        self._update("session", session_id, lambda s: s["tool_calls"].append(trace))
        return trace

    def rename_session(self, session_id, title):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:
            raise ValueError("Conversation title must contain 1 to 120 characters")
        return self._update("session", session_id, lambda s: s.update(title=title.strip()))

    def save_checkpoint(self, analysis_id, title="Saved search", parameters=None, session_id=None):
        analysis = self.get_analysis(analysis_id)
        if analysis is None:
            raise ValueError("Analysis not found")
        if not isinstance(parameters or {}, dict):
            raise ValueError("Search parameters must be an object")
        now = time.time()
        record = {"id": uuid.uuid4().hex, "created": now, "updated": now,
                  "analysis_id": analysis_id, "title": str(title)[:120],
                  "parameters": parameters or {}, "session_id": session_id,
                  "target": analysis.get("target"), "kind": "completed_search_snapshot"}
        self._put("checkpoint", record)
        return record

    def get_checkpoint(self, checkpoint_id):
        return self._get("checkpoint", checkpoint_id)

    def list_checkpoints(self, limit=50):
        return self._list("checkpoint", limit)

    def add_review(self, analysis_id, note, decision="pending", session_id=None):
        if self.get_analysis(analysis_id) is None:
            raise ValueError("Analysis not found")
        if not isinstance(note, str) or not 1 <= len(note.strip()) <= 4000:
            raise ValueError("Review note must contain 1 to 4000 characters")
        if decision not in ("pending", "approved", "rejected", "needs_revision"):
            raise ValueError("Invalid review decision")
        now = time.time()
        record = {"id": uuid.uuid4().hex, "created": now, "updated": now,
                  "analysis_id": analysis_id, "note": note.strip(), "decision": decision,
                  "session_id": session_id}
        self._put("review", record)
        return record

    def list_reviews(self, analysis_id=None, limit=50):
        # Filter in SQL before limiting so a busy workspace cannot hide an analysis review.
        if analysis_id is None:
            return self._list("review", limit)
        with self._lock, self._connection() as conn:
            rows = conn.execute("SELECT body FROM documents WHERE kind='review' "
                                "AND json_extract(body,'$.analysis_id')=? ORDER BY updated DESC LIMIT ?",
                                (analysis_id, max(1, min(int(limit), 200)))).fetchall()
        return [json.loads(row[0]) for row in rows]
