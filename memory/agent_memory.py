"""
memory/agent_memory.py — Persistent agent memory store.

Provides a lightweight SQLite-backed key/value + full-text memory system for
agents that need to recall facts, past decisions, or experiment outcomes across
pipeline runs.

Design:
  - MemoryStore   : low-level SQLite wrapper (CRUD + search)
  - MemoryEntry   : Pydantic model for a single memory record
  - AgentMemory   : high-level interface scoped to a (run_id, agent_name) pair

Harness Engineering principle (Self-Healing):
  Agents can store failure patterns here so that the debug agent can recall
  previously successful repair strategies without re-discovering them from scratch.

Usage:
    from memory.agent_memory import AgentMemory

    mem = AgentMemory(agent_name="training_agent", run_id="run-abc123")
    mem.remember("best_model", "XGBoostClassifier with n_estimators=200")
    value = mem.recall("best_model")
    similar = mem.search("XGBoost")
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "memory/agent_memory.db")


# ── Schema ───────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT    NOT NULL,
    agent_name TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    tags       TEXT    DEFAULT '',
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_run_agent ON memories (run_id, agent_name);
CREATE INDEX IF NOT EXISTS idx_memories_key       ON memories (key);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(run_id, agent_name, key, value, tags, content='memories', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS memories_ai
    AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, run_id, agent_name, key, value, tags)
        VALUES (new.id, new.run_id, new.agent_name, new.key, new.value, new.tags);
    END;

CREATE TRIGGER IF NOT EXISTS memories_au
    AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, run_id, agent_name, key, value, tags)
        VALUES ('delete', old.id, old.run_id, old.agent_name, old.key, old.value, old.tags);
        INSERT INTO memories_fts(rowid, run_id, agent_name, key, value, tags)
        VALUES (new.id, new.run_id, new.agent_name, new.key, new.value, new.tags);
    END;

CREATE TRIGGER IF NOT EXISTS memories_ad
    AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, run_id, agent_name, key, value, tags)
        VALUES ('delete', old.id, old.run_id, old.agent_name, old.key, old.value, old.tags);
    END;
"""


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    id: int
    run_id: str
    agent_name: str
    key: str
    value: str
    tags: list[str]
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: tuple) -> "MemoryEntry":
        id_, run_id, agent_name, key, value, tags_str, created_at, updated_at = row
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        return cls(
            id=id_,
            run_id=run_id,
            agent_name=agent_name,
            key=key,
            value=value,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "key": self.key,
            "value": self.value,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ── Low-level store ──────────────────────────────────────────────────────────


class MemoryStore:
    """Thread-safe SQLite-backed memory store."""

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        # For on-disk databases, create parent directory.
        # For :memory: we keep a single persistent connection so tables survive.
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # Single shared connection for in-memory databases
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a ready-to-use connection (persistent for :memory:, new for disk)."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _execute(self, fn):
        """
        Execute fn(conn) and commit. Manages connection lifetime for disk paths.
        """
        conn = self._get_conn()
        try:
            result = fn(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            # Only close new on-disk connections; keep persistent ones open
            if self._persistent_conn is None:
                conn.close()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_DDL)
        conn.commit()

    # CRUD ---------------------------------------------------------------

    def upsert(
        self,
        run_id: str,
        agent_name: str,
        key: str,
        value: str,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Insert or update a memory entry. Returns the row id."""
        now = time.time()
        tags_str = ",".join(tags or [])

        def _fn(conn):
            existing = conn.execute(
                "SELECT id FROM memories WHERE run_id=? AND agent_name=? AND key=?",
                (run_id, agent_name, key),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memories SET value=?, tags=?, updated_at=? WHERE id=?",
                    (value, tags_str, now, existing[0]),
                )
                return existing[0]
            else:
                cur = conn.execute(
                    "INSERT INTO memories (run_id, agent_name, key, value, tags, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, agent_name, key, value, tags_str, now, now),
                )
                return cur.lastrowid

        return self._execute(_fn)

    def get(self, run_id: str, agent_name: str, key: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory by (run_id, agent_name, key)."""
        def _fn(conn):
            return conn.execute(
                "SELECT id, run_id, agent_name, key, value, tags, created_at, updated_at "
                "FROM memories WHERE run_id=? AND agent_name=? AND key=?",
                (run_id, agent_name, key),
            ).fetchone()
        row = self._execute(_fn)
        return MemoryEntry.from_row(row) if row else None

    def get_all(self, run_id: str, agent_name: str) -> list[MemoryEntry]:
        """Return all memories for a (run_id, agent_name) pair."""
        def _fn(conn):
            return conn.execute(
                "SELECT id, run_id, agent_name, key, value, tags, created_at, updated_at "
                "FROM memories WHERE run_id=? AND agent_name=? ORDER BY updated_at DESC",
                (run_id, agent_name),
            ).fetchall()
        return [MemoryEntry.from_row(r) for r in self._execute(_fn)]

    def delete(self, run_id: str, agent_name: str, key: str) -> bool:
        """Delete a memory. Returns True if a row was deleted."""
        def _fn(conn):
            cur = conn.execute(
                "DELETE FROM memories WHERE run_id=? AND agent_name=? AND key=?",
                (run_id, agent_name, key),
            )
            return cur.rowcount

        return self._execute(_fn) > 0

    def search(self, query: str, run_id: Optional[str] = None, limit: int = 20) -> list[MemoryEntry]:
        """Full-text search across value and tags columns."""
        def _fn(conn):
            if run_id:
                return conn.execute(
                    "SELECT m.id, m.run_id, m.agent_name, m.key, m.value, m.tags, m.created_at, m.updated_at "
                    "FROM memories m "
                    "JOIN memories_fts f ON m.id = f.rowid "
                    "WHERE memories_fts MATCH ? AND m.run_id=? "
                    "ORDER BY rank LIMIT ?",
                    (query, run_id, limit),
                ).fetchall()
            else:
                return conn.execute(
                    "SELECT m.id, m.run_id, m.agent_name, m.key, m.value, m.tags, m.created_at, m.updated_at "
                    "FROM memories m "
                    "JOIN memories_fts f ON m.id = f.rowid "
                    "WHERE memories_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (query, limit),
                ).fetchall()

        return [MemoryEntry.from_row(r) for r in self._execute(_fn)]

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Return a structured summary of all memories for a pipeline run."""
        def _fn(conn):
            agents = conn.execute(
                "SELECT DISTINCT agent_name FROM memories WHERE run_id=? ORDER BY agent_name",
                (run_id,),
            ).fetchall()
            result: dict[str, Any] = {}
            for (agent_name,) in agents:
                rows = conn.execute(
                    "SELECT key, value FROM memories WHERE run_id=? AND agent_name=?",
                    (run_id, agent_name),
                ).fetchall()
                result[agent_name] = {k: v for k, v in rows}
            return result

        return self._execute(_fn)


# ── High-level agent interface ───────────────────────────────────────────────

# Module-level singleton store (lazy init on first use)
_store: Optional[MemoryStore] = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


class AgentMemory:
    """
    High-level memory interface scoped to a specific agent and pipeline run.

    Each agent should create its own AgentMemory instance:
        mem = AgentMemory(agent_name="training_agent", run_id="run-abc123")
        mem.remember("best_params", json.dumps({"n_estimators": 200}))
        params = mem.recall("best_params")
    """

    def __init__(
        self,
        agent_name: str,
        run_id: str,
        store: Optional[MemoryStore] = None,
    ) -> None:
        self.agent_name = agent_name
        self.run_id = run_id
        self._store = store or _get_store()

    # ── Core operations ──────────────────────────────────────────────

    def remember(
        self,
        key: str,
        value: Any,
        tags: Optional[list[str]] = None,
    ) -> None:
        """Store or update a memory. Non-string values are JSON-serialised."""
        if not isinstance(value, str):
            value = json.dumps(value, default=str)
        self._store.upsert(
            run_id=self.run_id,
            agent_name=self.agent_name,
            key=key,
            value=value,
            tags=tags,
        )

    def recall(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a memory by key.
        Returns the raw string value, or `default` if not found.
        Tries to JSON-decode the value; returns the raw string on failure.
        """
        entry = self._store.get(self.run_id, self.agent_name, key)
        if entry is None:
            return default
        try:
            return json.loads(entry.value)
        except (json.JSONDecodeError, ValueError):
            return entry.value

    def forget(self, key: str) -> bool:
        """Delete a memory entry. Returns True if it existed."""
        return self._store.delete(self.run_id, self.agent_name, key)

    def all_memories(self) -> list[MemoryEntry]:
        """Return all memory entries for this agent in this run."""
        return self._store.get_all(self.run_id, self.agent_name)

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Full-text search within this run's memories."""
        return self._store.search(query, run_id=self.run_id, limit=limit)

    def as_context_string(self, max_entries: int = 20) -> str:
        """
        Return a compact string suitable for injection into an agent prompt.
        Includes the most recently updated memories for this agent.
        """
        entries = self._store.get_all(self.run_id, self.agent_name)[:max_entries]
        if not entries:
            return "(no prior memories for this agent in this run)"
        lines = [f"Agent memory for {self.agent_name} (run {self.run_id}):"]
        for e in entries:
            lines.append(f"  [{e.key}]: {e.value[:200]}")
        return "\n".join(lines)

    # ── Cross-agent helpers ──────────────────────────────────────────

    def remember_global(self, key: str, value: Any, tags: Optional[list[str]] = None) -> None:
        """Store a run-level memory accessible by any agent using agent_name='global'."""
        if not isinstance(value, str):
            value = json.dumps(value, default=str)
        self._store.upsert(
            run_id=self.run_id,
            agent_name="global",
            key=key,
            value=value,
            tags=tags,
        )

    def recall_global(self, key: str, default: Any = None) -> Any:
        """Read a run-level global memory."""
        entry = self._store.get(self.run_id, "global", key)
        if entry is None:
            return default
        try:
            return json.loads(entry.value)
        except (json.JSONDecodeError, ValueError):
            return entry.value

    def run_summary(self) -> dict[str, Any]:
        """Return all memories for every agent in this run."""
        return self._store.get_run_summary(self.run_id)
