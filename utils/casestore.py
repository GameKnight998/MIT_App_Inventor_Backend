"""Persist an investigation's images and results so they can be reviewed together.

A case is the unit an investigator actually works in: several photos from one
incident, analysed and compared. Keeping them server-side means the client can
add images over time and re-read the combined picture, rather than holding a
growing blob of JSON in the app.

SQLite is the right size of tool here. It needs no service to run, no
credentials, and no scaling story for a workload of one image at a time, which
keeps the whole feature deployable on a free tier.

IMPORTANT deployment caveat: on Render's free tier the filesystem is ephemeral,
so this database is wiped on every deploy and container restart. That is fine for
its purpose (grouping a working session) but it is not durable storage. Point
`CASE_DB_PATH` at a mounted disk if cases need to survive.

Each call opens its own connection: SQLite handles that well, and it avoids
sharing a connection across FastAPI's worker threads, which is not safe.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "uploads" / "cases.db")
CASE_DB_PATH = os.getenv("CASE_DB_PATH", _DEFAULT_DB)
CASE_ENABLED = os.getenv("CASE_ENABLED", "1") not in ("0", "false", "False")
# Guard against one case growing without bound on a small instance.
CASE_MAX_ITEMS = int(os.getenv("CASE_MAX_ITEMS", "50"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id    TEXT PRIMARY KEY,
    title      TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS case_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id       TEXT NOT NULL,
    filename      TEXT,
    created_at    REAL NOT NULL,
    latitude      REAL,
    longitude     REAL,
    confidence    REAL,
    location_name TEXT,
    country       TEXT,
    region        TEXT,
    payload       TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_case_items_case ON case_items (case_id);
"""


def _connect() -> sqlite3.Connection:
    Path(CASE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CASE_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create the schema if it does not exist. Safe to call repeatedly."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def create_case(title: Optional[str] = None) -> str:
    """Start a new case and return its id."""
    case_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO cases (case_id, title, created_at) VALUES (?, ?, ?)",
            (case_id, title, time.time()),
        )
    return case_id


def case_exists(case_id: str) -> bool:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT 1 FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
    return row is not None


def item_count(case_id: str) -> int:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM case_items WHERE case_id = ?", (case_id,)
        ).fetchone()
    return int(row["n"]) if row else 0


def add_item(case_id: str, filename: Optional[str], payload: dict[str, Any]) -> int:
    """Store one analysed image in a case. Returns the new item's id."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        cursor = conn.execute(
            """
            INSERT INTO case_items (
                case_id, filename, created_at, latitude, longitude,
                confidence, location_name, country, region, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                filename,
                time.time(),
                payload.get("latitude"),
                payload.get("longitude"),
                payload.get("confidence"),
                payload.get("location_name"),
                payload.get("country"),
                payload.get("region"),
                json.dumps(payload),
            ),
        )
        return int(cursor.lastrowid or 0)


def list_items(case_id: str) -> list[dict[str, Any]]:
    """Every stored result for a case, oldest first, with payloads restored."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            "SELECT * FROM case_items WHERE case_id = ? ORDER BY created_at ASC",
            (case_id,),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            payload = {}
        items.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "created_at": row["created_at"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "confidence": row["confidence"],
                "location_name": row["location_name"],
                "country": row["country"],
                "region": row["region"],
                "alternatives": payload.get("alternatives", []),
                "payload": payload,
            }
        )
    return items


def get_case(case_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        row = conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "case_id": row["case_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "items": list_items(case_id),
    }


def list_cases(limit: int = 25) -> list[dict[str, Any]]:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute(
            """
            SELECT c.case_id, c.title, c.created_at,
                   COUNT(i.id) AS items
            FROM cases c
            LEFT JOIN case_items i ON i.case_id = c.case_id
            GROUP BY c.case_id
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
    return [
        {
            "case_id": r["case_id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "items": int(r["items"]),
        }
        for r in rows
    ]


def delete_case(case_id: str) -> bool:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        conn.execute("DELETE FROM case_items WHERE case_id = ?", (case_id,))
        cursor = conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
        return cursor.rowcount > 0
