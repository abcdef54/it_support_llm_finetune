"""Small transactional checkpoint for long benchmark runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class Checkpoint:
    def __init__(self, path: Path, fingerprint: str, *, resume: bool):
        exists = path.exists()
        if exists and not resume:
            raise FileExistsError(f"Benchmark progress exists at {path}; pass --resume to continue it")
        if resume and not exists:
            raise FileNotFoundError(f"No benchmark progress to resume at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("CREATE TABLE IF NOT EXISTS run (fingerprint TEXT NOT NULL)")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS items (stage TEXT NOT NULL, position INTEGER NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (stage, position))"
            )
            if exists:
                saved = self.db.execute("SELECT fingerprint FROM run").fetchall()
                if saved != [(fingerprint,)]:
                    raise ValueError("Checkpoint configuration or benchmark data changed; cannot safely resume")
            else:
                self.db.execute("INSERT INTO run VALUES (?)", (fingerprint,))
                self.db.commit()
        except BaseException:
            self.db.close()
            raise

    def load(self, stage: str) -> list:
        rows = self.db.execute("SELECT position, value FROM items WHERE stage = ? ORDER BY position", (stage,)).fetchall()
        if any(position != index for index, (position, _) in enumerate(rows)):
            raise ValueError(f"Checkpoint {stage} records are not a contiguous prefix")
        return [json.loads(value) for _, value in rows]

    def save(self, stage: str, start: int, values: list) -> None:
        with self.db:
            self.db.executemany(
                "INSERT INTO items VALUES (?, ?, ?)",
                ((stage, start + index, json.dumps(value)) for index, value in enumerate(values)),
            )

    def close(self) -> None:
        self.db.close()
