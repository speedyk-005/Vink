import json
import sqlite3
from typing import Generator

from vink.models import VectorRecords
from vink.utils.input_validation import validate_arguments

if sqlite3.sqlite_version_info < (3, 45, 0):
    try:
        import pysqlite3 as sqlite3
    except ImportError:
        raise ImportError(
            f"Your SQLite is {sqlite3.sqlite_version} but 3.45.0+ is required for JSONB support. "
            "Fix it: pip install pysqlite3"
        ) from None


class SQLiteWrapper:
    """Central SQLite connection and schema management for VinkDB."""

    @validate_arguments
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_tables_exist()

    @property
    def conn(self):
        """Expose the raw connection"""
        return self._conn

    def _ensure_tables_exist(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS content_fts5 USING fts5(id, content)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vec_records (
                id BLOB PRIMARY KEY,        -- UUID bytes
                metadata BLOB NOT NULL,     -- JSON binary format
                embedding BLOB,
                deleted BOOLEAN DEFAULT FALSE,
                buffer BOOLEAN DEFAULT FALSE
            )
        """)

        cursor.executescript("""
            CREATE INDEX IF NOT EXISTS idx_vec_records_deleted ON vec_records(deleted);
            CREATE INDEX IF NOT EXISTS idx_vec_records_buffer ON vec_records(buffer);
        """)

    def commit(self) -> None:
        """Explicitly commit the current transaction."""
        self._conn.commit()

    @validate_arguments
    def insert(self, vec_records: VectorRecords, buffer: bool = False) -> None:
        """Insert vec_records into SQLite.

        Args:
            vec_records: VectorRecords object.
            buffer: If True, marks all vec_records as buffer vec_records.
        """
        cursor = self._conn.cursor()

        records = [
            {
                "id": r.id,
                "content": r.content,
                "metadata": json.dumps(r.metadata),
                "embedding": r.embedding.tobytes(),
            }
            for r in vec_records.records
        ]

        cursor.executemany(
            "INSERT INTO vec_records (id, metadata, embedding, buffer) VALUES (?, jsonb(?), ?, ?)",
            [(r["id"], r["metadata"], r["embedding"], buffer) for r in records],
        )
        cursor.executemany(
            "INSERT INTO content_fts5 (id, content) VALUES (?, ?)",
            [(r["id"], r["content"]) for r in records],
        )

    @validate_arguments
    def soft_delete(self, ids: list[bytes]) -> None:
        """Soft-delete vec_records from SQLite (marks as deleted)."""
        cursor = self._conn.cursor()
        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"UPDATE vec_records SET deleted = TRUE WHERE id IN ({placeholders})",
            ids,
        )

    @validate_arguments
    def fetch(
        self,
        *,
        where: str | None = None,
        params: list | tuple | None = None,
        include_vectors: bool = False,
    ):
        """Fetch vec_records from SQLite."""
        cursor = self._conn.cursor()

        include_emb = ", embedding " if include_vectors else ""
        sql = f"""
            SELECT vec_records.id, content, json(metadata) {include_emb}
            FROM vec_records
            JOIN content_fts5 USING (id)
        """
        if where:
            sql += f" WHERE {where}"

        cursor.execute(sql, params or [])
        return cursor.fetchall()

    def count(self) -> int:
        """Count the number of active (non-deleted) vectors in the database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vec_records WHERE deleted = FALSE")
        return cursor.fetchone()[0]

    def clear_buffer(self) -> None:
        """Set all buffer flags to False."""
        cursor = self._conn.cursor()
        cursor.execute("UPDATE vec_records SET buffer = FALSE WHERE buffer = TRUE")

    def compact(self) -> None:
        """Hard-delete all soft-deleted records from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM vec_records WHERE deleted = TRUE")

    @validate_arguments
    def iter_embeddings(self, batch_size: int = 50000) -> Generator[list, None, None]:
        """Iterate over embeddings in batches."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT embedding FROM vec_records")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [row[0] for row in rows]
