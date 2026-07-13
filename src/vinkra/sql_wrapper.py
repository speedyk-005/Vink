import json
import sqlite3
from collections.abc import Iterator
from typing import Literal

from vinkra.utils.input_validation import validate_arguments

if sqlite3.sqlite_version_info < (3, 45, 0):
    try:
        import pysqlite3 as sqlite3
    except ImportError:
        raise ImportError(
            f"Your SQLite is {sqlite3.sqlite_version} but 3.45.0+ is required. "
            "Fix it: pip install pysqlite3"
        ) from None


class SQLiteWrapper:
    """Central SQLite connection and schema management for VinkraDB."""

    @validate_arguments
    def __init__(self, path: str, index_config: dict):
        """Initialize SQLite wrapper.

        Args:
            path: Path to SQLite database file.
            index_config: Optional dict with index metadata (dim, metric, strategy).
                Used to initialize db_meta table on first creation.
        """
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA mmap_size=268435456;")  # 256 MB
        self._ensure_tables_exist()
        self._validate_config(index_config)

        for k, v in index_config.items():
            self[k] = v
        self._conn.commit()

    @property
    def conn(self):
        """Expose the raw connection"""
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _ensure_tables_exist(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS content_fts5 USING fts5(
                id UNINDEXED,
                content,
                tokenize='trigram'
            )
         """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vec_records (
                id BLOB PRIMARY KEY,        -- UUID bytes
                metadata BLOB NOT NULL,     -- JSON binary format
                embedding BLOB,
                deleted BOOLEAN DEFAULT FALSE,
                buffered BOOLEAN DEFAULT FALSE
            )
        """)

        cursor.executescript("""
            CREATE INDEX IF NOT EXISTS idx_vec_records_deleted ON vec_records(deleted);
            CREATE INDEX IF NOT EXISTS idx_vec_records_buffer ON vec_records(buffered);
        """)

    def _validate_config(self, new_config: dict) -> None:
        """Validate incoming config against stored db_meta if database exists."""
        stored_dim = self["dim"]
        stored_metric = self["metric"]

        new_dim = new_config["dim"]
        if stored_dim is not None and new_dim != stored_dim:
            raise ValueError(
                f"Dimension mismatch: cannot open existing database with "
                f"dimension {new_dim}, stored dimension is {stored_dim}"
            )

        new_metric = new_config["metric"]
        if stored_metric is not None and new_metric != stored_metric:
            raise ValueError(
                f"Metric mismatch: cannot open existing database with "
                f"metric '{new_metric}', stored metric is '{stored_metric}'"
            )

    def commit(self) -> None:
        """Explicitly commit the current transaction."""
        self._conn.commit()

    @validate_arguments
    def insert(self, vec_records: list[dict], *, is_buffer: bool = False) -> None:
        """Insert vec_records into SQLite.

        Args:
            vec_records: List of dicts with id, content, metadata, embedding keys.
            is_buffer: If True, marks all vec_records as buffered.
        """
        cursor = self._conn.cursor()
        records = [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": json.dumps(r.get("metadata", {})),
                "embedding": r["embedding"].tobytes(),
            }
            for r in vec_records
        ]

        cursor.executemany(
            "INSERT OR REPLACE INTO content_fts5 (id, content) VALUES (?, ?)",
            [(r["id"], r["content"]) for r in records],
        )
        cursor.executemany(
            """
            INSERT INTO vec_records
            (id, metadata, embedding, buffered)
            VALUES (?, jsonb(?), ?, ?)
            ON CONFLICT(id)
            DO UPDATE SET
                metadata=excluded.metadata,
                embedding=excluded.embedding,
                buffered=excluded.buffered
            """,
            [(r["id"], r["metadata"], r["embedding"], is_buffer) for r in records],
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
        where_sql: str | None = None,
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
        if where_sql:
            sql += f" {where_sql}"

        cursor.execute(sql, params or [])
        return cursor.fetchall()

    @validate_arguments
    def count(self, status: Literal["active", "deleted", "all"] = "active") -> int:
        """Count vectors in the database.

        Args:
            status: Which vectors to count. Defaults to "active".

        Returns:
            Count of vectors.
        """
        cursor = self._conn.cursor()
        if status == "active":
            cursor.execute("SELECT COUNT(*) FROM vec_records WHERE deleted = FALSE")
        elif status == "deleted":
            cursor.execute("SELECT COUNT(*) FROM vec_records WHERE deleted = TRUE")
        else:
            cursor.execute("SELECT COUNT(*) FROM vec_records")
        return cursor.fetchone()[0]

    def clear_buffer(self) -> None:
        """Set all buffer flags to False."""
        cursor = self._conn.cursor()
        cursor.execute("UPDATE vec_records SET buffered = FALSE WHERE buffered = TRUE")

    def compact(self) -> None:
        """Hard-delete all soft-deleted records from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM vec_records WHERE deleted = TRUE")

    @validate_arguments
    def iter_embeddings(self) -> Iterator[list[bytes]]:
        """Iterate over embeddings in batches."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT embedding FROM vec_records")

        # Adaptive batch size for embedding scans.
        # Targets ~64MB of raw float32 embedding data per fetch.
        dim = self["dim"]
        batch_size = max(1, (64 * 1024 * 1024) // (dim * 4))

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [row[0] for row in rows]

    def __getitem__(self, key: str) -> str | None:
        """Get a metadata value from db_meta table."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT value FROM db_meta WHERE key = ?", (key,))
        fetched = cursor.fetchone()
        res = fetched[0] if fetched else None
        if key == "dim" and res is not None:
            res = int(res)
        return res

    def __setitem__(self, key: str, value: str) -> None:
        """Set a metadata value in db_meta table."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)", (key, value)
        )
