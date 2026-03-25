import json
import sqlite3
from typing import Generator, Literal
from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("vink")
except PackageNotFoundError:
    __version__ = "0.0.0"


class SQLiteWrapper:
    """Central SQLite connection and schema management for VinkDB."""

    @validate_arguments
    def __init__(self, path: str, index_config: dict):
        """Initialize SQLite wrapper.

        Args:
            path: Path to SQLite database file.
            index_config: Optional dict with index metadata (dimension, metric, strategy).
                Used to initialize db_meta table on first creation.
        """
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._ensure_tables_exist()

        self._validate_config(index_config)

        index_config["version"] = __version__
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
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS db_meta (key TEXT PRIMARY KEY, value TEXT)"
        )

        cursor.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS content_fts5 USING fts5(id, content)"
        )

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
        if "dimension" not in new_config or "metric" not in new_config:
            return

        stored_dim = self["dimension"]
        stored_metric = self["metric"]

        new_dim = new_config["dimension"]
        if stored_dim is not None and str(new_dim) != stored_dim:
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
    def insert(self, vec_records: VectorRecords, is_buffer: bool = False) -> None:
        """Insert vec_records into SQLite.

        Args:
            vec_records: VectorRecords object.
            is_buffer: If True, marks all vec_records as buffered vec_records.
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
            "INSERT INTO vec_records (id, metadata, embedding, buffered) VALUES (?, jsonb(?), ?, ?)",
            [(r["id"], r["metadata"], r["embedding"], is_buffer) for r in records],
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

    @validate_arguments
    def count(self, mode: Literal["active", "deleted", "all"]) -> int:
        """Count vectors in the database.

        Args:
            mode (Literal["active", "deleted", "all"]): Which vectors to count - "active", "deleted", or "all".

        Returns:
            int: Count of vectors.
        """
        cursor = self._conn.cursor()
        if mode == "active":
            cursor.execute("SELECT COUNT(*) FROM vec_records WHERE deleted = FALSE")
        elif mode == "deleted":
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
    def iter_embeddings(self, batch_size: int = 50000) -> Generator[list, None, None]:
        """Iterate over embeddings in batches."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT embedding FROM vec_records")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [row[0] for row in rows]

    def __getitem__(self, key: str) -> str | None:
        """Get a metadata value from db_meta table."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT value FROM db_meta WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else None

    def __setitem__(self, key: str, value: str) -> None:
        """Set a metadata value in db_meta table."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)",
            (key, value)
        )
