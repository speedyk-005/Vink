from abc import ABC, abstractmethod
from typing import Any, Literal
import json
from pathlib import Path
from uuid import UUID
import numpy as np

import pysqlite3 as sqlite3

from vink.models import VectorRecords


class BaseStrategy(ABC):
    """
    Base class for search strategies.

    Provides abstract interface for exact and approximate search implementations.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        dir_path: Path | None,
        dim: int,
        is_exact: bool,
        in_memory: bool,
        metric: Literal["l2", "dot"] = "l2",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            conn (sqlite3.Connection): SQLite database connection for storing records.
            dir_path (Path | None): Path to store vector data for querying. Defaults to None.
            dim (int): Dimension of the vectors.
            is_exact (bool): Whether this strategy uses exact search.
            in_memory (bool): Whether using in-memory storage.
            metric (Literal["l2", "dot"], optional): Distance metric to use. Defaults to "l2".
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        self.conn = conn
        self.dir_path = dir_path
        self.is_exact = is_exact
        self.dim = dim
        self.in_memory = in_memory
        self.metric = metric
        self.verbose = verbose

        if dir_path is None and not in_memory:
            raise ValueError("in_memory must be True if no dir_path is provided.")

    @abstractmethod
    def add(self, vector_records: VectorRecords) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (VectorRecords): Container with list of vector records.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors from the index by their IDs.
        
        Args:
            ids (list[bytes]): List of UUIDv7 IDs to delete.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Search for k nearest neighbors using the configured metric.

        Args:
            query_vec (np.ndarray): The query vector as a 2D numpy array with shape (1, d).
            top_k (int, optional): Number of nearest neighbors to return. Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        pass

    def _bytes_to_uuid_str(self, id_bytes: bytes) -> str:
        """Convert UUIDv7 bytes to UUID string format.
        
        Args:
            id_bytes (bytes): 16-byte UUIDv7.
            
        Returns:
            str: UUID string in standard format.
        """
        return str(UUID(bytes=id_bytes))

    def _perform_sqlite_insert(self, record: VectorRecords) -> None:
        """Insert a single record into SQLite.
        
        Args:
            record: The vector record to insert.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO records (id, content, metadata, embedding, deleted) 
            VALUES (?, ?, jsonb(?), ?, 0)
            """,
            (
                record.id,
                record.content,
                json.dumps(record.metadata),
                record.embedding.tobytes(),
            ),
        )

    def _perform_sqlite_delete(self, ids: list[bytes], soft: bool = True) -> None:
        """Delete records from SQLite.
        
        Args:
            ids: List of record IDs to delete.
            soft: If True, mark as deleted (soft delete). If False, actually remove from DB.
        """
        if soft:
            placeholders = ','.join('?' * len(ids))
            cursor = self.conn.cursor()
            cursor.execute(
                f"UPDATE records SET deleted = 1 WHERE id IN ({placeholders})",
                ids,
            )
        else:
            placeholders = ','.join('?' * len(ids))
            cursor = self.conn.cursor()
            cursor.execute(
                f"DELETE FROM records WHERE id IN ({placeholders})",
                ids,
            )


    def _perform_sqlite_fetch(self, ids: list[bytes], include_vectors: bool = False) -> dict:
        """Fetch records from SQLite by IDs.
        
        Args:
            ids: List of record IDs to fetch.
            include_vectors: If True, include embedding in the results.
            
        Returns:
            dict: Mapping of ID bytes to SQLite row data.
        """
        if not ids:
            return {}
        
        placeholders = ','.join('?' * len(ids))
        sql = (
            f"SELECT id, content, json(metadata) {', embedding ' if include_vectors else ''}"
            f"FROM records WHERE id IN ({placeholders})"
        )
        cursor = self.conn.cursor()
        cursor.execute(sql, ids)
        rows = cursor.fetchall()
        return {row[0]: row for row in rows}

    def _build_results(
        self, 
        ids: list[bytes], 
        scores: np.ndarray, 
        id_to_row: dict,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Build result dictionaries maintaining ranking order.
        
        Args:
            ids (list[bytes]): Ranked list of record IDs.
            scores (np.ndarray): Corresponding distance/similarity scores.
            id_to_row (dict): Mapping of ID bytes to SQLite row data.
            include_vectors (bool): Whether to include embedding in results.
        
        Returns:
            list[dict]: List of result dicts with id, content, metadata, distance, 
                and optionally embedding.
        """
        results = []
        for id_bytes, score in zip(ids, scores):
            row = id_to_row[id_bytes]
            
            result = {
                "id": self._bytes_to_uuid_str(id_bytes),
                "content": row[1],
                "metadata": json.loads(row[2]),
                "distance": float(score),
            }
            
            if include_vectors:
                result["embedding"] = np.frombuffer(row[3], dtype=np.float32)
            
            results.append(result)
        
        return results
