import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np

from vinkra.filter_parser import FilterToSql
from vinkra.sql_wrapper import SQLiteWrapper


class BaseStrategy(ABC):
    """
    Base class for search strategies.

    Provides abstract interface for exact and approximate search implementations.
    """

    def __init__(
        self,
        db: SQLiteWrapper,
        dir_path: Path | None,
        dim: int,
        metric: Literal["euclidean", "cosine"],
        *,
        verbose: bool,
    ) -> None:
        """
        Initialize the strategy.

        Args:
            db: SQLite wrapper for database operations.
            dir_path: Path to store vector data. None for in-memory storage.
            dim: Dimension of the vectors.
            metric: Distance metric to use.
            verbose: Enable verbose output.
        """
        self.db = db
        self.dir_path = dir_path
        self.dim = dim
        self.metric = metric
        self.verbose = verbose
        self._filter_to_sql = FilterToSql()

    @abstractmethod
    def add(self, vector_records: list[dict]) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records: List of dict records with 'id', 'content',
                'metadata', and 'embedding' keys.

        Returns:
            List of assigned UUIDv7 IDs.
        """
        ...

    @abstractmethod
    def soft_delete(self, ids: list[str]) -> None:
        """Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids: List of UUIDv7 IDs to soft-delete.
        """
        ...

    @abstractmethod
    def compact(self) -> None:
        """Hard-delete soft-deleted records and rebuild the index."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Save the index to disk."""
        ...

    @abstractmethod
    def load(self, *, overwrite: bool) -> None:
        """Load the index from disk.

        Args:
            overwrite: If True, replace in-memory state with loaded data.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        *,
        include_vectors: bool = False,
        filters: list[str] | None = None,
    ) -> list[dict]:
        """Search for k nearest neighbors using the configured metric.

        Args:
            query_vec: The query vector as a 2D numpy array with shape (1, d).
            top_k: Number of nearest neighbors to return. Defaults to 10.
            include_vectors: If True, include 'embedding' key in results.
                Defaults to False.
            filters: Filter expressions to apply before scoring.

        Returns:
            List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        ...

    def _bytes_to_uuid_str(self, id: str | bytes) -> str:
        """Convert UUIDv7 bytes to UUID string format."""
        if isinstance(id, bytes):
            return str(UUID(bytes=id))
        return id

    def _build_results(
        self,
        ids: list[bytes],
        scores: np.ndarray,
        id_to_row: dict,
        *,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Build result dictionaries maintaining ranking order.

        Args:
            ids: Ranked list of record IDs.
            scores: Corresponding distance/similarity scores.
            id_to_row: Mapping of ID bytes to SQLite row data.
            include_vectors: Whether to include embedding in results.

        Returns:
            List of result dicts with id, content, metadata, distance,
                and optionally embedding.
        """
        results = []
        for id_bytes, score in zip(ids, scores, strict=True):
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
