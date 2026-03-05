from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal
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
            metric (Literal["l2", "dot"], optional): Distance metric to use.
                Defaults to "l2".
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

    def _bytes_to_uuid_str(self, id_bytes: bytes) -> str:
        """Convert UUIDv7 bytes to UUID string format.
        
        Args:
            id_bytes (bytes): 16-byte UUIDv7.
            
        Returns:
            str: UUID string in standard format.
        """
        return str(UUID(bytes=id_bytes))

    @abstractmethod
    def add(self, vector_records: VectorRecords) -> list[str]:
        """
        Add vectors to the index.

        Args:
            vector_records (VectorRecords): Container with list of vector records.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors from the index by their IDs.

        Args:
            ids (list[str]): List of UUIDv7 IDs to delete.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
    ) -> list[dict]:
        """
        Search for k nearest neighbors.

        Args:
            query (np.ndarray): The query vector as a 2D numpy array with shape (1, d).
            top_k (int, optional): Number of nearest neighbors to return.
                Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        pass
