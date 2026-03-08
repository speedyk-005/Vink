import json
from pathlib import Path
from threading import Lock
from typing import Literal
import numpy as np

import pysqlite3 as sqlite3

from vink.strategies.base import BaseStrategy


class ExactSearch(BaseStrategy):
    """
    Exact search strategy using brute-force distance computation.

    This strategy computes exact nearest neighbors by calculating distances
    to all stored vectors. It is suitable for smaller datasets or when
    maximum recall is required.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        dir_path: Path | None,
        dim: int,
        in_memory: bool,
        metric: Literal["l2", "dot"] = "l2",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ExactSearch.

        Args:
            conn (sqlite3.Connection): SQLite database connection for storing records.
            dir_path (Path | None): Path to store vector data. Defaults to None.
            dim (int): Dimension of the vectors.
            in_memory (bool): Whether using in-memory storage.
            metric (Literal["l2", "dot"], optional): Distance metric to use. Defaults to "l2".
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        super().__init__(
            conn=conn,
            dir_path=dir_path,
            dim=dim,
            is_exact=True,
            in_memory=in_memory,
            metric=metric,
            verbose=verbose,
        )

        # Lock for thread-safe add/delete operations
        self._lock = Lock()
        
        self.vectors: list[np.ndarray] = []
        self.ids: list[bytes] = []
        self.id_to_idx: dict[bytes, int] = {}  # Fast O(1) lookup for deletion
        
        # Boolean mask for active/deleted status
        self.mask: np.ndarray = np.array([], dtype=bool)

        # Cache placeholders
        self.active_vectors: np.ndarray | None = None
        self.active_ids: np.ndarray | None = None

    def _ensure_cache(self) -> None:
        """Build cache of active vectors and IDs if not already cached."""
        if not (self.active_vectors is None or self.active_ids is None):
            return

        with self._lock:
            # Get indices of non-deleted items (True in self.mask)
            active_indices = self.mask.nonzero()[0]
        
            if len(active_indices) == 0:
                self.active_vectors = np.empty((0, self.dim), dtype=np.float32)
                self.active_ids = np.empty((0,), dtype='S16')
                return

            self.active_vectors = np.vstack(self.vectors).astype(np.float32, copy=False)[active_indices]
            self.active_ids = np.array(self.ids, dtype='S16')[active_indices]

    def add(self, vector_records) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (VectorRecords): Container with list of vector records.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.
        """
        with self._lock:
            assigned_ids = []
            cursor = self.conn.cursor()
            
            for record in vector_records.records:
                # Extend in-memory storage
                idx = len(self.ids)  # Current index
                self.vectors.append(record.embedding)
                self.ids.append(record.id)
                self.id_to_idx[record.id] = idx  # Fast O(1) lookup for deletion
                self.mask = np.append(self.mask, True)
                
                # Invalidate cache
                self.active_vectors = None
                self.active_ids = None
                
                # Insert into SQLite
                self._perform_sqlite_insert(record)
                
                assigned_ids.append(self._bytes_to_uuid_str(record.id))
            
            self.conn.commit()
        
        return assigned_ids

    def delete(self, ids: list[bytes]) -> None:
        """
        Delete vectors from the index by their IDs.
        
        Args:
            ids (list[bytes]): List of UUIDv7 IDs to delete.
        """
        with self._lock:
            cursor = self.conn.cursor()
            
            for id_bytes in ids:
                idx = self.id_to_idx.get(id_bytes)
                if idx is not None:
                    # Mark as deleted in mask
                    self.mask[idx] = False
            
            # Delete in SQLite
            self._perform_sqlite_delete(ids, soft=True)
            
            self.conn.commit()
            
            # Invalidate cache
            self.active_vectors = None
            self.active_ids = None

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
        self._ensure_cache()

        filtered_vectors = self.active_vectors
        filtered_ids = self.active_ids

        # TODO: support metadata filtering
        """
        # Query SQLite for active IDs (structure for metadata filtering later)
        if metadata_filter_exists:  # (Future capability)
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM records WHERE ... AND deleted = 0")
            match_set = {row[0] for row in cursor.fetchall()}
            
            # Create the mask against your ACTIVE (cached) IDs
            # This keeps the indices perfectly aligned with your active_vectors matrix
            temp_mask = np.array([uid in match_set for uid in self.active_ids])

            filtered_vectors = self.active_vectors[temp_mask]
            filtered_ids = self.active_ids[temp_mask] 
        else:
            # Use cached versions
            filtered_vectors = self.active_vectors
            filtered_ids = self.active_ids
        """

        if self.metric == "dot":
            ids, scores = self._cosine_similarity(query_vec, filtered_vectors, filtered_ids, top_k)
        else:  # "l2/euclidian"
            ids, scores = self._euclidian_distance(query_vec, filtered_vectors, filtered_ids, top_k)
        
        if not ids:
            return []
        
        # Re-query SQLite for full records of top_k IDs
        id_to_row = self._perform_sqlite_fetch(ids, include_vectors)
        
        return self._build_results(ids, scores, id_to_row, include_vectors)

    def _cosine_similarity(
        self,
        query_vec: np.ndarray,
        vectors: np.ndarray,
        ids: list[bytes],
        top_k: int,
    ) -> tuple[list[bytes], np.ndarray]:
        """
        Compute cosine similarity between query vector and provided vectors.

        Args:
            query_vec (np.ndarray): Query vector with shape (1, d).
            vectors (np.ndarray): Vectors to compute similarity with, shape (n, d).
            ids (list[bytes]): Corresponding IDs for each vector.
            top_k (int): Number of top results to return.

        Returns:
            tuple[list[bytes], np.ndarray]: Top-k IDs and similarity scores,
                ordered by similarity (descending).
        """
        if not ids.any():
            return [], np.array([])

        # Use dot product since both query and vectors are already L2-normalized
        similarities = (vectors @ query_vec.T).flatten()
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        top_ids = [ids[i] for i in sorted_indices]
        top_scores = similarities[sorted_indices]
        
        return top_ids, top_scores

    def _euclidian_distance(
        self,
        query_vec: np.ndarray,
        vectors: np.ndarray,
        ids: list[bytes],
        top_k: int,
    ) -> tuple[list[bytes], np.ndarray]:
        """
        Compute Euclidean distance between query vector and provided vectors.

        Args:
            query_vec (np.ndarray): Query vector with shape (1, d).
            vectors (np.ndarray): Vectors to compute distance with, shape (n, d).
            ids (list[bytes]): Corresponding IDs for each vector.
            top_k (int): Number of top results to return.

        Returns:
            tuple[list[bytes], np.ndarray]: Top-k IDs and distance scores, ordered by distance 
                (ascending, closest first).
        """
        if not ids.any():
            return [], np.array([])

        # Compute L2 distance: sqrt((x - y)^2)
        # Since vectors are normalized: ||x - y||^2 = 2 - 2*(x·y)
        similarities = (vectors @ query_vec.T).flatten()
        distances = np.sqrt((2 - 2 * similarities).clip(min=0))  # Clip for numerical stability
        
        # Sort by distance in ascending order (closest first)
        indices = np.argsort(distances)[:top_k]
        top_ids = [ids[i] for i in indices]
        top_scores = distances[indices]
        
        return top_ids, top_scores
