from pathlib import Path
from readerwriterlock import rwlock
from typing import Literal

import numpy as np

from vink.sql_wrapper import SQLiteWrapper
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
        db: SQLiteWrapper,
        dir_path: Path | None,
        dim: int,
        in_memory: bool,
        metric: Literal["euclidean", "cosine"],
        verbose: bool,
    ) -> None:
        """
        Initialize the ExactSearch.

        Args:
            db (SQLiteWrapper): SQLite wrapper for database operations.
            dir_path (Path | None): Path to store vector data. Defaults to None.
            dim (int): Dimension of the vectors.
            in_memory (bool): Whether using in-memory storage.
            metric (Literal["euclidean", "cosine"]): Distance metric to use.
            verbose (bool): Enable verbose output.
        """
        super().__init__(
            db=db,
            dir_path=dir_path,
            dim=dim,
            is_exact=True,
            in_memory=in_memory,
            metric=metric,
            verbose=verbose,
        )

        self._rwlock = rwlock.RWLockFair()

        self.all_vectors: list[np.ndarray] = []
        self.all_ids: list[bytes] = []
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

        # Caller holds the lock; no nested lock acquisition.
        active_indices = self.mask.nonzero()[0]

        if len(active_indices) == 0:
            self.active_vectors = np.empty((0, self.dim), dtype=np.float32)
            self.active_ids = np.empty((0,), dtype="S16")
            return

        self.active_vectors = np.vstack(self.all_vectors).astype(
            np.float32, copy=False
        )[active_indices]
        self.active_ids = np.array(self.all_ids, dtype="S16")[active_indices]

    def add(self, vector_records, is_buffer: bool = False) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (VectorRecords): Container with list of vector records.
            is_buffer (bool): If True, records are already in SQLite. Defaults to False.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.
        """
        with self._rwlock.gen_wlock():
            assigned_ids = []

            for record in vector_records.records:
                idx = len(self.all_ids)
                self.all_vectors.append(record.embedding)
                self.all_ids.append(record.id)
                self.id_to_idx[record.id] = idx
                self.mask = np.append(self.mask, True)

                # Invalidate cache
                self.active_vectors = None
                self.active_ids = None

                assigned_ids.append(self._bytes_to_uuid_str(record.id))

            if not is_buffer:
                self.db.insert(vector_records)

        return assigned_ids

    def soft_delete(self, ids: list[bytes]) -> None:
        """
        Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids (list[bytes]): List of UUIDv7 IDs to soft-delete.
        """
        with self._rwlock.gen_wlock():
            for id_bytes in ids:
                idx = self.id_to_idx.pop(id_bytes, None)
                if idx is not None:
                    self.mask[idx] = False

            self.db.soft_delete(ids)

            # Invalidate cache
            self.active_vectors = None
            self.active_ids = None

    def compact(self) -> None:
        """Hard-delete soft-deleted records and rebuild the index."""
        with self._rwlock.gen_wlock():
            active_indices = self.mask.nonzero()[0]

            self.active_vectors = np.vstack(self.all_vectors).astype(
                np.float32, copy=False
            )[active_indices]
            self.active_ids = np.array(self.all_ids, dtype="S16")[active_indices]

            self.all_vectors = self.active_vectors.tolist()
            self.all_ids = self.active_ids.tolist()
            self.mask = np.ones(len(self.all_ids), dtype=bool)

            self.db.compact()

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
        with self._rwlock.gen_rlock():
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

        if self.metric == "cosine":
            ids, scores = self._cosine_similarity(
                query_vec, filtered_vectors, filtered_ids, top_k
            )
        else:
            ids, scores = self._euclidean_distance(
                query_vec, filtered_vectors, filtered_ids, top_k
            )

        if not ids:
            return []

        # Query SQLite for full records of top_k IDs
        placeholders = ",".join("?" * len(ids))
        where = f"id IN ({placeholders})"
        rows = self.db.fetch(where=where, params=ids, include_vectors=include_vectors)
        id_to_row = {row[0]: row for row in rows}

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

    def _euclidean_distance(
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
        distances = np.sqrt(
            (2 - 2 * similarities).clip(min=0)
        )  # Clip for numerical stability

        # Sort by distance in ascending order (closest first)
        indices = np.argsort(distances)[:top_k]
        top_ids = [ids[i] for i in indices]
        top_scores = distances[indices]

        return top_ids, top_scores
