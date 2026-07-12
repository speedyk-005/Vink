from pathlib import Path
from typing import Literal

import numpy as np
from readerwriterlock import rwlock

from vinkra.sql_wrapper import SQLiteWrapper
from vinkra.strategies.base import BaseStrategy
from vinkra.utils.logging import log_info


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
        metric: Literal["euclidean", "cosine"],
        *,
        verbose: bool,
    ) -> None:
        """
        Initialize the ExactSearch.

        Args:
            db: SQLite wrapper for database operations.
            dir_path: Path to store vector data. None for in-memory storage.
            dim: Dimension of the vectors.
            metric: Distance metric to use.
            verbose: Enable verbose output.
        """
        super().__init__(
            db=db,
            dir_path=dir_path,
            dim=dim,
            metric=metric,
            verbose=verbose,
        )

        self._rwlock = rwlock.RWLockFair()

        self._all_vectors: list[np.ndarray] = []
        self._all_ids: list[bytes] = []
        self._id_to_idx: dict[bytes, int] = {}

        # Boolean mask for active/deleted status
        self._mask: list[bool] = []

        # Cache placeholders
        self.active_vectors_arr = None
        self.active_ids_arr = None

    def _ensure_cache(self) -> None:
        """Build cache of active vectors and IDs if not already cached."""
        if not (self.active_vectors_arr is None or self.active_ids_arr is None):
            return

        # Caller holds the lock; no nested lock acquisition.
        active_indices = [i for i, m in enumerate(self._mask) if m]

        if len(active_indices) == 0:
            self.active_vectors_arr = np.empty((0, self.dim), dtype=np.float32)
            self.active_ids_arr = np.empty((0,), dtype="S16")
            return

        self.active_vectors_arr = np.vstack(self._all_vectors)[active_indices]
        self.active_ids_arr = np.array(self._all_ids, dtype="S16")[active_indices]

    def add(self, vector_records: list[dict]) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records: List of dicts with 'id', 'embedding' keys.

        Returns:
            List of assigned UUIDv7 IDs.
        """
        with self._rwlock.gen_wlock():
            assigned_ids = []

            for record in vector_records:
                idx = len(self._all_ids)
                self._all_vectors.append(record["embedding"])
                self._all_ids.append(record["id"])
                self._id_to_idx[record["id"]] = idx
                self._mask.append(True)

                # Invalidate cache
                self.active_vectors_arr = None
                self.active_ids_arr = None

                assigned_ids.append(self._bytes_to_uuid_str(record["id"]))

            self.db.insert(vector_records)

        return assigned_ids

    def soft_delete(self, ids: list[bytes]) -> None:
        """
        Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids: List of UUIDv7 IDs to soft-delete.
        """
        with self._rwlock.gen_wlock():
            for id_bytes in ids:
                idx = self._id_to_idx.get(id_bytes)
                if idx is not None:
                    self._mask[idx] = False

            self.db.soft_delete(ids)

            # Invalidate cache
            self.active_vectors_arr = None
            self.active_ids_arr = None

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
        with self._rwlock.gen_rlock():
            self._ensure_cache()

            if filters:
                where_sql, params = self._filter_to_sql.translate(filters)
                where_sql += " AND deleted = FALSE"
                rows = self.db.fetch(
                    where_sql=where_sql,
                    params=params,
                )
                match_set = {row[0] for row in rows}
                filtered_mask = np.array(
                    [uid in match_set for uid in self.active_ids_arr]
                )

                filtered_vectors = self.active_vectors_arr[filtered_mask]
                filtered_ids = self.active_ids_arr[filtered_mask]
            else:
                # Use cached versions
                filtered_vectors = self.active_vectors_arr
                filtered_ids = self.active_ids_arr

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
        where_sql = f"WHERE id IN ({placeholders})"
        rows = self.db.fetch(
            where_sql=where_sql, params=ids, include_vectors=include_vectors
        )
        id_to_row = {row[0]: row for row in rows}

        return self._build_results(
            ids, scores, id_to_row, include_vectors=include_vectors
        )

    def compact(self) -> None:
        """Hard-delete soft-deleted records and rebuild the index."""
        with self._rwlock.gen_wlock():
            active_indices = [i for i, m in enumerate(self._mask) if m]

            self.active_vectors_arr = np.vstack(self._all_vectors).astype(
                np.float32, copy=False
            )[active_indices]
            self.active_ids_arr = np.array(self._all_ids, dtype="S16")[active_indices]

            self._all_vectors = self.active_vectors_arr.tolist()
            self._all_ids = self.active_ids_arr.tolist()
            self._id_to_idx = {
                id_bytes: idx for idx, id_bytes in enumerate(self._all_ids)
            }
            self._mask = [True] * len(self._all_ids)

            self.db.compact()

    def save(self) -> None:
        """Save the index to disk by committing the database."""
        self.db.commit()

    def load(self, *, overwrite: bool) -> None:
        """Load the index from SQLite.

        Args:
            overwrite: If True, replace in-memory state with loaded data.
        """
        if not overwrite and self._all_ids:
            log_info(self.verbose, "Index already loaded, skipping.")
            return

        if self.db.count("active") == 0:
            return

        with self._rwlock.gen_wlock():
            cursor = self.db.conn.execute(
                "SELECT id, embedding, deleted FROM vec_records"
            )
            rows = cursor.fetchall()

            self._all_ids = [row[0] for row in rows]
            self._all_vectors = np.vstack(
                [np.frombuffer(row[1], dtype=np.float32) for row in rows]
            )
            self._mask = [bool(row[2]) for row in rows]
            self._id_to_idx = {
                id_bytes: idx for idx, id_bytes in enumerate(self._all_ids)
            }

            # Ensure cache is invalidated
            self.active_vectors_arr = None
            self.active_ids_arr = None

    def _cosine_similarity(
        self,
        query_vec: np.ndarray,
        vectors: np.ndarray,
        id_vecs: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query vector and provided vectors.

        Args:
            query_vec: Query vector with shape (1, d).
            vectors: Vectors to compute similarity with, shape (n, d).
            id_vecs: IDs corresponding to each vector, shape (n,).
            top_k: Number of top results to return.

        Returns:
            Top-k IDs and similarity scores,
                ordered by similarity (descending).
        """
        if len(id_vecs) == 0:
            return np.array([]), np.array([])

        # Query and stored vectors are already L2-normalized, so cosine
        # similarity is equivalent to a dot product.
        similarities = (vectors @ query_vec.T).flatten()

        # Avoid sorting every similarity score. Partition only finds the
        # top-k candidates. The candidates are sorted afterward to preserve
        # descending similarity order.
        candidate_indices = np.argpartition(
            similarities,
            -top_k,
        )[-top_k:]

        sorted_indices = candidate_indices[
            np.argsort(similarities[candidate_indices])[::-1]
        ]

        top_ids = [id_vecs[i] for i in sorted_indices]
        top_scores = similarities[sorted_indices]

        return top_ids, top_scores

    def _euclidean_distance(
        self,
        query_vec: np.ndarray,
        vectors: np.ndarray,
        id_vecs: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Euclidean distance between query vector and provided vectors.

        Args:
            query_vec: Query vector with shape (1, d).
            vectors: Vectors to compute distance with, shape (n, d).
            id_vecs: IDs corresponding to each vector, shape (n,).
            top_k: Number of top results to return.

        Returns:
            Top-k IDs and distance scores,
                ordered by distance (ascending).
        """
        if len(id_vecs) == 0:
            return np.array([]), np.array([])

        distances = np.sqrt(np.sum((vectors - query_vec) ** 2, axis=1))

        # Partition finds the nearest top-k candidates without fully sorting
        # all distances. Only the selected candidates are sorted to guarantee
        # ascending distance order.
        candidate_indices = np.argpartition(
            distances,
            top_k - 1,
        )[:top_k]

        sorted_indices = candidate_indices[np.argsort(distances[candidate_indices])]

        top_ids = [id_vecs[i] for i in sorted_indices]
        top_scores = distances[sorted_indices]

        return top_ids, top_scores
