import os
import pickle
from pathlib import Path
from threading import Thread
from typing import Literal

import nanopq
import numpy as np
import rii
from readerwriterlock import rwlock

from vinkra.exceptions import (
    DatabaseCorruptedError,
    IndexNotFittedError,
    InvalidInputError,
)
from vinkra.models import AnnConfig
from vinkra.sql_wrapper import SQLiteWrapper
from vinkra.strategies.base import BaseStrategy
from vinkra.utils.logging import log_info, logger


class ApproximateSearch(BaseStrategy):
    """
    Approximate search strategy using Product Quantization (PQ) or Optimized PQ.

    This strategy compresses high-dimensional vectors into compact codes using
    subspace quantization. It provides significantly faster search performance
    and reduced memory usage compared to exact methods, making it suitable
    for large-scale vector datasets where sub-millisecond latency is required.

    Note:
        This strategy relies on codebooks generated during the 'fit'
        initialization step. Precision is subject to the number of subspaces
        and the quantization method employed.
    """

    NANOPQ_METRIC_MAP = {"euclidean": "l2", "cosine": "dot"}

    def __init__(
        self,
        db: SQLiteWrapper,
        dir_path: Path | None,
        dim: int,
        metric: Literal["euclidean", "cosine"],
        *,
        verbose: bool,
        ann_config: AnnConfig,
    ) -> None:
        """
        Initialize the ApproximateSearch.

        Args:
            db: SQLite wrapper for database operations.
            dir_path: Path to store vector data. None for in-memory storage.
            dim: Dimension of the vectors.
            metric: Distance metric to use.
            verbose: Enable verbose output.
            ann_config: ANN configuration.
        """
        super().__init__(
            db=db,
            dir_path=dir_path,
            dim=dim,
            metric=metric,
            verbose=verbose,
        )
        self._ann_config = ann_config

        self._rwlock = rwlock.RWLockFair()
        self._delta_since_reconfig = 0
        self.is_reconfig = False
        self.is_compacting = False

        self.index: rii.Rii | None = None

        self._all_ids: list[bytes] = []
        self._id_to_idx: dict[bytes, int] = {}

        # Boolean mask for active/deleted status
        self._mask: list[bool] = []

        # Cache placeholder
        self.active_ids_arr = None

        if self.dir_path is not None:
            self._ann_index_path = self.dir_path / "ann_index.pkl"
            self._ann_shadow_index_path = self.dir_path / "ann_index.pkl.tmp"

    def fit(
        self,
        vectors: np.ndarray,
        active_ids_arr: np.ndarray,
    ) -> None:
        """
        Initialize the Approximate Search index by training the Quantizer.

        It processes all currently indexed vectors to generate the subspace codebooks
        required for approximate search.
        The quantizer is trained with randomly sampled vectors.

        Args:
            vectors: A 2D array of shape (N, D) representing the N vectors
                of dimensionality D to be indexed.
            active_ids_arr: Array of active IDs corresponding to the vectors.
        """
        log_info(self.verbose, "Starting ANN index fit with {} vectors.", len(vectors))

        if self._ann_config.codebook_size >= len(vectors):
            raise InvalidInputError(
                f"Codebook size ({self._ann_config.codebook_size}) must be less than "
                f"the number of training vectors ({len(vectors)}). "
                "This constraint is required by Product Quantization."
            )

        self.active_ids_arr = active_ids_arr
        self._all_ids = list(active_ids_arr)
        self._id_to_idx = {id_bytes: idx for idx, id_bytes in enumerate(self._all_ids)}

        # Initialize mask - all True since these are active
        self._mask = [True] * len(self._all_ids)

        self.reconfig_threshold = self._ann_config.reconfig_threshold

        # Sample training vectors for codec training
        n_vecs = len(vectors)
        ks = self._ann_config.codebook_size
        max_train_size = min(n_vecs, max(ks * 10, 5000))
        if n_vecs > max_train_size:
            rng = np.random.default_rng()
            train_indices = rng.choice(n_vecs, size=max_train_size, replace=False)
            train_vectors = vectors[train_indices]
        else:
            train_vectors = vectors

        log_info(
            self.verbose,
            "Training codec with {} vectors using {} subspaces and codebook_size {}.",
            len(train_vectors),
            self._ann_config.num_subspaces,
            self._ann_config.codebook_size,
        )

        pq_class = nanopq.PQ if self._ann_config.quantizer == "pq" else nanopq.OPQ
        codec = pq_class(
            M=self._ann_config.num_subspaces,
            Ks=self._ann_config.codebook_size,
            metric=self.NANOPQ_METRIC_MAP[self.metric],
            verbose=False,
        )

        # minit="points" is fast since training vectors are already randomly sampled.
        codec.fit(train_vectors, minit="points")

        # Initialize Rii with the trained codec
        self.index = rii.Rii(fine_quantizer=codec)
        self.index.add_configure(vecs=vectors)

        log_info(self.verbose, "ANN index fit completed successfully.")

    def _validate_fitted(self) -> None:
        """
        Validates the index state.

        Raises:
            IndexNotFittedError: If the index has not been fitted yet.
        """
        if self.index is None:
            raise IndexNotFittedError(
                "The index parameters are not yet learned. Please run .fit() "
                "on your training vectors before performing any index operations."
            )

    def _swap_index(self) -> None:
        """Replace shadow index with main index and fsync the directory."""
        self._ann_shadow_index_path.replace(self._ann_index_path)
        dir_fd = os.open(str(self.dir_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _do_reconfigure(self) -> None:
        """Run index.reconfigure() in background thread."""
        log_info(self.verbose, "ANN index reconfiguration in progress...")

        self.index.reconfigure()
        self._delta_since_reconfig = 0
        self.is_reconfig = False

        log_info(self.verbose, "ANN index reconfigured.")

    def _ensure_cache(self) -> None:
        """Build cache of IDs if not already cached."""
        if self.active_ids_arr is not None:
            return

        # Caller holds the lock; no nested lock acquisition.
        active_indices = [i for i, m in enumerate(self._mask) if m]

        if len(active_indices) == 0:
            self.active_ids_arr = np.empty((0,), dtype="S16")
            return

        self.active_ids_arr = np.array(self._all_ids, dtype="S16")[active_indices]

    def add(self, vector_records: list[dict], *, is_buffer: bool = False) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records: List of dicts with 'id', 'embedding' keys.

        Returns:
            List of assigned UUIDv7 IDs.

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()

        with self._rwlock.gen_wlock():
            assigned_ids = []
            embeddings = []

            for record in vector_records:
                idx = len(self._all_ids)
                self._all_ids.append(record["id"])
                self._id_to_idx[record["id"]] = idx
                self._mask.append(True)
                self.active_ids_arr = None
                embeddings.append(record["embedding"])
                assigned_ids.append(self._bytes_to_uuid_str(record["id"]))

        self.index.add(np.vstack(embeddings))

        if not is_buffer:
            self.db.insert(vector_records)

        self._delta_since_reconfig += len(vector_records)
        if (
            not (self.is_reconfig or self.is_compacting)
            and self._delta_since_reconfig >= self.reconfig_threshold
        ):
            self.is_reconfig = True
            thread = Thread(target=self._do_reconfigure, daemon=True)
            thread.start()

        return assigned_ids

    def soft_delete(self, ids: list[bytes]) -> None:
        """
        Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids: List of UUIDv7 IDs to soft-delete.

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()

        with self._rwlock.gen_wlock():
            for id_bytes in ids:
                idx = self._id_to_idx.get(id_bytes)
                if idx is not None:
                    self._mask[idx] = False

            self.db.soft_delete(ids)

            # Invalidate cache
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

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()
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

                filtered_ids = self.active_ids_arr[filtered_mask]
            else:
                # Use cached versions
                filtered_ids = self.active_ids_arr

            if self.is_reconfig:
                # Use linear scan during reconfiguration to avoid inconsistent results
                # from inverted index being updated in background. Linear is still fast
                # since it uses ADist on PQ-coded vectors (M lookups, not full vectors).
                ids, scores = self._query_index(
                    query_vec, filtered_ids, top_k, method="linear"
                )
            else:
                ids, scores = self._query_index(query_vec, filtered_ids, top_k)

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
        self.is_compacting = True

        with self._rwlock.gen_wlock():
            active_indices = [i for i, m in enumerate(self._mask) if m]
            if len(active_indices) <= self._ann_config.codebook_size:
                logger.warning(
                    "Skipping ANN index rebuild: only {} active vectors, "
                    "codebook_size {} requires strictly fewer.",
                    len(active_indices),
                    self._ann_config.codebook_size,
                )
                return

            self.db.compact()

            self.active_ids_arr = np.array(self._all_ids, dtype="S16")[active_indices]
            self._all_ids = self.active_ids_arr.tolist()
            self._id_to_idx = {
                id_bytes: idx for idx, id_bytes in enumerate(self._all_ids)
            }
            self._mask = [True] * len(self._all_ids)

            gen = self.db.iter_embeddings()
            first_batch = next(gen, None)
            if first_batch:
                embeddings = np.vstack(
                    [np.frombuffer(vecs, dtype=np.float32) for vecs in first_batch]
                )
                self.fit(embeddings, self.active_ids_arr)

            for batch in gen:
                embeddings = np.vstack(
                    [np.frombuffer(vecs, dtype=np.float32) for vecs in batch]
                )
                self.index.add(embeddings)

        self.is_compacting = False

    def save(self) -> None:
        """Save the index to disk using double-write strategy for tight syncing."""
        self._validate_fitted()

        with self._ann_shadow_index_path.open("wb") as f:
            pickle.dump(self.index, f, protocol=5)
            f.flush()  # Flush internal buffer
            os.fsync(f.fileno())  # Force OS to write to physical disk

        self.db.commit()

        self._swap_index()

    def load(self, *, overwrite: bool) -> None:
        """Load the index from disk.

        Args:
            overwrite: If True, replace in-memory state with loaded data.
        """
        if not (self._ann_index_path and self._ann_index_path.exists()):
            log_info(self.verbose, "No ANN index file found, skipping index load.")
            return

        if not overwrite and self._all_ids:
            log_info(self.verbose, "Index already loaded, skipping.")
            return

        if self.db.is_empty():
            return

        with self._rwlock.gen_wlock():
            cursor = self.db.conn.execute("SELECT id, deleted FROM vec_records")
            rows = cursor.fetchall()

            self._all_ids = [row[0] for row in rows]
            self._mask = [bool(row[1]) for row in rows]
            self._id_to_idx = {
                id_bytes: idx for idx, id_bytes in enumerate(self._all_ids)
            }

            self._safe_load_ann_index()

            # Ensure cache is invalidated
            self.active_ids_arr = None

    def _safe_load_ann_index(self):
        """Safely load the ann index with recovering step in case of desyncronisation"""

        def load_index(path):
            with path.open("rb") as f:
                index = pickle.load(f)

            if len(self._all_ids) != index.N:
                raise DatabaseCorruptedError(
                    "ANN index and database records are out of sync"
                )

            return index

        try:
            self.index = load_index(self._ann_index_path)
            return
        except (
            FileNotFoundError,
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            DatabaseCorruptedError,
        ):
            # Recover from partial save
            log_info(
                self.verbose, "Partial save detected... Recovering from backup file"
            )

        try:
            self.index = load_index(self._ann_shadow_index_path)
            self._swap_index()

        except FileNotFoundError as e:
            raise DatabaseCorruptedError(
                "Index recovery failed - backup file not found"
            ) from e

        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            DatabaseCorruptedError,
        ) as e:
            self._ann_shadow_index_path.unlink(missing_ok=True)
            raise DatabaseCorruptedError(
                "Index recovery failed - both index and backup files are corrupted"
            ) from e

    def _query_index(
        self,
        query_vec: np.ndarray,
        id_vecs: np.ndarray,
        top_k: int,
    ) -> tuple[list[bytes], np.ndarray]:
        """
        Search for nearest neighbors to the query vector via the Rii engine.

        Args:
            query_vec: Query vector with shape (1, d).
            id_vecs: IDs corresponding to each vector, shape (n,).
            top_k: Number of top results to return.

        Returns:
            Top-k IDs and distance scores, ordered by distance
                (ascending, closest first).
        """
        # Ensure query vector is 1D (rii expects 1D array)
        if query_vec.ndim == 2:
            query_vec = query_vec.flatten()

        # Map application IDs to internal index offsets
        target_indices = np.array(
            [self._id_to_idx[uid] for uid in id_vecs if uid in self._id_to_idx]
        )

        if len(target_indices) == 0:
            return [], np.array([])

        indices, top_scores = self.index.query(
            query_vec,
            topk=top_k,  # rii index uses topk instead of top_k
            target_ids=target_indices,
        )
        top_ids = [self._all_ids[i] for i in indices]

        if self.metric == "cosine":
            # Since rii uses ascending order
            top_scores = 1 - (top_scores / 2)

        return top_ids, top_scores
