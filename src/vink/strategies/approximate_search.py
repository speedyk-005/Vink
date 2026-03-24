import os
import random
from pathlib import Path
from typing import Literal

from loguru import logger
from readerwriterlock import rwlock
import larch.pickle as pickle
import nanopq
import numpy as np
import rii

from vink.exceptions import IndexNotFittedError, InvalidInputError
from vink.models import AnnConfig
from vink.filter_parser import FilterToSql
from vink.sql_wrapper import SQLiteWrapper
from vink.strategies.base import BaseStrategy
from vink.utils.logging import log_info


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
        in_memory: bool,
        metric: Literal["euclidean", "cosine"],
        verbose: bool,
        ann_config: AnnConfig,
    ) -> None:
        """
        Initialize the ApproximateSearch.

        Args:
            db (SQLiteWrapper): SQLite wrapper for database operations.
            dir_path (Path | None): Path to store vector data. Defaults to None.
            dim (int): Dimension of the vectors.
            in_memory (bool): Whether using in-memory storage.
            metric (Literal["euclidean", "dot"]): Distance metric to use.
            verbose (bool): Enable verbose output.
            ann_config (AnnConfig): ANN configuration.
        """
        super().__init__(
            db=db,
            dir_path=dir_path,
            dim=dim,
            is_exact=False,
            in_memory=in_memory,
            metric=metric,
            verbose=verbose,
        )
        self.metric = self.NANOPQ_METRIC_MAP[metric]

        self._rwlock = rwlock.RWLockFair()
        self._filter_to_sql = FilterToSql()

        self.index: rii.Rii | None = None
        self._delta_since_reconfig = 0
        self._ann_config = ann_config

        self.all_ids: list[bytes] = []
        self.id_to_idx: dict[bytes, int] = {}

        # Boolean mask for active/deleted status
        self.mask: np.ndarray = np.array([], dtype=bool)

        # Cache placeholder
        self.active_ids_arr = None

        # File paths for save/load
        if self.dir_path is not None:
            self._ann_index_path = self.dir_path / "ann_index.pkl"
            self._ann_index_temp_path = self.dir_path / "ann_index.pkl.tmp"
        else:
            self._ann_index_path = None
            self._ann_index_temp_path = None

    def fit(
        self,
        vectors: np.ndarray,
        active_ids_arr: np.ndarray,
    ) -> None:
        """
        Initialize the Approximate Search index by training the Quantizer.

        It processes all currently indexed vectors to generate the subspace codebooks
        required for approximate search.
        The quantizer is initialized with K-means++ ('++') to ensure robust initialization
        of codebooks across the feature space, improving clustering stability
        and reconstruction accuracy.

        Args:
            vectors (np.ndarray): A 2D array of shape (N, D) representing the N vectors
                of dimensionality D to be indexed.
            active_ids_arr (np.ndarray): Array of active IDs corresponding to the vectors.
        """
        log_info(self.verbose, "Starting ANN index fit with {} vectors.", len(vectors))

        if self._ann_config.codebook_size >= len(vectors):
            raise InvalidInputError(
                f"Codebook size ({self._ann_config.codebook_size}) must be strictly less than "
                f"the number of training vectors ({len(vectors)}). "
                "This constraint is required by Product Quantization."
            )

        self.active_ids_arr = active_ids_arr
        self.all_ids = list(active_ids_arr)
        self.id_to_idx = {id_bytes: idx for idx, id_bytes in enumerate(self.all_ids)}

        # Initialize mask - all True since these are active
        self.mask = np.ones(len(self.all_ids), dtype=bool)

        self.reconfig_threshold = self._ann_config.reconfig_threshold

        # Sample subset for training for codec training
        max_train_size = 1000
        n_vectors = len(vectors)
        if n_vectors > max_train_size:
            rng = np.random.default_rng()
            train_indices = rng.choice(n_vectors, size=max_train_size, replace=False)
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

        codec_params = {
            "M": self._ann_config.num_subspaces,
            "Ks": self._ann_config.codebook_size,
            "metric": self.metric,
            "verbose": False,
        }

        if self._ann_config.quantizer == "opq":
            codec = nanopq.OPQ(**codec_params)

            # Use parametric_init=True for rotation optimization. Since our vectors
            # are normalized, this aligns subspaces with principal components.
            codec.fit(train_vectors, parametric_init=True, minit="++")
            indexed_vecs = codec.rotate(vectors)
        else:
            codec = nanopq.PQ(**codec_params)
            codec.fit(train_vectors, minit="++")
            indexed_vecs = vectors

        # Initialize Rii with the trained codec
        self.index = rii.Rii(fine_quantizer=codec)
        self.index.add_configure(vecs=indexed_vecs)

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

    def _do_reconfigure(self) -> None:
        log_info(self.verbose, "ANN index reconfiguration in progress (this may take a moment)...")
        self.index.reconfigure()
        self._delta_since_reconfig = 0
        log_info(self.verbose, "ANN index reconfigured.")

    def _ensure_cache(self) -> None:
        """Build cache of IDs if not already cached."""
        if self.active_ids_arr is not None:
            return

        # Caller holds the lock; no nested lock acquisition.
        active_indices = self.mask.nonzero()[0]

        if len(active_indices) == 0:
            self.active_ids_arr = np.empty((0,), dtype="S16")
            return

        self.active_ids_arr = np.array(self.all_ids, dtype="S16")[active_indices]

    def add(self, vector_records, is_buffer: bool = False) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (VectorRecords): Container with list of vector records.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()

        with self._rwlock.gen_wlock():
            assigned_ids = []
            embeddings = []

            for record in vector_records.records:
                idx = len(self.all_ids)
                self.all_ids.append(record.id)
                self.id_to_idx[record.id] = idx
                self.mask = np.append(self.mask, True)
                self.active_ids_arr = None
                embeddings.append(record.embedding)
                assigned_ids.append(self._bytes_to_uuid_str(record.id))

        self.index.add(np.vstack(embeddings))
        if not is_buffer:
            self.db.insert(vector_records)
            self.db.commit()

        self._delta_since_reconfig += len(vector_records.records)
        if self._delta_since_reconfig >= self.reconfig_threshold:
            self._do_reconfigure()

        return assigned_ids

    def soft_delete(self, ids: list[bytes]) -> None:
        """
        Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids (list[bytes]): List of UUIDv7 IDs to soft-delete.

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()

        with self._rwlock.gen_wlock():
            for id_bytes in ids:
                idx = self.id_to_idx.pop(id_bytes, None)
                if idx is not None:
                    self.mask[idx] = False

            self.db.soft_delete(ids)
            self.db.commit()

            # Invalidate cache
            self.active_ids_arr = None

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
        filters: list[str] | None = None,
    ) -> list[dict]:
        """Search for k nearest neighbors using the configured metric.

        Args:
            query_vec (np.ndarray): The query vector as a 2D numpy array with shape (1, d).
            top_k (int, optional): Number of nearest neighbors to return. Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.
            filters (list[str] | None, optional): Filter expressions to apply before scoring.

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()
        with self._rwlock.gen_rlock():
            self._ensure_cache()

            if filters:
                where_clause, params = self._filter_to_sql.translate(filters)
                rows = self.db.fetch(
                    where=f"{where_clause} AND deleted = FALSE",
                    params=params,
                )
                match_set = {row[0] for row in rows}
                temp_mask = np.array([uid in match_set for uid in self.active_ids_arr])

                filtered_ids = self.active_ids_arr[temp_mask]
            else:
                # Use cached versions
                filtered_ids = self.active_ids_arr

            ids, scores = self._query_index(query_vec, filtered_ids, top_k)

        if not ids:
            return []

        # Query SQLite for full records of top_k IDs
        placeholders = ",".join("?" * len(ids))
        where = f"id IN ({placeholders})"
        rows = self.db.fetch(where=where, params=ids, include_vectors=include_vectors)
        id_to_row = {row[0]: row for row in rows}

        return self._build_results(ids, scores, id_to_row, include_vectors)

    def compact(self) -> None:
        """Hard-delete soft-deleted records and rebuild the index."""
        with self._rwlock.gen_wlock():
            active_indices = self.mask.nonzero()[0]
            if len(active_indices) <= self._ann_config.codebook_size:
                logger.warning(
                    "Skipping ANN index rebuild: only {} active vectors, "
                    "codebook_size {} requires strictly fewer.",
                    len(active_indices),
                    self._ann_config.codebook_size,
                )
                return

            self.db.compact()

            self.active_ids_arr = np.array(self.all_ids, dtype="S16")[active_indices]
            self.all_ids = self.active_ids_arr.tolist()
            self.mask = np.ones(len(self.all_ids), dtype=bool)

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

    def save(self) -> None:
        """Save the index to disk using double-write strategy for tight syncing."""
        self._validate_fitted()

        with open(self._ann_index_temp_path, "wb") as f:
            pickle.dump(self.index, f, protocol=5)
            os.fsync(f.fileno())  # Force OS to write to physical disk

        self.db.commit()

        self._ann_index_temp_path.replace(self._ann_index_path)

    def load(self, overwrite: bool) -> None:
        """Load the index from disk.

        Args:
            overwrite (bool): If True, replace in-memory state with loaded data.
        """
        if not (self._ann_index_path and self._ann_index_path.exists()):
            log_info(self.verbose, "No ANN index file found, skipping index load.")
            return
            
        if not overwrite and self.all_ids:
            log_info(self.verbose, "Index already loaded, skipping.")
            return

        if self.db.count() == 0:
            return

        with self._rwlock.gen_wlock():
            cursor = self.db.conn.execute("SELECT id, deleted FROM vec_records")
            rows = cursor.fetchall()

            self.all_ids = [row[0] for row in rows]
            self.mask = np.array([row[1] for row in rows], dtype=bool)
            self.id_to_idx = {id_bytes: idx for idx, id_bytes in enumerate(self.all_ids)}

            with open(self._ann_index_path, "rb") as f:
                self.index = pickle.load(f)

            # Recover from partial save (rare)
            if len(self.all_ids) > self.index.N and self._ann_index_temp_path.exists():
                log_info(
                    self.verbose,
                    "Partial save detected: DB has {} IDs but index has {} vectors. "
                    "Recovering from temp file.",
                    len(self.all_ids),
                    self.index.N,
                )
                with open(self._ann_index_temp_path, "rb") as f:
                    self.index = pickle.load(f)
                self._ann_index_temp_path.unlink()    

            # Ensure cache is invalidated
            self.active_ids_arr = None

    def _query_index(
        self,
        query_vec: np.ndarray,
        ids: list[bytes],
        top_k: int,
    ) -> tuple[list[bytes], np.ndarray]:
        """
        Search for nearest neighbors to the query vector via the Rii engine.

        Args:
            query_vec (np.ndarray): Query vector with shape (1, d).
            ids (list[bytes]): Corresponding IDs for each vector.
            top_k (int): Number of top results to return.

        Returns:
            tuple[list[bytes], np.ndarray]: Top-k IDs and distance scores, ordered by distance
                (ascending, closest first).
        """
        # Ensure query vector is 1D (rii expects 1D array)
        if query_vec.ndim == 2:
            query_vec = query_vec.flatten()

        # Map application IDs to internal index offsets
        target_indices = np.array([self.id_to_idx[uid] for uid in ids if uid in self.id_to_idx])

        if len(target_indices) == 0:
            return [], np.array([])

        # If OPQ is active, the query must be rotated into the same
        # optimized subspace used during the training phase.
        if self.metric == "opq":
            query_vec = self.index.fine_quantizer.rotate(query_vec)

        indices, top_scores = self.index.query(
            query_vec,
            topk=top_k,  # rii index uses topk instead of top_k
            target_ids=target_indices,
        )
        top_ids = [self.all_ids[i] for i in indices]

        return top_ids, top_scores
