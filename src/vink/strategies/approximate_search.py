import random
from pathlib import Path
from readerwriterlock import rwlock
from typing import Literal

import nanopq
import numpy as np
import rii

from vink.exceptions import IndexNotFittedError, InvalidInputError
from vink.models import AnnConfig
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
        verbose: bool = False,
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

        self.index: rii.Rii | None = None
        self._is_fitted: bool = False
        self._delta_since_reconfig = 0

        self.ids: list[bytes] = []
        self.id_to_idx: dict[bytes, int] = {}

        # Boolean mask for active/deleted status
        self.mask: np.ndarray = np.array([], dtype=bool)

        # Cache placeholder
        self.active_ids: np.ndarray | None = None

    def fit(
        self,
        vectors: np.ndarray,
        active_ids: np.ndarray,
        ann_config: AnnConfig,
    ) -> None:
        """
        Initialize the Approximate Search index by training the Quantizer.

        It processes all currently indexed vectors to generate the subspace codebooks
        required for approximate search.
        The quantizer is initialized with K-means++ ('++') to ensure robust initialization
        of codebooks across the feature space, improving clustering stability
        and reconstruction accuracy.

        Warning:
            This method is non-reentrant. Once the index has been fitted, it is
            locked into this configuration for the life of the database. Invoking
            this method more than once will raise an error.

        Args:
            vectors (np.ndarray): A 2D array of shape (N, D) representing the N vectors
                of dimensionality D to be indexed.
            active_ids (np.ndarray): Array of active IDs corresponding to the vectors.
            ann_config (AnnConfig): Configuration for approximate nearest neighbor search.

        Raises:
            RuntimeError: If called on an index that has already been fitted.
        """
        if self._is_fitted:
            raise RuntimeError(
                f"Index already initialized for {self.dir_path or 'in-memory storage'}. "
                "The quantization parameters cannot be modified after fitting."
            )

        log_info(self.verbose, "Starting ANN index fit with {} vectors.", len(vectors))

        if ann_config.codebook_size >= len(vectors):
            raise InvalidInputError(
                f"Codebook size ({ann_config.codebook_size}) must be strictly less than "
                f"the number of training vectors ({len(vectors)}). "
                "This constraint is required by Product Quantization."
            )

        self.active_ids = active_ids
        self.ids = list(active_ids)

        # Build id_to_idx mapping
        self.id_to_idx = {id_bytes: idx for idx, id_bytes in enumerate(self.ids)}

        # Initialize mask - all True since these are active
        self.mask = np.ones(len(self.ids), dtype=bool)

        self.reconfig_threshold = ann_config.reconfig_threshold

        # Sample subset for training (max 1000 vectors to limit training time)
        # Keep all vectors for the index, only sample for codec training
        max_train_size = 1000
        n_vectors = len(vectors)
        if n_vectors > max_train_size:
            train_indices = random.sample(range(n_vectors), k=max_train_size)
            train_vectors = vectors[train_indices]
        else:
            train_vectors = vectors

        log_info(
            self.verbose,
            "Training codec with {} vectors using {} subspaces and codebook_size {}.",
            len(train_vectors),
            ann_config.num_subspaces,
            ann_config.codebook_size,
        )

        codec_params = {
            "M": ann_config.num_subspaces,
            "Ks": ann_config.codebook_size,
            "metric": self.metric,
            "verbose": False,
        }

        if ann_config.quantizer == "opq":
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

        self._is_fitted = True
        log_info(self.verbose, "ANN index fit completed successfully.")

    def _validate_fitted(self) -> None:
        """
        Validates the index state.

        Raises:
            IndexNotFittedError: If the quantization parameters (codebooks,
                rotation matrix) have not been learned via fit().
        """
        if not self._is_fitted:
            raise IndexNotFittedError(
                "The index parameters are not yet learned. Please run .fit() "
                "on your training vectors before performing any index operations."
            )

    def _do_reconfigure(self) -> dict:
        """Reconfigure index in background thread without blocking queries."""
        log_info(self.verbose, "ANN index reconfiguration in progress (this may take a moment)...")
        self.index.reconfigure()
        self._delta_since_reconfig = 0
        log_info(self.verbose, "ANN index reconfigured.")

    def _ensure_cache(self) -> None:
        """Build cache of IDs if not already cached."""
        if self.active_ids is not None:
            return

        # Caller holds the lock; no nested lock acquisition.
        active_indices = self.mask.nonzero()[0]

        if len(active_indices) == 0:
            self.active_ids = np.empty((0,), dtype="S16")
            return

        self.active_ids = np.array(self.ids, dtype="S16")[active_indices]

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
                idx = len(self.ids)
                self.ids.append(record.id)
                self.id_to_idx[record.id] = idx
                self.mask = np.append(self.mask, True)
                self.active_ids = None
                embeddings.append(record.embedding)
                assigned_ids.append(self._bytes_to_uuid_str(record.id))

        self.index.add(np.vstack(embeddings))
        if not is_buffer:
            self.db.insert(vector_records)

        self._delta_since_reconfig += len(vector_records.records)
        if self._delta_since_reconfig >= self.reconfig_threshold:
            self._do_reconfigure()

        return assigned_ids

    def delete(self, ids: list[bytes]) -> None:
        """
        Delete vectors from the index by their IDs.

        Args:
            ids (list[bytes]): List of UUIDv7 IDs to delete.

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()

        with self._rwlock.gen_wlock():
            for id_bytes in ids:
                idx = self.id_to_idx.get(id_bytes)
                if idx is not None:
                    self.mask[idx] = False

            self.db.delete(ids, soft=True)

            # Invalidate cache
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

        Raises:
            IndexNotFittedError: If called on an index that has not been fitted yet.
        """
        self._validate_fitted()
        with self._rwlock.gen_rlock():
            self._ensure_cache()
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
                filtered_ids = self.active_ids[temp_mask]
            else:
                # Use cached versions
                filtered_ids = self.active_ids
            """

            ids, scores = self._query_index(query_vec, filtered_ids, top_k)

        if not ids:
            return []

        # Query SQLite for full records of top_k IDs
        placeholders = ",".join("?" * len(ids))
        where = f"id IN ({placeholders})"
        rows = self.db.fetch(where=where, params=ids, include_vectors=include_vectors)
        id_to_row = {row[0]: row for row in rows}

        return self._build_results(ids, scores, id_to_row, include_vectors)

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
        target_indices = sorted(
            [self.id_to_idx[uid] for uid in ids if uid in self.id_to_idx]
        )

        if not target_indices:
            return [], np.array([])

        # If OPQ is active, the query must be rotated into the same
        # optimized subspace used during the training phase.
        if self.metric == "opq":
            query_vec = self.index.fine_quantizer.rotate(query_vec)

        indices, top_scores = self.index.query(
            query_vec,
            topk=top_k,  # rii index uses topk instead of top_k
            target_ids=np.array(target_indices),
        )
        top_ids = [self.ids[i] for i in indices]

        return top_ids, top_scores
