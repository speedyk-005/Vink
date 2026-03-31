import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Callable, Literal, Annotated

import numpy as np
import regex as re
from pydantic import Field, ValidationError
from readerwriterlock import rwlock

from vink.exceptions import InvalidInputError, VectorDimensionError
from vink.models import AnnConfig, VectorRecords
from vink.sql_wrapper import SQLiteWrapper

# The strategies are lazy imported
from vink.strategies.base import BaseStrategy
from vink.utils.input_validation import (
    pretty_errors,
    validate_arguments,
    validate_embedding,
    validate_id,
)
from vink.utils.logging import log_info, logger


class VinkDB:
    """
    Pure Python vector database with hybrid exact/approximate nearest neighbor search.

    VinkDB automatically switches from exact brute-force search to approximate
    nearest neighbor (ANN) search based on dataset size, using Reconfigurable Inverted
    Index (RII) and Product Quantization (PQ) for efficient ANN.

    Note:
        ANN switching is one-way — once switched, the system never switches back to exact search.

    Features:
        - Hybrid search: exact for small datasets, ANN for large datasets.
        - Automatic strategy switching based on runtime-calibrated latency prediction.
        - Normalized embeddings for consistent distance metrics.
        - Supports Euclidean (L2) and cosine (dot) product similarity.
        - Soft deletes: efficient deletion without data reorganization.

    Getting ANNConfig:
        To customize ANN behavior, create an ANNConfig instance:

        >>> from vink import AnnConfig
        >>> config = AnnConfig(
        ...     num_subspaces=16,
        ...     codebook_size=128,
        ...     quantizer="pq"
        ... )
        >>> db = VinkDB(dir_path="./data", dim=384, ann_config=config)

        For help with AnnConfig parameters, call AnnConfig.help()
    """

    @validate_arguments
    def __init__(
        self,
        dir_path: str | Path,
        dim:  Annotated[int, Field(ge=32)],
        metric: Literal["euclidean", "cosine"] = "euclidean",
        force_exact: bool = False,
        ann_config: AnnConfig | None = None,
        switch_latency_ms: float = 300,
        embedding_callback: Callable | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a VinkDB instance.

        Note:
            The only editable attributes after initialization are:
                - ann_config
                - embedding_callback (validated once)
                - verbose
            Everything else is read-only properties.

        Args:
            dir_path (str | Path): Directory path to store vector data. Contains the pickled index
                and SQLite database for vector records.
                Use ":memory:" for volatile in-memory storage.
            dim (int): Dimension of the vectors. Must be higher than 32.
            metric (Literal["euclidean", "cosine"], optional): Distance metric to use.
                Defaults to "euclidean".
            force_exact (bool, optional): If True, only exact calculation is used.
                If False, switches between exact and ANN based on runtime calibration.
                Defaults to False.
            ann_config (AnnConfig, optional): Configuration for approximate nearest neighbor search.
                Used during switching and compacting. Defaults to ANNConfig with standard settings.
                Only applicable when force_exact is False.
            embedding_callback (Callable, optional): Callback function to generate embeddings
                from content. If provided, 'embedding' key is optional in
                vector records as it will be generated via this callback. Defaults to None.
            overwrite (bool, optional): Overwrite existing index if exists. Defaults to False.
            verbose (bool, optional): Enable verbose output. Defaults to False.
            switch_latency_ms (float, optional): Switch to ANN when exact search
                predicted latency exceeds this threshold in milliseconds.
                Defaults to 300ms.
        """
        # Determine if in-memory mode
        self._in_memory = isinstance(dir_path, str) and dir_path.strip() == ":memory:"

        self._dir_path = Path(dir_path)
        self._dim = dim
        self._metric = metric
        self._ann_config = ann_config
        self._force_exact = force_exact
        self.embedding_callback = embedding_callback
        self.verbose = verbose

        # Default the config with standard settings if force_exact is not true
        if not (self._force_exact or self._ann_config):
            self._ann_config = AnnConfig()

        self._switch_latency_ms = switch_latency_ms

        self._validate_config()

        self._ops_per_ms = self._calibrate_throughput()

        if not self._in_memory:
            if overwrite and self._dir_path.exists():
                shutil.rmtree(self._dir_path)

            self._dir_path.mkdir(parents=True, exist_ok=True)
            self._records_db_path = str(self._dir_path / "records.sqlite")
        else:
            self._records_db_path = ":memory:"

        index_config = {
            "dimension": str(self._dim),
            "metric": self._metric,
            "strategy": "exact",
        }
        self._records_db = SQLiteWrapper(self._records_db_path, index_config=index_config)

        self._strategy: BaseStrategy | None = None
        self.load()

        # Threading components for ANN auto-switch
        self._ann_building = False
        self._rwlock = rwlock.RWLockFair()

    def _validate_config(self) -> None:
        """
        Handshake to verify embedding dimensions and PQ constraints.
        """
        # Callback Handshake validation
        if self.embedding_callback:
            try:
                raw_vec = self.embedding_callback("vink_warmup_test")

                # This handles casting, shape normalization (1, d), and L2 projection
                validated_vec = validate_embedding(raw_vec, metric=self._metric)

                if validated_vec.shape[-1] != self._dim:
                    raise VectorDimensionError(
                        f"Embedding callback output dimension ({validated_vec.shape[-1]}) "
                        f"does not match VinkDB dimension ({self._dim})."
                    )
            except (VectorDimensionError, InvalidInputError):
                # Let these specific errors bubble up for the test/user
                raise
            except Exception as e:
                raise InvalidInputError("Embedding callback crashed during handshake") from e

        if not self._force_exact:
            self._ann_config.validate_vector_dim(self._dim)

    @property
    def dir_path(self) -> Path:
        return self._dir_path

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def force_exact(self) -> bool:
        return self._force_exact

    @property
    def in_memory(self) -> bool:
        return self._in_memory

    @property
    def strategy(self) -> str:
        """The internal indexing strategy currently active, formatted in snake_case."""
        if self._strategy is None:
            return "exact_search"

        strategy_name = self._strategy.__class__.__name__
        parts = re.split(r"(?<!^)(?=\p{LU})", strategy_name)
        return "_".join([p.lower() for p in parts])

    def count(self, status: Literal["active", "deleted"] | None = None) -> int:
        """Count vectors in the database.

        Args:
            status (Literal["active", "deleted"], optional): Which vectors to count.
                Count all if not provided.

        Returns:
            int: Count of vectors.
        """
        return self._records_db.count(status)

    def stats(self) -> dict:  # pragma: no cover
        """Return database statistics and metadata.

        Returns:
            dict: Database metadata including version, dimension, metric, strategy,
                last_saved_at, last_deleted_at, active_count, deleted_count,
                and other stored metadata.
        """
        return {
            "version": self._records_db["version"],
            "dimension": self._records_db["dimension"],
            "metric": self._records_db["metric"],
            "strategy": self._records_db["strategy"],
            "last_saved_at": self._records_db["last_saved_at"],
            "last_deleted_at": self._records_db["last_deleted_at"],
            "active_count": self.count("active"),
            "deleted_count": self.count("deleted"),
        }

    @validate_arguments
    def add(self, vector_records: list[dict]) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (list[dict]): List of dicts with 'content', 'metadata',
                and 'embedding' keys. 'id' is optional
                If not provided, a UUIDv7 will be auto-generated.

        Note:
            The first batch (when database is empty) is limited to 10,000 vectors to avoid
            expensive initial index operations. This constraint only applies to the first add()
            call. Subsequent batches can be any size.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.

        Raises:
            InvalidInputError: If validation fails or if the first batch exceeds 10,000 vectors.
        """
        try:
            records = VectorRecords(
                dim=self.dim,
                metric=self._metric,
                records=vector_records,
                embedding_callback=self.embedding_callback,
            )
        except ValidationError as e:
            raise InvalidInputError(
                f"Invalid vector records: {pretty_errors(e)}"
            ) from None

        log_info(
            self.verbose, "Adding {} vector records to index.", len(vector_records)
        )

        if self.strategy == "exact_search":
            if self._ann_building:
                assigned_ids = [r.id for r in records.records]
                self._records_db.insert(records, is_buffer=True)
                log_info(
                    self.verbose,
                    "Successfully added {} records to buffer.",
                    len(assigned_ids),
                )
                return assigned_ids

            assigned_ids = self._strategy.add(records)

            # Check if switch should be triggered based on new count
            if not self._ann_building and self._should_switch():
                self._ann_building = True
                Thread(target=self._prepare_approx_strategy, daemon=True).start()
        else:
            assigned_ids = self._strategy.add(records)

        log_info(
            self.verbose, "Successfully added {} records to index.", len(assigned_ids)
        )
        return assigned_ids

    @validate_arguments
    def soft_delete(self, ids: list[str]) -> None:
        """Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids (list[str]): List of UUIDv7 IDs to soft-delete.
        """
        log_info(self.verbose, "Soft-deleting {} vectors from index.", len(ids))

        id_bytes = [validate_id(id_str) for id_str in ids]

        # If ANN is building, write to buffer for replay after switch
        if self.strategy != "approximate_search" and self._ann_building:
            self._records_db.soft_delete(id_bytes)
            self._records_db["last_deleted_at"] = datetime.now(timezone.utc).isoformat()
            log_info(
                self.verbose, "Marked {} vectors for soft-deletion in buffer.", len(ids)
            )
            return

        self._strategy.soft_delete(id_bytes)
        self._records_db["last_deleted_at"] = datetime.now(timezone.utc).isoformat()

    def compact(self) -> None:
        """Hard-delete soft-deleted records and rebuild the index.

        Note:
            For ApproximateSearch, the ANN index is rebuilt from scratch which can take
            20-200+ seconds depending on data size. This operation should be called
            during maintenance windows or off-peak hours.
            If not enough vectors remain to retrain the codec, rebuild is skipped.
        """
        log_info(self.verbose, "Compacting database...")
        self._strategy.compact()
        log_info(self.verbose, "Compaction complete.")

    def save(self) -> None:
        """Save the index to disk."""
        log_info(self.verbose, "Saving index to {}.", self._dir_path)
        self._strategy.save()
        self._records_db["last_saved_at"] = datetime.now(timezone.utc).isoformat()
        log_info(self.verbose, "Index saved successfully.")

    def load(self, overwrite: bool = False) -> None:
        """Load the index from disk.

        Args:
            overwrite (bool): If True, replace in-memory state with loaded data.
                Defaults to False.
        """
        log_info(self.verbose, "Loading index from {}.", self._dir_path)

        if self._strategy is None:
            params = {
                "db": self._records_db,
                "dir_path": self._dir_path if not self._in_memory else None,
                "dim": self._dim,
                "in_memory": self._in_memory,
                "metric": self.metric,
                "verbose": self.verbose,
            }
            if self.strategy == "exact_search":
                from vink.strategies.exact_search import ExactSearch
                strategy_class = ExactSearch
            else:
                from vink.strategies.approximate_search import ApproximateSearch
                strategy_class = ApproximateSearch
                params["ann_config"] = self._ann_config

            self._strategy = strategy_class(**params)

        self._strategy.load(overwrite=overwrite)
        log_info(self.verbose, "Index loaded successfully.")

    @validate_arguments
    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
        filters: list[str] | None = None,
    ) -> list[dict]:
        """Search for k nearest neighbors using the configured metric.

        Args:
            query_vec (list[float] | np.ndarray): The query vector as a list of floats,
                1D numpy array (d,), or 2D numpy array (1, d).
            top_k (int, optional): Number of nearest neighbors to return. Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.
            filters (list[str] | None, optional): Filter expressions to apply before scoring.
                E.g., ["category == 'science'", "price >= 10"].

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        log_info(
            self.verbose,
            "Searching for {} nearest neighbors using {}.",
            top_k,
            self.strategy,
        )

        active_count = self.count("active")
        if top_k > active_count:
            top_k = active_count

        validated_query = validate_embedding(query_vec, metric=self._metric)
        results = self._strategy.search(
            validated_query, top_k=top_k, include_vectors=include_vectors, filters=filters
        )

        log_info(self.verbose, "Found {} results for query.", len(results))
        return results

    def _calibrate_throughput(self, test_n: int = 10000) -> float:
        """Calibrate throughput by measuring device performance.

        Runs a quick BLAS matrix-vector multiply benchmark to measure raw compute
        throughput, then applies a 2x overhead factor to account for the gap between
        optimized BLAS and actual exact search (Python loops, SQLite lookups, etc.).

        This calibrated throughput is used to predict exact search latency at any
        given vector count: predicted_time_ms = (n_vectors * dim) / ops_per_ms

        Args:
            test_n: Number of vectors to use for calibration. Defaults to 10000.

        Returns:
            float: Calibrated operations per millisecond the device can sustain.
        """
        log_info(self.verbose, "Calibrating throughput...",)

        start = time.perf_counter()

        vectors = np.random.randn(test_n, self._dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

        query = np.random.randn(self._dim).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        _ = vectors @ query

        elapsed = time.perf_counter() - start
        ops = test_n * self._dim
        ops_per_sec = (ops / elapsed) * 2
        ops_per_ms = ops_per_sec / 1000

        log_info(self.verbose, "Calibration complete: {:.0f} ops/ms", ops_per_ms)
        return ops_per_ms

    def _should_switch(self) -> bool:
        """
        Check if ANN switch should be triggered based on:

        1. Sufficiency: num_vectors >= min_required (num_subspaces * codebook_size)
        2. Latency prediction: predicted exact search time > switch_latency_ms
        """
        if self._force_exact:
            return False

        n_vecs = self.count()
        if n_vecs == 0:
            return False

        cfg = self._ann_config
        min_required = cfg.num_subspaces * cfg.codebook_size

        if n_vecs < min_required:
            return False

        ops = n_vecs * self._dim
        predicted_time = ops / self._ops_per_ms

        return predicted_time > self._switch_latency_ms

    def _prepare_approx_strategy(self) -> None:
        """
        Build ANN strategy in a background daemon thread.

        Runs in a daemon thread so add()/search() remain unblocked.
        Replays buffered records after the strategy switch completes.
        """
        self._strategy._ensure_cache()
        vectors = self._strategy.active_vectors_arr
        ids = self._strategy.active_ids_arr

        from vink.strategies.approximate_search import ApproximateSearch

        approx_strategy = ApproximateSearch(
            db=self._records_db,
            dir_path=self._dir_path if not self._in_memory else None,
            dim=self._dim,
            in_memory=self._in_memory,
            metric=self.metric,
            verbose=self.verbose,
            ann_config=self._ann_config,
        )
        approx_strategy.fit(vectors, ids)

        log_info(self.verbose, "ANN index fit complete, switching strategy.")

        # Automatically switch after successful build
        self._switch_to_approx_strategy(approx_strategy)

    def _switch_to_approx_strategy(self, strategy) -> None:
        """Switch to approximate search and auto dumps buffer."""
        with self._rwlock.gen_wlock():
            self._strategy = strategy

        self._ann_building = False

        cursor = self._records_db.conn.cursor()
        buffer_rows = cursor.execute("""
            SELECT id, embedding FROM vec_records
            WHERE buffered = TRUE AND deleted = FALSE
        """).fetchall()

        if not buffer_rows:
            return

        records = [
            {
                "id": row[0],
                "embedding": np.frombuffer(row[1], dtype=np.float32),

                # Not used, kept for validation
                "content": "",
                "metadata": {},
            }
            for row in buffer_rows
        ]

        strategy.add(VectorRecords(dim=self.dim, metric=self.metric, records=records), is_buffer=True)
        self._records_db.clear_buffer()
        log_info(
            self.verbose,
            "Buffer dump: added {} vectors to ANN index.",
            len(records),
        )

    def __enter__(self) -> "VinkDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            logger.error(f"Transaction failed: {exc_val}")
            return False # Tell python to reraise it
        self.save()
