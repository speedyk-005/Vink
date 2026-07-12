import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, ValidationError
from readerwriterlock import rwlock

from vinkra import __version__
from vinkra.exceptions import InvalidInputError, VectorDimensionError
from vinkra.latency_predictor import LatencyPredictor
from vinkra.models import AnnConfig, VectorRecords
from vinkra.sql_wrapper import SQLiteWrapper
from vinkra.strategies.base import BaseStrategy
from vinkra.strategies.exact_search import ExactSearch

# ApproximateSearch and the latency predictor are lazy imported
from vinkra.utils.input_validation import (
    pretty_errors,
    validate_arguments,
    validate_embedding,
    validate_id,
)
from vinkra.utils.logging import log_info, logger


class VinkraDB:
    """
    Vector database with hybrid exact/approximate nearest neighbor search.

    VinkraDB automatically switches from exact brute-force search to approximate
    nearest neighbor (ANN) search based on dataset size, using Reconfigurable Inverted
    Index (RII) and Product Quantization (PQ) for efficient ANN.

    Note:
        ANN switching is one-way. Once switched, the system never switches back to exact search.

    Features:

        - Hybrid search: exact for small datasets, ANN for large datasets.
        - Automatic strategy reconfig based on runtime-calibrated latency prediction.
        - Normalized embeddings for consistent distance metrics.
        - Supports Euclidean (L2) and cosine (dot) product similarity.
        - Soft deletes: efficient deletion without data reorganization.

    Getting AnnConfig:

        To customize ANN behavior, create an AnnConfig instance:

        >>> from vinkra import AnnConfig
        >>> config = AnnConfig(
        ...     num_subspaces=16,
        ...     codebook_size=128,
        ...     switch_latency_ms=300,
        ...     quantizer="pq"
        ... )

        >>> db = VinkraDB(dir_path="./data", dim=384, ann_config=config)

        For help with AnnConfig parameters, call AnnConfig.help()
    """

    @validate_arguments
    def __init__(
        self,
        dim: Annotated[int, Field(ge=16)],
        dir_path: str | Path | None = None,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        force_exact: bool = False,
        ann_config: AnnConfig | None = None,
        embedding_callback: Callable | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a VinkraDB instance.

        Note:
            The only editable attributes after initialization are:
                - ann_config
                - embedding_callback
                - verbose
            Everything else is read-only properties.

        Args:
            dim: Dimension of the vectors. Must be higher than 16.
            dir_path: Directory path to store vector data. Contains the pickled index
                and SQLite database for vector records.
                Pass None for volatile in-memory storage.
            metric: Distance metric to use.
                Defaults to "euclidean".
            force_exact: If True, only exact calculation is used.
                If False, switches between exact and ANN based on runtime calibration.
                Defaults to False.
            ann_config: Configuration for approximate nearest neighbor search.
                Used during switching and compacting. Defaults to ANNConfig with standard settings.
                Only applicable when force_exact is False.
            embedding_callback: Callback function to generate embeddings
                from content. If provided, 'embedding' key is optional in
                vector records as it will be generated via this callback. Defaults to None.
            overwrite: Overwrite existing index if exists. Defaults to False.
            verbose: Enable verbose output. Defaults to False.
        """
        self._dir_path = Path(dir_path) if dir_path else None
        self._dim = dim
        self._metric = metric
        self._ann_config = ann_config
        self._force_exact = force_exact
        self.embedding_callback = embedding_callback
        self.verbose = verbose

        self._strategy: BaseStrategy | None = None
        self._latency_predictor: LatencyPredictor | None = None

        # Default the ann_config with standard settings if user doesn't provide their own
        if not (self._force_exact or self._ann_config):
            self._ann_config = AnnConfig()
        self._validate_config()

        if self._dir_path is not None:
            if overwrite and self.dir_path.exists():
                shutil.rmtree(self.dir_path)

            self.dir_path.mkdir(parents=True, exist_ok=True)
            self._records_db_path = str(self.dir_path / "records.sqlite")
        else:
            self._records_db_path = ":memory:"

        self._records_db = SQLiteWrapper(
            self._records_db_path,
            index_config={
                "dim": str(self.dim),
                "metric": self.metric,
                "strategy": "exact",
            },
        )

        # Threading components for ANN auto-switch
        self._ann_building = False
        self._rwlock = rwlock.RWLockFair()

        self.load()

    @property
    def dir_path(self) -> Path | None:
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
    def strategy(self) -> str:
        """The internal indexing strategy currently active."""
        if self._strategy is None:
            return "exact_search"
        return (
            "exact_search"
            if isinstance(self._strategy, ExactSearch)
            else "approximate_search"
        )

    def count(self, status: Literal["active", "deleted"] | None = None) -> int:
        """Count vectors in the database.

        Args:
            status: Which vectors to count. Count all if not provided.

        Returns:
            Count of vectors.
        """
        return self._records_db.count(status)

    def stats(self) -> dict:  # pragma: no cover
        """Return database statistics and metadata.

        Returns:
            Database metadata including version, dimension, metric, strategy,last_saved_at, last_deleted_at, active_count, deleted_count,
                and other stored metadata.
        """
        return {
            "version": __version__,
            "dim": self._records_db["dim"],
            "metric": self._records_db["metric"],
            "strategy": self._records_db["strategy"],
            "last_saved_at": self._records_db["last_saved_at"],
            "last_deleted_at": self._records_db["last_deleted_at"],
            "active_count": self.count("active"),
            "deleted_count": self.count("deleted"),
        }

    def _validate_config(self) -> None:
        """
        Internal handshake to verify embedding dimensions and PQ constraints.
        """
        if not self._force_exact:
            self._ann_config.validate_vector_dim(self._dim)

        # Callback Handshake validation
        if self.embedding_callback:
            try:
                raw_vec = self.embedding_callback("vinkra_warmup_test")
                validated_vec = validate_embedding(
                    raw_vec, dim=self.dim, metric=self.metric
                )
                if validated_vec.shape[-1] != self._dim:
                    raise VectorDimensionError(
                        f"Embedding callback output dimension ({validated_vec.shape[-1]}) "
                        f"does not match VinkraDB dimension ({self._dim})."
                    )
            except (VectorDimensionError, InvalidInputError):
                # Let these specific errors bubble up for the test/user
                raise
            except Exception as e:
                raise InvalidInputError(
                    "Embedding callback crashed during handshake"
                ) from e

    @validate_arguments
    def add(self, vector_records: list[dict]) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records: List of dicts with 'content', 'metadata',
                and 'embedding' keys. 'id' is optional
                If not provided, a UUIDv7 will be auto-generated.

        Note:
            The first batch (when database is empty) is limited to 10,000 vectors to avoid
            expensive initial index operations. This constraint only applies to the first add()
            call. Subsequent batches can be any size.

        Returns:
            List of assigned UUIDv7 IDs.

        Raises:
            InvalidInputError: If validation fails or if the first batch exceeds 10,000 vectors.
        """
        if not vector_records:
            log_info(self.verbose, "Input is empty, returning empty list.")
            return []

        if split := self._maybe_split_first_batch(vector_records):
            return split

        try:
            validated = VectorRecords(
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

        validated_records = [r.model_dump() for r in validated.records]

        if self._ann_building:
            assigned_ids = [r["id"] for r in validated_records]
            self._records_db.insert(validated_records, is_buffer=True)
            log_info(
                self.verbose,
                "Successfully added {} records to buffer.",
                len(assigned_ids),
            )
            return assigned_ids

        assigned_ids = self._strategy.add(validated_records)

        if self.strategy == "exact_search":
            # Check if switch should be triggered based on new count
            if self._should_switch():
                self._ann_building = True
                Thread(target=self._prepare_approx_strategy, daemon=True).start()

        log_info(
            self.verbose, "Successfully added {} records to index.", len(assigned_ids)
        )
        return assigned_ids

    @validate_arguments
    def soft_delete(self, ids: list[str]) -> None:
        """Soft-delete vectors from the index by their IDs (marks as deleted).

        Args:
            ids: List of UUIDv7 IDs to soft-delete.
        """
        log_info(self.verbose, "Soft-deleting {} vectors from index.", len(ids))

        id_bytes = [validate_id(id_str) for id_str in ids]

        # If ANN is building, write to buffer for replay after switch
        if self.strategy != "approximate_search" and self._ann_building:
            self._records_db.soft_delete(id_bytes)
            self._records_db["last_deleted_at"] = datetime.now(UTC).isoformat()
            log_info(
                self.verbose, "Marked {} vectors for soft-deletion in buffer.", len(ids)
            )
            return

        self._strategy.soft_delete(id_bytes)
        self._records_db["last_deleted_at"] = datetime.now(UTC).isoformat()

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

    def close(self) -> None:
        """Save and close the database."""
        self.save()
        self._records_db.close()

    def save(self) -> None:
        """Save the index to disk."""
        log_info(self.verbose, "Saving index to {}.", self._dir_path)
        self._strategy.save()
        self._records_db["last_saved_at"] = datetime.now(UTC).isoformat()
        log_info(self.verbose, "Index saved successfully.")

    def load(self, overwrite: bool = False) -> None:
        """Load the index from disk.

        Args:
            overwrite: If True, replace in-memory state with loaded data.
                Defaults to False.
        """
        log_info(self.verbose, "Loading index from {}.", self._dir_path)

        if self._strategy is None:
            params = {
                "db": self._records_db,
                "dir_path": self._dir_path,
                "dim": self._dim,
                "metric": self.metric,
                "verbose": self.verbose,
            }
            if self.strategy == "exact_search":
                strategy_class = ExactSearch
            else:
                from vinkra.strategies.approximate_search import ApproximateSearch

                strategy_class = ApproximateSearch
                params["ann_config"] = self._ann_config

            self._strategy = strategy_class(**params)

        self._strategy.load(overwrite=overwrite)

        # Lazy init predictor only if strategy is exact
        if self.strategy == "exact_search":
            from vinkra.latency_predictor import LatencyPredictor

            self._latency_predictor = LatencyPredictor(dim=self._dim)

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
            query_vec: The query vector as a list of floats,
                1D numpy array (d,), or 2D numpy array (1, d).
            top_k: Number of nearest neighbors to return. Defaults to 10.
            include_vectors: If True, include 'embedding' key in results.
                Defaults to False.
            filters: Filter expressions to apply before scoring.
                E.g., ["category == 'science'", "price >= 10"].

        Returns:
            List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        log_info(
            self.verbose,
            "Searching for {} nearest neighbors using {}.",
            top_k,
            self.strategy,
        )

        start = time.perf_counter()

        validated_query = validate_embedding(
            query_vec,
            dim=self.dim,
            metric=self.metric,
        )
        results = self._strategy.search(
            validated_query,
            top_k=top_k,
            include_vectors=include_vectors,
            filters=filters,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Tune predictor with actual latency
        if self._latency_predictor is not None:
            self._latency_predictor.tune(self.count(), elapsed_ms)

        log_info(
            self.verbose,
            "Found {} results for query in {} ms.",
            len(results),
            round(elapsed_ms, 2),
        )
        return results

    def _maybe_split_first_batch(self, vector_records: list[dict]) -> list[str] | None:
        """Split the first batch if it would exceed the latency threshold.

        Returns the concatenated result of adding each part separately
            or None if the batch is small enough or the DB is not empty.
        """
        if (
            self.count() > 0
            or self._latency_predictor.predict(len(vector_records))
            <= self._ann_config.switch_latency_ms
        ):
            return None

        # Find max subset that stays under latency threshold via halving.
        n_cand = n_total // 2
        while n_cand > 0:
            if (
                self._latency_predictor.predict(n_cand)
                <= self._ann_config.switch_latency_ms
            ):
                break
            n_cand //= 2
        optimal = max(n_cand, 1)

        log_info(
            self.verbose,
            "First batch exceeds threshold: splitting {} into {} + {}",
            len(vector_records),
            optimal,
            len(vector_records) - optimal,
        )
        return self.add(vector_records[:optimal]) + self.add(vector_records[optimal:])

    def _should_switch(self) -> bool:
        """
        Check if ANN switch should be triggered based on dual conditions:

        1. Sufficiency: num_vectors >= min_required (num_subspaces * codebook_size)
        2. latency: predicted latency exceeds this threshold
        """
        if self._force_exact:
            return False

        n_vecs = self.count()
        cfg = self._ann_config
        min_required = cfg.num_subspaces * cfg.codebook_size

        if n_vecs < min_required:
            return False

        predicted_latency = self._latency_predictor.predict(n_vecs)
        return predicted_latency >= cfg.switch_latency_ms

    def _prepare_approx_strategy(self) -> None:
        """
        Build ANN strategy in a background daemon thread.

        Runs in a daemon thread so add()/search() remain unblocked.
        Replays buffered records after the strategy switch completes.
        """
        with self._strategy._rwlock.gen_rlock():
            self._strategy._ensure_cache()
            vectors = self._strategy.active_vectors_arr
            ids = self._strategy.active_ids_arr

        from vinkra.strategies.approximate_search import ApproximateSearch

        approx_strategy = ApproximateSearch(
            db=self._records_db,
            dir_path=self._dir_path,
            dim=self._dim,
            metric=self.metric,
            verbose=self.verbose,
            ann_config=self._ann_config,
        )
        approx_strategy.fit(vectors, ids)

        log_info(self.verbose, "ANN index fit complete, switching strategy.")
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

        buffered = [
            {"id": row[0], "embedding": np.frombuffer(row[1], dtype=np.float32)}
            for row in buffer_rows
        ]

        strategy.add(buffered, is_buffer=True)
        self._records_db.clear_buffer()
        log_info(
            self.verbose,
            "Buffer dump: added {} vectors to ANN index.",
            len(records),
        )

    def __enter__(self) -> "VinkraDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            logger.error(f"Transaction failed: {exc_val}")
            return False  # Tell python to reraise it
        self.close()
