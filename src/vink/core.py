from typing import Callable, Literal
from pathlib import Path
import shutil
import json
import math
from threading import Lock
import numpy as np

import regex as re
import pysqlite3 as sqlite3
from pydantic import ValidationError

from vink.utils.input_validation import (
    validate_arguments,
    validate_embedding,
    validate_id,
    pretty_errors,
)

from vink.tasker import Tasker
from vink.models import VectorRecord, VectorRecords, ANNConfig
from vink.utils.logging import log_info
from vink.exceptions import VectorDimensionError, InvalidInputError



class VinkDB:
    """
    Pure Python vector database with hybrid exact/approximate nearest neighbor search.

    VinkDB automatically switches between exact brute-force search and approximate 
    nearest neighbor (ANN) search based on dataset size, using Reconfigurable Inverted 
    Index (RII) and Product Quantization (PQ) for efficient ANN.

    Features:
        - Hybrid search: exact for small datasets, ANN for large datasets.
        - Automatic strategy switching based on configurable ratio.
        - L2-normalized embeddings for consistent distance metrics.
        - Supports Euclidean (L2) and dot product similarity.
        - Soft deletes: efficient deletion without data reorganization.
    
    Getting ANNConfig:
        To customize ANN behavior, create an ANNConfig instance:
        
        >>> from vink import ANNConfig
        >>> config = ANNConfig(
        ...     num_subspaces=16,
        ...     codebook_size=128,
        ...     switch_ratio=4.0,
        ...     quantizer="pq"
        ... )
        >>> db = VinkDB(dir_path="./data", dim=384, ann_config=config)
        
        For help with ANNConfig parameters, call ANNConfig.help()
    """

    @validate_arguments
    def __init__(
        self,
        dir_path: str | Path,
        dim: int,
        metric: Literal["euclidian", "dot"] = "euclidian",
        force_exact: bool = False,
        ann_config: ANNConfig | None = None,
        embedding_callback: Callable | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a VinkDB instance.

        Note:
            The only editable attributes after initialization are:
                - reconfig_threshold
                - verbose
                - embedding_callback
            Everything else is read-only properties.

        Args:
            dir_path (str | Path): Directory path to store vector data. Contains the pickled index
                and SQLite database for vector records.
                Use ":memory:" for volatile in-memory storage.
            dim (int): Dimension of the vectors.
            metric (Literal["euclidian", "dot"], optional): Distance metric to use.
                Defaults to "euclidian".
            force_exact (bool, optional): If True, only exact calculation is used.
                If False, switches between exact and ANN based on switch_ratio.
                Defaults to False.
            ann_config (ANNConfig | None, optional): Configuration for approximate nearest neighbor search.
                Only applicable when force_exact is False.
                If not provided, defaults to ANNConfig with standard settings.
            embedding_callback (Callable | None, optional): Callback function to generate embeddings
                from content. If provided, 'embedding' key is optional in
                vector records as it will be generated via this callback. Defaults to None.
            overwrite (bool, optional): Overwrite existing index if exists. Defaults to False.
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        # Determine if in-memory mode
        self._in_memory = isinstance(dir_path, str) and dir_path.strip() == ":memory:"

        self._dir_path = Path(dir_path)
        self._dim = dim

        # L2 norm is the Euclidean distance metric; we use "l2" internally for compatibility with rii/nanopq
        self._metric = "l2" if metric == "euclidian" else metric

        self._force_exact = force_exact
        self._ann_config = ann_config

        # Default the config with standard settings if force_exact is not true
        if not (self._force_exact or self._ann_config):
            self._ann_config = ANNConfig()
            
        self.embedding_callback = embedding_callback
        self.verbose = verbose

        self._validate_config()

        if not self._in_memory:
            if overwrite and self._dir_path.exists():
                shutil.rmtree(self._dir_path)
            
            self._dir_path.mkdir(parents=True, exist_ok=True)
            self.record_db = self._dir_path / "records.sqlite"
        else:
            self.record_db = ":memory:"
        
        self._conn = sqlite3.connect(str(self.record_db), check_same_thread=False)
        self._ensure_table_exists()

        # Initialize with ExactSearchStrategy by default
        from vink.strategies.exact_search import ExactSearch
        self._strategy = ExactSearch(
            conn=self._conn,
            dir_path=self._dir_path if not self._in_memory else None,
            dim=self._dim,
            in_memory=self._in_memory,
            metric=self._metric,
            verbose=self.verbose,
        )

        # Threading components for ANN auto-switch
        self._ann_building = False
        self._lock = Lock()
        self._tasker = Tasker(task=lambda: self._build_approx(), once=True)

    @property
    def dir_path(self) -> Path:
        """The absolute path to the directory where database files are persisted."""
        return self._dir_path

    @property
    def dim(self) -> int:
        """The dimension of the vectors (embedding length)."""
        return self._dim

    @property
    def metric(self) -> str:
        """The distance metric used for similarity search."""
        return "euclidean" if self._metric == "l2" else self._metric

    @property
    def force_exact(self) -> bool:
        """Whether the database uses exact brute-force calculations only."""
        return self._force_exact

    @property
    def in_memory(self) -> bool:
        """Whether storage is volatile (':memory:') or persisted on disk."""
        return self._in_memory

    @property
    def strategy(self) -> str:
        """The internal indexing strategy currently active, formatted in snake_case."""
        strategy_name = self._strategy.__class__.__name__
        parts = re.split(r"(?<!^)(?=\p{LU})", strategy_name)
        return "_".join([p.lower() for p in parts])

    @property
    def ann_config(self) -> dict:
        """The ANN configuration settings as a dictionary."""
        return self._ann_config.model_dump() if self._ann_config else {}

    def _validate_config(self) -> None:
        """
        Internal handshake to verify embedding dimensions and PQ constraints.
        """
        # Callback Handshake validation
        if self.embedding_callback:
            try:
                raw_vec = self.embedding_callback("vink_warmup_test")
                
                # This handles casting, shape normalization (1, d), and L2 projection
                validated_vec = validate_embedding(raw_vec)
                
                if validated_vec.shape[-1] != self._dim:
                    raise VectorDimensionError(
                        f"Embedding callback output dimension ({validated_vec.shape[-1]}) "
                        f"does not match VinkDB dimension ({self._dim})."
                    )
            except VectorDimensionError:
                # Let this specific error bubble up for the test/user
                raise
            except Exception as e:
                raise InvalidInputError(f"Embedding callback crashed during handshake: {e}")

        # Ann config validation
        if not self._force_exact:
            self._ann_config.validate_dim(self._dim)

    def _ensure_table_exists(self) -> None:
        """Create the SQLite database tables if they don't exist yet.""" 
        cursor = self._conn.cursor()
        
        # Main records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id BLOB PRIMARY KEY,        -- UUID bytes
                content TEXT NOT NULL,
                metadata BLOB NOT NULL,     -- JSON binary format
                embedding BLOB,
                deleted BOOLEAN DEFAULT 0   -- Soft-delete flag (0=active, 1=deleted)
            )
        """)
        
        # Buffer table for pending operations during ANN fit
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buffer (
                id BLOB PRIMARY KEY,
                content TEXT NOT NULL,
                metadata BLOB NOT NULL,
                embedding BLOB,
                deleted BOOLEAN DEFAULT 0
            )
        """)
        
        self._conn.commit()

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
        # Prevent large first batch to avoid expensive initial ANN builds
        if self.count() == 0 and len(vector_records) > 10000:
            raise InvalidInputError(
                f"First batch cannot exceed 10,000 vectors (got {len(vector_records)}). "
                f"This limit applies only to the initial add() call when the database is empty."
            )

        # Validate and convert to VectorRecords
        try:
            records = VectorRecords(
                dim=self.dim,
                records=vector_records,
                embedding_callback=self.embedding_callback,
            )
        except ValidationError as e:
            raise InvalidInputError(f"Invalid vector records: {pretty_errors(e)}") from None

        log_info(self.verbose, "Adding {} vector records to index.", len(vector_records))

        if self.strategy == "exact_search" and self._ann_building:
            assigned_ids = [r.id for r in records.records]
            self._add_to_buffer(records)
            log_info(self.verbose, "Successfully added {} records to buffer.", len(assigned_ids))
            return assigned_ids

        assigned_ids = self._strategy.add(records)

        # After adding, check if switch should be triggered based on new count
        if self.strategy == "exact_search" and self._should_switch():
            if not self._ann_building:
                self._ann_building = True
                self._tasker.run()
            
        log_info(self.verbose, "Successfully added {} records to index.", len(assigned_ids))
        return assigned_ids

    def _add_to_buffer(self, records: VectorRecords) -> None:
        """Add records to buffer table."""
        cursor = self._conn.cursor()
        for record in records.records:
            cursor.execute(
                "INSERT INTO buffer (id, content, metadata, embedding, deleted) "
                "VALUES (?, ?, jsonb(?), ?, 0)",
                (record.id, record.content, json.dumps(record.metadata), record.embedding.tobytes())
            )
        self._conn.commit()

    @validate_arguments
    def delete(self, ids: list[str]) -> None:
        """Delete vectors from the index by their IDs.

        Args:
            ids (list[str]): List of UUIDv7 IDs to delete.
        """
        log_info(self.verbose, "Deleting {} vectors from index.", len(ids))
        
        id_bytes = [validate_id(id_str) for id_str in ids]

        # If ANN is building, write to buffer for replay after switch
        if self.strategy != "approximate_search" and self._ann_building:
            placeholders = ','.join('?' * len(id_bytes))
            cursor = self._conn.cursor()
            cursor.execute(
                f"UPDATE buffer SET deleted = 1 WHERE id IN ({placeholders})",
                id_bytes,
            )
            self._conn.commit()
            log_info(self.verbose, "Marked {} vectors for deletion in buffer.", len(ids))
            return

        self._strategy.delete(id_bytes)
        log_info(self.verbose, "Successfully deleted {} vectors.", len(ids))

    def count(self) -> int:
        """Count the number of active (non-deleted) vectors in the database.
        
        Returns:
            int: The number of active vectors where deleted is 0, including buffered vectors.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM records WHERE deleted = 0")
        strategy_count = cursor.fetchone()[0]

        # If ANN is building, also count buffered vectors
        if self._ann_building:
            cursor.execute("SELECT COUNT(*) FROM buffer WHERE deleted = 0")
            buffer_count = cursor.fetchone()[0]
            return strategy_count + buffer_count

        return strategy_count

    @validate_arguments
    def search(
        self,
        query_vec: list[float] | np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Search for k nearest neighbors using the configured metric.

        Args:
            query_vec (list[float] | np.ndarray): The query vector as a list of floats,
                1D numpy array (d,), or 2D numpy array (1, d).
            top_k (int, optional): Number of nearest neighbors to return. Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        log_info(self.verbose, "Searching for {} nearest neighbors using {}.", top_k, self.strategy)
        
        query = validate_embedding(query_vec)
        results = self._strategy.search(query_vec, top_k=top_k, include_vectors=include_vectors)
        
        log_info(self.verbose, "Found {} results for query.", len(results))
        return results

    def _should_switch(self) -> bool:
        """
        Check if ANN switch should be triggered based on dual conditions:
        1. Sufficiency: num_vectors >= min_required (num_subspaces * codebook_size)
        2. Complexity: sqrt(dim * num_vectors) / 1000 >= switch_ratio

        Returns:
            bool: True if conditions met to switch to ANN, False otherwise.
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

        complexity = math.sqrt(self._dim * n_vecs) / 1000
        return complexity >= cfg.switch_ratio

    def _build_approx(self) -> dict:
        """
        Tasker task: Build ANN strategy in daemon thread.
        
        This is the callable submitted to Tasker. It runs in a background
        daemon thread and returns the new strategy without side effects.
        
        Returns:
            dict: An empty dictionnary to satisfy the tasker return type
        """
        self._strategy._ensure_cache()
        vectors = self._strategy.active_vectors
        ids = self._strategy.active_ids

        from vink.strategies.approximate_search import ApproximateSearch
        approx_strategy = ApproximateSearch(
            conn=self._conn,
            dir_path=self._dir_path if not self._in_memory else None,
            dim=self._dim,
            in_memory=self._in_memory,
            metric=self._metric,
            verbose=self.verbose,
        )
        approx_strategy.fit(vectors, ids, self._ann_config)
        
        # Automatically switch after successful build
        self._switch_to_approx_with_strategy(approx_strategy)

        return {}

    def _dump_buffer(self) -> None:
        """Read buffer where deleted=0 and apply to self._strategy."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, content, json(metadata), embedding FROM buffer WHERE deleted = 0")
        buffer_rows = cursor.fetchall()
        
        if not buffer_rows:
            return
        
        records = [
            {
                "id": id_bytes,
                "content": content,
                "metadata": json.loads(meta),
                "embedding": np.frombuffer(embed, dtype=np.float32),
            }
            for id_bytes, content, meta, embed in buffer_rows
        ]
        self._strategy.add(VectorRecords(dim=self._dim, records=records))

    def _switch_to_approx_with_strategy(self, new_strategy) -> None:
        """Apply strategy switch with the given new_strategy."""
        # Swap strategy
        with self._lock:
            self._strategy = new_strategy

        # Change the flag upfront to prevent new operations to use buffer
        self._ann_building = False

        self._dump_buffer()
        
        # Clear buffer
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM buffer")
        self._conn.commit()
