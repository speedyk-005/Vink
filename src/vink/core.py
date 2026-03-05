from typing import Annotated, Callable, Literal
from pathlib import Path
import shutil
from uuid import UUID

import numpy as np
import rii
import nanopq
import pysqlite3 as sqlite3
from pydantic import Field, ValidationError

from vink.utils.input_validation import validate_arguments, validate_embedding, validate_id 
from vink.exceptions import VectorDimensionError, InvalidInputError
from vink.models import VectorRecords


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
    """

    @validate_arguments
    def __init__(
        self,
        dir_path: str | Path,
        dim: int,
        metric: Literal["euclidian", "dot"] = "euclidian",
        num_subspaces: Annotated[int, Field(ge=0)] = 32,
        quantizer: Literal["pq", "opq"] = "pq",
        force_exact: bool = False,
        switch_ratio: Annotated[float, Field(ge=2, le=16)] = 4.0,
        reconfig_threshold: Annotated[int, Field(ge=0)] = 1000,
        verbose: bool = False,
        embedding_callback: Callable | None = None,
        overwrite: bool = False,
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
            dir_path (str | Path): Directory path to store vector data. Contains
                the pickled index and SQLite database for vector records.
                Use ":memory:" for volatile in-memory storage.
            dim (int): Dimension of the vectors.
            metric (Literal["euclidian", "dot"], optional): Distance metric to use.
                Defaults to "euclidian".
            num_subspaces (int, optional): Number of subspaces for product quantization.
                Only applicable when force_exact is False. Defaults to 32.
            quantizer (Literal["pq", "opq"], optional): Quantization method.
                Only applicable when force_exact is False.
                OPQ (Optimized Product Quantization) is slightly more accurate but
                slower than PQ. Defaults to "pq".
            force_exact (bool, optional): If True, only exact calculation is used.
                If False, switches between exact and ANN based on switch_ratio.
                Defaults to False.
            switch_ratio (float, optional): Ratio threshold for switching between
                exact and approximate search. Only applicable when force_exact is False.
                Recommended values are powers of 2 (2, 4, 8, or 16) as the ratio
                is not linear. Defaults to 4.0.
            reconfig_threshold (int, optional): Number of inserts before reconfiguring the
                index to maintain search performance. Only applicable when force_exact
                is False. Defaults to 1000.
            verbose (bool, optional): Enable verbose output. Defaults to False.
            embedding_callback (Callable | None, optional): Callback function to generate
                embeddings from content. If provided, 'embedding' key is optional in
                vector records as it will be generated via this callback. Defaults to None.
            overwrite (bool, optional): Overwrite existing index if exists. Defaults to False.
        """
        # Determine if in-memory mode
        self._in_memory = isinstance(dir_path, str) and dir_path.strip() == ":memory:"
        
        self._dir_path = Path(dir_path)
        self._dim = dim
        self._num_subspaces = num_subspaces

        # L2 norm is the Euclidean distance metric; we use "l2" internally for compatibility with rii/nanopq
        self._metric = "l2" if metric == "euclidian" else metric

        self._quantizer = quantizer
        self._force_exact = force_exact
        self._switch_ratio = switch_ratio
        self.reconfig_threshold = reconfig_threshold
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
        
        self._conn = sqlite3.connect(str(self.record_db))
        self._ensure_table_exists()

        # Initialize with ExactSearchStrategy by default
        from vink.strategies.exact_search import ExactSearchStrategy
        self._strategy = ExactSearchStrategy(
            conn=self._conn,
            dir_path=self._dir_path if not self._in_memory else None,
            dim=self._dim,
            in_memory=self._in_memory,
            metric=self._metric,
            verbose=self.verbose,
        )

    def _validate_config(self) -> None:
        """
        Internal handshake to verify embedding dimensions and PQ constraints.
        """
        # Callback Handshake using existing validation logic
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

        # PQ Constraint Logic (Geometric safety)
        if not self._force_exact:
            if self._num_subspaces > self._dim:
                raise VectorDimensionError(
                    f"num_subspaces ({self._num_subspaces}) cannot exceed dim ({self._dim})."
                )
            
            if self._dim % self._num_subspaces != 0:
                remainder = self._dim / self._num_subspaces
                raise VectorDimensionError(
                    f"Dimension ({self._dim}) must be divisible by num_subspaces ({self._num_subspaces}). "
                    f"Result: {remainder:.2f}"
                )

    def _ensure_table_exists(self) -> None:
        """Create the SQLite database table if it doesn't exist yet.""" 
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id BLOB PRIMARY KEY,        -- UUID bytes
                content TEXT NOT NULL,
                metadata BLOB NOT NULL,     -- JSON binary format
                embedding BLOB,
                deleted BOOLEAN DEFAULT 0   -- Soft-delete flag (0=active, 1=deleted)
            )
        """)
        self._conn.commit()

    @validate_arguments
    def add(self, vector_records: list[dict]) -> list[str]:
        """Add vectors to the index.

        Args:
            vector_records (list[dict]): List of dicts with 'content', 'metadata',
                and 'embedding' keys. 'id' is optional - if not provided, a UUIDv7
                will be auto-generated.

        Returns:
            list[str]: List of assigned UUIDv7 IDs.
            
        Raises:
            InvalidInputError: If validation fails.
        """
        # Validate and convert to VectorRecords
        try:
            records = VectorRecords(
                dim=self.dim,
                records=vector_records,
                embedding_callback=self.embedding_callback,
            )
        except ValidationError as e:
            print(e.errors())
            raise InvalidInputError(f"Invalid vector records: {str(e)}")

        return self._strategy.add(records)

    @validate_arguments
    def delete(self, ids: list[str]) -> None:
        """Delete vectors from the index by their IDs.

        Args:
            ids (list[str]): List of UUIDv7 IDs (as strings) to delete.
        """
        id_bytes = [validate_id(id_str) for id_str in ids]
        return self._strategy.delete(id_bytes)

    @validate_arguments
    def search(
        self,
        query: list[float] | np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
    ) -> list[dict]:
        """Search for k nearest neighbors.

        Args:
            query (list[float] | np.ndarray): The query vector as a list of floats,
                1D numpy array (d,), or 2D numpy array (1, d).
            top_k (int, optional): Number of nearest neighbors to return.
                Defaults to 10.
            include_vectors (bool, optional): If True, include 'embedding' key in results.
                Defaults to False.

        Returns:
            list[dict]: List of dicts with 'id', 'content', 'metadata', 'distance',
                and optionally 'embedding' (if include_vectors is True).
        """
        query = validate_embedding(query)
        return self._strategy.search(query, top_k=top_k, include_vectors=include_vectors)


# Read-only properties (cannot be changed after initialization)
# Defined outside class to keep it clean
VinkDB.metric = property(lambda self: "euclidian" if self._metric == "l2" else self._metric)
VinkDB.dir_path = property(lambda self: self._dir_path)
VinkDB.dim = property(lambda self: self._dim)
VinkDB.num_subspaces = property(lambda self: self._num_subspaces)
VinkDB.quantizer = property(lambda self: self._quantizer)
VinkDB.force_exact = property(lambda self: self._force_exact)
VinkDB.switch_ratio = property(lambda self: self._switch_ratio)
VinkDB.in_memory = property(lambda self: self._in_memory)
VinkDB.strategy = property(lambda self: self._strategy)  
