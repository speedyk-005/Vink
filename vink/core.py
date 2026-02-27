from typing import Literal
from pathlib import Path
import shutil

import rii
import nanopq
import pysqlite3 as sqlite3

from vink.utils.validation import validate_input


class VinkDB:
    """
    A pure Python vector database using Reconfigurable Inverted Index (RII) and
    Product Quantization (PQ).

    VinkDB is a lightweight, fast vector database built on top of RII and nanopq.
    It provides approximate nearest neighbor (ANN) search for high-dimensional
    vectors with minimal memory footprint using the IVFADC/IVFPQ algorithm.

    Features:
        - Fast and memory-efficient ANN search based on IVFADC/IVFPQ.
        - Subset search via linear PQ scan for filtered queries.
        - Reconfigurable index to maintain search performance after insertions.
        - O(1) item retrieval by identifier.
        - Supports both Euclidean (L2) and dot product metrics.
        - Serverless: no external dependencies or services required.
    """

    @validate_input
    def __init__(
        self,
        dir_path: str | Path,
        dim: int,
        metric: Literal["euclidian", "dot"] = "euclidian",
        num_subspaces: int = 32,
        quantizer: Literal["pq", "opq"] = "pq",
        reconfig_threshold: int = 500,
        verbose: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize a VinkDB instance.

        Args:
            dir_path (str | Path): Directory path to store vector data. Contains
                the pickled index and SQLite database for vector records.
            dim (int): Dimension of the vectors.
            metric (Literal["euclidian", "dot"], optional): Distance metric to use.
                Defaults to "euclidian".
            num_subspaces (int, optional): Number of subspaces for product quantization.
                Defaults to 32.
            quantizer (Literal["pq", "opq"], optional): Quantization method.
                OPQ (Optimized Product Quantization) is slightly more accurate but
                slower than PQ. Defaults to "pq".
            reconfig_threshold (int, optional): Number of inserts before reconfiguring the
                index to maintain search performance. Defaults to 500.
            verbose (bool, optional): Enable verbose output. Defaults to False.
            overwrite (bool, optional): Overwrite existing index if exists. Defaults to False.
        """
        self.dir_path = Path(dir_path)
        self.dim = dim
        self.num_subspaces = num_subspaces
        self._metric = "l2" if metric == "euclidian" else metric
        self.reconfig_threshold = reconfig_threshold
        self.verbose = verbose
        
        if overwrite and self.dir_path.exists():
            shutil.rmtree(self.dir_path)
            
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.engine = self.dir_path / "engine.vink"
        self.record_db = self.dir_path / "records.sqlite"

        self._index : rii.Rii | None = None
        self._since_reconfig = 0

    @property
    def metric(self) -> str:
        """
        Get the distance metric used for vector similarity search.

        Returns:
            str: The metric type ("euclidian" or "dot").
        """
        return "euclidian" if self._metric == "l2" else self._metric
        
