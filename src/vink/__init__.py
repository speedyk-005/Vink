"""
Vink: Vector Incremental Nano Kit

A lightweight vector database that incrementally switches from exact to
approximate search as your data grows — without full index rebuilds.

Key differentiators:
    - **Incremental inserts**: Add vectors anytime — no rebuild per insert.
    - **Automatic strategy switching**: No manual tuning — exact for small datasets, ANN for large.
    - **Thread-safe**: Background ANN building doesn't block new operations.
    - **Soft deletes + compact**: Efficient deletion with explicit storage reclamation.

Features:
    - ~100x faster using RII and Product Quantization for large datasets.
    - Pure Python: No external services or dependencies required.
    - Supports Euclidean (L2) and dot product similarity.
    - SQLite-backed persistent storage.

Technical Background:
    Vink uses Reconfigurable Inverted Index (RII) with Product Quantization (PQ)
    for approximate nearest neighbor search. The switch from exact to approximate
    happens when the normalized power-law complexity reaches 1.0:
    (dim * vectors / 1M) ^ switch_exp >= 1.0. Default switch_exp is 1.0.

References:
    .. [Matsui18] Matsui et al., "Reconfigurable Inverted Index", ACM MM 2018.

    .. [Jegou11] Jegou et al., "Product Quantization for Nearest Neighbor Search",
       IEEE TPAMI 2011.

    .. [Matsui15] Matsui et al., "Optimized Product Quantization for Nearest
       Neighbor Search", CVPR 2015.

See Also:
    - RII: https://github.com/matsui528/rii
    - nanopq: https://github.com/matsui528/nanopq
"""

from importlib.metadata import PackageNotFoundError, version

from vink.core import VinkDB
from vink.exceptions import *
from vink.models import AnnConfig

try:
    __version__ = version("vink")
except PackageNotFoundError:
    __version__ = "0.0.0"
