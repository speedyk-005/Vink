"""
Vink: A vector database that automatically switches from slow-but-accurate 
search to fast-but-approximate search as your data grows.

Vink intelligently balances precision and performance by automatically switching
between exact brute-force search (for small datasets) and approximate nearest 
neighbor (ANN) search (for large datasets) without any manual configuration.

Features:
    - **Automatic strategy switching**: No manual tuning needed - Vink knows when to switch.
    - **Approximate search**: ~100x faster using RII and Product Quantization.
    - **Thread-safe**: Background ANN building doesn't block new operations.
    - **Soft deletes**: Efficient deletion with minimal reorganization.
    - **Pure Python**: No external services or dependencies required.
    - **Distance metrics**: Supports Euclidean (L2) and dot product similarity.
    - **Persistent storage**: SQLite-backed durability.

Technical Background:
    Vink uses Reconfigurable Inverted Index (RII) with Product Quantization (PQ)
    for approximate nearest neighbor search. The switch from exact to approximate
    happens when sqrt(dim * vectors) / 1000 >= switch_ratio (default 4.0).

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
from vink.models import AnnConfig
from vink.exceptions import *

try:
    __version__ = version("vink")
except PackageNotFoundError:
    __version__ = "0.0.0"
