"""
Vink: A pure Python vector database with hybrid exact/approximate nearest neighbor search.

Vink automatically switches between exact brute-force search and approximate 
nearest neighbor (ANN) search based on dataset size, using Reconfigurable 
Inverted Index (RII) and Product Quantization (PQ) for efficient ANN.

Features:
    - Hybrid search: exact for small datasets, ANN for large datasets.
    - Automatic strategy switching based on configurable ratio.
    - L2-normalized embeddings for consistent distance metrics.
    - Supports Euclidean (L2) and dot product similarity.
    - Soft deletes: efficient deletion without data reorganization.
    - Serverless: no external services required.

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
from vink.models import ANNConfig
from vink.exceptions import *

try:
    __version__ = version("vink")
except PackageNotFoundError:
    __version__ = "0.0.0"
