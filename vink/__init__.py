"""
Vink: A pure Python vector database using Reconfigurable Inverted Index (RII)
and Product Quantization (PQ).

Vink provides approximate nearest neighbor (ANN) search for high-dimensional
vectors with minimal memory footprint using the IVFADC/IVFPQ algorithm.

Features:
    - Fast and memory-efficient ANN search.
    - Subset search via linear PQ scan.
    - Reconfigurable index for dynamic datasets.
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
from vink.exceptions import *

try:
    __version__ = version("vink")
except PackageNotFoundError:
    __version__ = "0.0.0"
