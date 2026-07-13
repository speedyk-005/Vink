"""
Vinkra: Vector Incremental Nano Kit — Reconfigurated Automatically

A lightweight vector database that incrementally switches from exact to
approximate search as your data grows without full index rebuilds.

Technical Background:
    Vinkra uses Reconfigurable Inverted Index (RII) with Product Quantization (PQ)
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

try:
    __version__ = version("vinkra")
except PackageNotFoundError:
    __version__ = "0.0.0"

from vinkra.core import VinkraDB
from vinkra.exceptions import *
from vinkra.models import AnnConfig
