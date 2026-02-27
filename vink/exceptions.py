class VinkError(Exception):
    """Base exception for all Vink errors."""
    pass


class VectorDimensionError(VinkError):
    """Raised when vector dimensions don't match the index configuration."""
    pass


class InvalidInputError(VinkError):
    """Raised when one or multiple invalid input(s) are encountered."""
    pass


class IndexNotBuiltError(VinkError):
    """Raised when attempting to query before building the index."""
    pass
