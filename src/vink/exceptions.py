class VinkError(Exception):
    """Base exception for all Vink errors."""
    pass


class InvalidInputError(VinkError):
    """Raised when one or multiple invalid input(s) are encountered."""
    pass


class VectorDimensionError(InvalidInputError):
    """Raised when vector dimensions don't match the index configuration."""
    pass


class InvalidIdError(InvalidInputError):
    """Raised when an invalid UUID is provided."""
    pass


class IndexNotBuiltError(VinkError):
    """Raised when attempting to query before building the index."""
    pass
