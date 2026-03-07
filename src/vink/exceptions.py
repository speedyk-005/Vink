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
    """Raised when an invalid UUIDv7 is provided."""
    pass


class IndexNotFittedError(Exception):
    """Raised when an operation requiring learned quantization is called on an unitialized index."""
    pass
