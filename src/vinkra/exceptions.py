class VinkraraDBError(Exception):
    """Base exception for all Vinkra errors."""


class InvalidInputError(VinkraraDBError):
    """Raised when one or multiple invalid input(s) are encountered."""


class VectorDimensionError(InvalidInputError):
    """Raised when vector dimensions don't match the index configuration."""


class InvalidIdError(InvalidInputError):
    """Raised when an invalid UUIDv7 is provided."""


class IndexNotFittedError(Exception):
    """Raised when an operation requiring learned quantization is called on an unitialized index."""


class FilterError(InvalidInputError):
    """Raised when a filter expression fails to parse."""


class DatabaseCorruptedError(VinkraraDBError):
    """Raised when database files are corrupted."""
