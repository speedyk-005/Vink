class VinkraraDBError(Exception):
    """Base exception for all Vinkra errors."""

    pass


class InvalidInputError(VinkraraDBError):
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


class FilterError(InvalidInputError):
    """Raised when a filter expression fails to parse."""

    pass


class DatabaseCorruptedError(VinkraraDBError):
    """Raised when database files are corrupted."""

    pass
