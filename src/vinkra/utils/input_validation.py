import reprlib
from functools import wraps
from uuid import UUID

import numpy as np
from pydantic import ConfigDict, ValidationError, validate_call

from vinkra.exceptions import InvalidIdError, InvalidInputError, VectorDimensionError


def pretty_errors(error: ValidationError) -> str:
    """
    Formats Pydantic validation errors into a clean, human-readable summary.

    Args:
        error (ValidationError): The Pydantic error object to format.

    Returns:
        str: A scannable string containing error counts, locations, and input types.
    """
    lines = [
        f"{error.error_count()} validation error for {getattr(error, 'subtitle', '') or error.title}."
    ]
    for ind, err in enumerate(error.errors(), start=1):
        msg = err["msg"]

        loc = err.get("loc", [])
        formatted_loc = ""
        if len(loc) >= 1:
            formatted_loc = str(loc[0]) + "".join(f"[{step!r}]" for step in loc[1:])
            formatted_loc = f"({formatted_loc})" if formatted_loc else ""

        input_value = err["input"]
        input_type = type(input_value).__name__

        # reprlib.repr adds quotes and ellipsis around string truncation, which
        # looks odd for long strings. Plain slice is cleaner and faster for str.
        if isinstance(input_value, str):
            input_repr = (
                input_value[:200] + "..." if len(input_value) > 200 else input_value
            )
        else:
            input_repr = reprlib.repr(input_value)

        lines.append(
            f"{ind}) {formatted_loc} {msg}.\n"
            f"  Found: (input={input_repr}, type={input_type})"
        )

    # Append hint if available
    hint = getattr(error, "hint", "")
    if hint:
        lines.append(f"  Hint: {hint}")

    return "\n".join(lines)


def validate_arguments(fn):
    """
    Decorator that enforces type safety on function inputs and outputs.

    Args:
        fn (Callable): The function to be validated.

    Returns:
        Callable: A wrapped function that re-raises Pydantic errors as InvalidInputError.
    """
    validated_fn = validate_call(fn, config=ConfigDict(arbitrary_types_allowed=True))

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return validated_fn(*args, **kwargs)
        except ValidationError as e:
            # Re-raise with the human-readable summary from pretty_errors
            raise InvalidInputError(pretty_errors(e)) from None

    return wrapper


def validate_embedding(
    vecs: list[float] | np.ndarray, dim: int, metric: str
) -> np.ndarray:
    """
    Validate and optionally normalize input vectors.

    Ensures the input is a valid numeric array and enforces a 2D (1, d) shape.
    Normalization is only applied for cosine metric; euclidian skips normalization.

    Args:
        vecs (list[float] | np.ndarray): Input embedding. Accepts 1D arrays of
            shape (d,) or 2D row vectors of shape (1, d).
        dim (int): The required dimension for the embedding.
        metric (str): Distance metric.

    Returns:
        np.ndarray: A float32 row vector of shape (1, d). Normalized for cosine,
            raw for euclidean.

    Raises:
        InvalidInputError: If the input contains non-numeric values or (for cosine)
            has a zero-magnitude (null) norm.
        VectorDimensionError: If the input dimensionality or shape is incompatible
            with a single-vector representation.
    """
    # Cast to float32 immediately to catch non-numeric types
    try:
        vecs = np.asarray(vecs, dtype=np.float32)
    except (ValueError, TypeError):
        raise InvalidInputError(
            "Vector components must be numeric (int or float)."
        ) from None

    # Validate shape: must be 1D (d,) or 2D with single row (1, d)
    if not (vecs.ndim == 1 or (vecs.ndim == 2 and vecs.shape[0] == 1)):
        raise VectorDimensionError(
            f"Embedding must be (d,) or (1, d). Got shape {vecs.shape}."
        )

    actual_dim = vecs.shape[-1]
    if actual_dim != dim:
        raise VectorDimensionError(
            f"Dimension mismatch. Expected {dim}, got {actual_dim}."
        )

    # Standardize to 2D row vector
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)

    if metric == "cosine":
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Prevent division by zero / NaN poisoning
        if np.any(norm < 1e-9):
            raise InvalidInputError(
                "Cannot normalize a zero-magnitude vector. "
                "Ensure the embedding contains non-zero variance."
            )
        return vecs / norm

    return vecs


def validate_id(id: str | bytes) -> bytes:
    """
    Validate an ID or generate a new UUIDv7. Always returns 16 bytes.

    Args:
        id (str | bytes): UUIDv7 as string or bytes.

    Returns:
        bytes: 16-byte binary UUIDv7.
    """
    try:
        # Explicit type check to prevent AttributeError in UUID constructor
        if not isinstance(id, (str, bytes)):
            raise InvalidIdError(f"ID must be str or bytes, got {type(id).__name__}")

        if isinstance(id, bytes):
            if len(id) != 16:
                raise InvalidIdError(
                    f"Invalid UUID bytes length: expected 16, got {len(id)}"
                )
            val = UUID(bytes=id)
        else:
            val = UUID(id)

        if val.version != 7:
            raise InvalidIdError(f"Not a UUIDv7: {id}")

        return val.bytes

    except (ValueError, TypeError):
        raise InvalidIdError(f"Invalid UUIDv7 format: {id}") from None
