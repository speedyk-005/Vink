from typing import Annotated, Callable, Literal

import numpy as np
from pydantic import BaseModel, FailFast, Field, field_validator, model_validator
from pydantic.fields import PydanticUndefined

from vink.exceptions import InvalidInputError, VectorDimensionError
from vink.utils.id_generation import generate_id_bytes
from vink.utils.input_validation import validate_embedding, validate_id


class AnnConfig(BaseModel):
    """Configuration for Approximate Nearest Neighbor (ANN) search settings."""

    num_subspaces: Annotated[int, Field(ge=1)] = Field(
        32,
        description="The number of sub-vectors to split each embedding into. Must be a divisor of the embedding dimension.",
    )
    quantizer: Literal["pq", "opq"] = Field(
        "pq",
        description="The quantization algorithm. OPQ is more accurate but slower than PQ.",
    )
    codebook_size: Annotated[int, Field(ge=2)] = Field(
        256,
        description="Number of centroids per subspace. Affects memory usage and search accuracy.",
    )
    switch_ratio: Annotated[float, Field(ge=2, le=16)] = Field(
        4,
        description="Ratio threshold for switching between exact and approximate search. Recommended values are powers of 2.",
    )
    reconfig_threshold: Annotated[int, Field(ge=0)] = Field(
        1000,
        description="Number of inserts before reconfiguring the index to maintain search performance.",
    )

    def validate_dim(self, dim: int) -> None:
        """Validate ANN config against a specific vector dimension."""
        if self.num_subspaces > dim:
            raise VectorDimensionError(
                f"num_subspaces ({self.num_subspaces}) cannot exceed dim ({dim})."
            )

        if dim % self.num_subspaces != 0:
            remainder = dim / self.num_subspaces
            raise VectorDimensionError(
                f"Dimension ({dim}) must be divisible by num_subspaces ({self.num_subspaces}). "
                f"Result: {remainder:.2f}"
            )

    @classmethod
    def help(cls):
        """Print configuration arguments and descriptions."""
        print(f"{cls.__doc__}\n\nArgs:" if cls.__doc__ else "Args:")
        for name, field in cls.model_fields.items():
            has_default = field.default is not PydanticUndefined
            optional = ", optional" if has_default else ""
            default = f" Defaults to {field.default!r}" if has_default else ""

            print(
                f"    {name} ({field.annotation.__name__}{optional}): "
                f"{field.description}.{default}"
            )


class VectorRecord(BaseModel):
    """Model for a vector record entry."""

    model_config = {"arbitrary_types_allowed": True}

    id: str | bytes | None = Field(
        default_factory=lambda: generate_id_bytes(),
        description="UUIDv7 ID as bytes. Auto-generated if not provided.",
        validate_default=True,
    )
    content: str = Field(description="The text content or data to be indexed.")
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata as key-value pairs."
    )
    embedding: Annotated[list[float], FailFast] | np.ndarray | None = Field(
        None, description="Vector embedding. Auto-converted and L2-normalized."
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v):
        """Validate an ID or generate a new UUIDv7. Always returns 16 bytes."""
        return validate_id(v)

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, v):
        """Validate, cast, and L2-normalize input vectors."""
        if v is None:
            return v
        return validate_embedding(v)


class VectorRecords(BaseModel):
    """Container for multiple vector records with dimension enforcement."""

    dim: int = Field(gt=0, description="The required dimension for all embeddings.")
    records: list[VectorRecord] = Field(
        description="List of vector records to be indexed.", fail_fast=True
    )
    embedding_callback: Callable[[str], np.ndarray] | None = Field(
        default=None,
        description="Optional function to generate vectors for records missing 'embedding' values.",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "VectorRecords":
        """Ensure all embeddings match the specified dimension."""
        for i, record in enumerate(self.records):
            if record.embedding is None and self.embedding_callback is not None:
                record.embedding = self.embedding_callback(record.content)

            if record.embedding is not None:
                actual_dim = record.embedding.shape[-1]
                if actual_dim != self.dim:
                    raise VectorDimensionError(
                        f"Dimension mismatch at record[{i}]. "
                        f"Expected {self.dim}, got {actual_dim}."
                    )
            else:
                # If still None and no callback, it's an error
                raise InvalidInputError(
                    f"Record[{i}] is missing an embedding and no default callback is set."
                )
        return self
