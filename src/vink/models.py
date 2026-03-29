from typing import Annotated, Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
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
    switch_exp: Annotated[float, Field(ge=0.25, le=5)] = Field(
        1.0,
        description="Power-law exponent for complexity: values >1.0 stay lower for small data but trigger the ANN switch faster after 1M total operations (D * N).",
    )
    reconfig_threshold: Annotated[int, Field(ge=5000)] = Field(
        100_000,
        description="Number of inserts before reconfiguring the index to maintain search performance.",
    )

    def validate_vector_dim(self, dim: int) -> None:
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
    embedding: Any = Field(
        default=None, description="Vector embedding. Validated and normalized by VectorRecords."
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v):
        """Validate an ID or generate a new UUIDv7. Always returns 16 bytes."""
        return validate_id(v)

class VectorRecords(BaseModel):
    """Container for multiple vector records with dimension enforcement."""

    dim: int = Field(gt=0, description="The required dimension for all embeddings.")
    metric: Literal["cosine", "euclidean"] = Field(description="Distance metric for vector normalization.")
    records: list[VectorRecord] = Field(fail_fast=True)
    embedding_callback: Callable[[str], np.ndarray] | None = Field(
        default=None,
        description="Optional function to generate vectors for records missing 'embedding' values.",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "VectorRecords":
        """Ensure all embeddings match the specified dimension and normalize if needed."""
        for i, record in enumerate(self.records):
            if record.embedding is None and self.embedding_callback is not None:
                record.embedding = self.embedding_callback(record.content)

            if record.embedding is not None:
                validated = validate_embedding(record.embedding, metric=self.metric)
                actual_dim = validated.shape[-1]
                if actual_dim != self.dim:
                    raise VectorDimensionError(
                        f"Dimension mismatch at record[{i}]. "
                        f"Expected {self.dim}, got {actual_dim}."
                    )
                record.embedding = validated
            else:
                raise InvalidInputError(
                    f"Record[{i}] is missing an embedding and no default callback is set."
                )
        return self
