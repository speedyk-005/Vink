import numpy as np
import pytest

from vink import VinkDB
from vink.exceptions import InvalidIdError, InvalidInputError, VectorDimensionError
from vink.models import VectorRecord, VectorRecords


def mock_embedding_callback(text: str) -> np.ndarray:
    """A valid mock callback returning a 128-dim vector."""
    return np.random.rand(128).astype(np.float32)


def mock_invalid_dim_callback(text: str) -> np.ndarray:
    """An invalid mock callback returning the wrong dimension."""
    return np.random.rand(64).astype(np.float32)


def mock_crashing_callback(text: str) -> np.ndarray:
    """A callback that simulates a model failure."""
    raise RuntimeError("Model server down")


@pytest.mark.parametrize(
    "id_val, match",
    [
        ("not-a-uuid", "Invalid UUIDv7 format"),
        ("12345678-1234-5678-1234-567812345678", "Not a UUIDv7"),
        (b"not-16-bytes", "Invalid UUID bytes length"),
        (12345, "ID must be str or bytes"),
    ],
)
def test_invalid_id(id_val, match):
    """Test that invalid ID raises InvalidIdError."""
    embedding = np.random.rand(128).astype(np.float32)
    with pytest.raises(InvalidIdError, match=match):
        VectorRecord(id=id_val, content="test", embedding=embedding)


@pytest.mark.parametrize(
    "callback, expected_exc, match",
    [
        (mock_embedding_callback, None, ""),  # Success case
        (
            mock_invalid_dim_callback,
            VectorDimensionError,
            "Embedding callback output dimension",
        ),
        (mock_crashing_callback, InvalidInputError, "Embedding callback crashed"),
        (
            lambda x: "not-a-vector",
            InvalidInputError,
            "Vector components must be numeric",
        ),
        (None, InvalidInputError, r"Record\[0\] is missing an embedding"),
    ],
)
def test_vinkdb_init_handshake(callback, expected_exc, match, tmp_path):
    """Test the embedding_callback handshake during VinkDB initialization."""
    if expected_exc:
        with pytest.raises(expected_exc, match=match):
            db = VinkDB(dir_path=tmp_path, dim=128, embedding_callback=callback)
            if not callback:
                db.add([{"content": "test"}])
    else:
        # Should initialize without error
        db = VinkDB(dir_path=tmp_path, dim=128, embedding_callback=callback)
        assert db.embedding_callback == callback


def test_vinkdb_lazy_embedding(tmp_path):
    """Test that VinkDB uses the callback to populate missing embeddings in add()."""
    dim = 128
    db = VinkDB(dir_path=tmp_path, dim=dim, embedding_callback=mock_embedding_callback)

    # Record missing the 'embedding' key
    records = [{"content": "hello world"}]
    ids = db.add(records)

    assert len(ids) == 1
    # Verify the record actually got a vector in the DB (via search)
    results = db.search(np.random.rand(dim), top_k=1)
    assert len(results) == 1


@pytest.mark.parametrize(
    "embedding, exc_type",
    [
        ([[1, 2, 3], [4, 5, 6]], VectorDimensionError),  # Too many rows
        ("not-an-array", InvalidInputError),  # String
        (np.zeros(128), InvalidInputError),  # Zero-magnitude
        # --- Valid Cases ---
        (np.random.rand(128), None),  # 1D array
        ([1.0] * 128, None),  # List of floats
        (np.random.rand(1, 128), None),  # 2D row vector
    ],
)
def test_embedding_validation(embedding, exc_type):
    """Test that embedding validation correctly flags errors or allows valid inputs."""
    if exc_type:
        with pytest.raises(exc_type):
            VectorRecord(content="test", embedding=embedding)
    else:
        # Should not raise any exception
        record = VectorRecord(content="test", embedding=embedding)
        assert record.embedding.shape == (1, 128)
        assert np.isclose(np.linalg.norm(record.embedding), 1.0)


def test_dimension_mismatch():
    """Test that dimension mismatch raises VectorDimensionError."""
    expected_dim = 128
    embedding = np.random.rand(10)
    with pytest.raises(VectorDimensionError):
        VectorRecords(
            dim=expected_dim, records=[{"content": "test", "embedding": embedding}]
        )
