import numpy as np
import pytest

from vink.utils.id_generation import generate_id_bytes

DIM = 128
SEED = 42


@pytest.fixture
def sample_embeddings(request):
    """Generate sample embeddings with optional normalization.

    Usage:
        # Default (not normalized)
        def test_foo(sample_embeddings): ...

        # Normalized
        @pytest.mark.parametrize("sample_embeddings", [{"normalize": True}], indirect=True)
        def test_bar(sample_embeddings): ...
    """
    params = getattr(request, "param", {"normalize": False})

    rng = np.random.default_rng(SEED)
    embeddings = rng.standard_normal((1, DIM), dtype=np.float32)

    if params.get("normalize"):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norm + 1e-9)

    return embeddings


@pytest.fixture
def sample_records(request):
    """Generate static test records with hardcoded seed.

    Usage:
        # Default (10 records)
        def test_foo(sample_records): ...

        # Custom num
        @pytest.mark.parametrize("sample_records", [{"num": 5}], indirect=True)
        def test_bar(sample_records): ...
    """
    params = getattr(request, "param", {"num": 10})
    num = params.get("num", 10)

    rng = np.random.default_rng(SEED)
    records = []
    for i in range(num):
        vec = rng.standard_normal(DIM, dtype=np.float32)
        norm = np.linalg.norm(vec)
        vec = vec / norm
        records.append(
            {
                "id": generate_id_bytes(),
                "content": f"test document {i}",
                "metadata": {"index": i},
                "embedding": vec,
            }
        )
    return records
