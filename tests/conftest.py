import numpy as np
import pytest

from vinkra.utils.id_generation import generate_id_bytes
from vinkra.utils.input_validation import validate_embedding

DIM = 128
SEED = 42


@pytest.fixture
def sample_embeddings(request):
    """Generate sample embeddings using validate_embedding.

    Usage:
        # Default (euclidean)
        def test_foo(sample_embeddings): ...

        # Cosine metric
        @pytest.mark.parametrize("sample_embeddings", [{"metric": "cosine"}], indirect=True)
        def test_bar(sample_embeddings): ...
    """
    params = getattr(request, "param", {"metric": "euclidean"})
    metric = params.get("metric", "euclidean")

    rng = np.random.default_rng(SEED)
    embeddings = rng.standard_normal((1, DIM), dtype=np.float32)

    return validate_embedding(embeddings, dim=DIM, metric=metric)


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
        records.append(
            {
                "id": generate_id_bytes(),
                "content": f"test document {i}",
                "metadata": {"index": i},
                "embedding": vec,
            }
        )
    return records
