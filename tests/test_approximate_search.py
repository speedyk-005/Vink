import numpy as np
import time
import pysqlite3 as sqlite3
import pytest

from vink.models import AnnConfig, VectorRecords
from vink.sql_wrapper import SQLiteWrapper
from vink.strategies.approximate_search import ApproximateSearch
from vink.utils.id_generation import generate_id_bytes

IDS_TO_DELETE = []


@pytest.fixture(scope="module")
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id BLOB PRIMARY KEY,
            content TEXT NOT NULL,
            metadata BLOB NOT NULL,
            embedding BLOB,
            deleted BOOLEAN DEFAULT 0
        )
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def approx_search_strategy(in_memory_db):
    """Create an ApproximateSearchStrategy instance for testing."""
    config = AnnConfig(num_subspaces=4, codebook_size=8)

    strategy = ApproximateSearch(
        db=SQLiteWrapper(":memory:"),
        dir_path=None,
        dim=128,
        in_memory=True,
        metric="euclidean",
        verbose=False,
        ann_config=config,
    )

    # N must be greater than codebook_size (8)
    num_training = 10
    rng = np.random.default_rng(seed=42)
    train_vectors = rng.standard_normal((num_training, 128), dtype=np.float32)

    norm = np.linalg.norm(train_vectors, axis=1, keepdims=True)
    train_vectors = train_vectors / (norm + 1e-9)

    # Generate active IDs for training vectors
    ids = [generate_id_bytes() for _ in range(num_training)]
    strategy.fit(train_vectors, np.array(ids, dtype="S16"))

    # Use wrapper insert to simulate exact search data before the switch
    records = [
        {"id": id, "content": "fit content", "metadata": {}, "embedding": vec}
        for id, vec in zip(ids, train_vectors)
    ]
    strategy.db.insert(VectorRecords(dim=128, metric="euclidean", records=records))

    global IDS_TO_DELETE
    IDS_TO_DELETE = ids  # Store them for the deletion test case

    return strategy


def test_add(approx_search_strategy, sample_embeddings):
    """
    Test adding vector records by checking if internal structures are synced and SQLite count.
    """
    records = [
        {
            "content": "content 1",
            "metadata": {"index": 1},
            "embedding": sample_embeddings,
        },
        {
            "content": "content 2",
            "metadata": {"index": 2},
            "embedding": sample_embeddings,
        },
    ]
    ids = approx_search_strategy.add(VectorRecords(dim=128, metric="euclidean", records=records))

    n_ids = len(approx_search_strategy.all_ids)
    n_map = len(approx_search_strategy.id_to_idx)

    # Including the ones added in the fitting process in the strategy fixture
    expected = 2 + 10

    assert n_ids == n_map == expected, (
        f"Sync Error! Expected {expected} records, but got: IDs={n_ids}, Map={n_map}"
    )

    cursor = approx_search_strategy.db.conn.cursor()
    db_count = approx_search_strategy.db.count()
    assert db_count == expected, f"Database count mismatch: {db_count} != {expected}"


def test_soft_delete(approx_search_strategy):
    """Test soft-deleting vector records by checking if they aren't active anymore"""
    approx_search_strategy.soft_delete(IDS_TO_DELETE)
    time.sleep(0.2)
    approx_search_strategy._ensure_cache()

    n_ids = len(approx_search_strategy.active_ids)
    n_mask = sum(approx_search_strategy.mask)
    expected = 2  # the two ones added in the test above

    assert n_ids == n_mask == expected, (
        f"Sync Error! Expected {expected} active records, but got: "
        f"IDs={n_ids}, Mask={n_mask}"
    )

    cursor = approx_search_strategy.db.conn.cursor()
    db_count = approx_search_strategy.db.count()
    assert db_count == expected, f"Database count mismatch: {db_count} != {expected}"


@pytest.mark.parametrize("sample_records", [{"num": 4}], indirect=True)
def test_search(approx_search_strategy, sample_records):
    """Test that search retrieves, ranks, and returns correct vector fields."""
    approx_search_strategy.add(VectorRecords(dim=128, metric="euclidean", records=sample_records))

    # Use the first embedding from sample_records as query
    query_embedding = sample_records[0]["embedding"]
    results = approx_search_strategy.search(
        query_embedding, top_k=4, include_vectors=True
    )
    id_to_res = {res["id"]: res for res in results}

    assert len(results) == 4, f"Expected 4 results, but got {len(results)}"

    for record in sample_records:
        rec_id_str = approx_search_strategy._bytes_to_uuid_str(record["id"])

        # Only validate if the record actually made it into the top_k
        if rec_id_str in id_to_res:
            res_item = id_to_res[rec_id_str]
            assert res_item["content"] == record["content"], (
                f"Content mismatch for {rec_id_str}"
            )
            assert res_item["metadata"] == record["metadata"], (
                f"Metadata mismatch for {rec_id_str}"
            )
            assert np.allclose(res_item["embedding"], record["embedding"]), (
                f"Embedding mismatch for {rec_id_str}"
            )


def test_compact(approx_search_strategy):
    """Test that compact hard-deletes soft-deleted records and rebuilds the ANN index."""
    # Add extra records so compact has enough vectors to rebuild the ANN index
    rng = np.random.default_rng(seed=42)
    extra_records = [
        {"id": generate_id_bytes(), "content": f"extra {i}", "embedding": rng.standard_normal(128).astype(np.float32), "metadata": {}}
        for i in range(5)
    ]
    approx_search_strategy.add(VectorRecords(dim=128, metric="euclidean", records=extra_records))

    index_before = approx_search_strategy.index
    approx_search_strategy.compact()
    time.sleep(0.2)

    assert approx_search_strategy.index is not None, "Index should be rebuilt"
    assert approx_search_strategy.index is not index_before, "Index should be a new instance"

    cursor = approx_search_strategy.db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM vec_records WHERE deleted = TRUE")
    assert cursor.fetchone()[0] == 0, "All soft-deleted records should be hard-deleted from SQLite"
