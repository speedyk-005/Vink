import pytest
import numpy as np

import pysqlite3 as sqlite3

from vink.models import VectorRecords
from vink.strategies.exact_search import ExactSearch
from vink.utils.id_generation import generate_id_bytes


IDS_TO_DELETE = [generate_id_bytes() for _ in range(2)]

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
def exact_search_strategy(in_memory_db):
    """Create an ExactSearchStrategy instance for testing."""
    return ExactSearch(
        conn=in_memory_db,
        dir_path=None,
        dim=128,
        in_memory=True,
        metric="l2",
        verbose=False,
    )


def test_add(exact_search_strategy, sample_embeddings):
    """
    Test adding vector records by checking if internal structures are synced and SQLite count.
    """
    records=[
        {"id": IDS_TO_DELETE[0], "content": "content 1", "metadata": {"index": 1}, "embedding": sample_embeddings},
        {"id": IDS_TO_DELETE[1], "content": "content 2", "metadata": {"index": 2}, "embedding": sample_embeddings},
        {"content": "content 3", "metadata": {"index": 3}, "embedding": sample_embeddings},
        {"content": "content 4", "metadata": {"index": 4}, "embedding": sample_embeddings},
    ]
    ids = exact_search_strategy.add(VectorRecords(dim=128, records=records))

    n_ids = len(exact_search_strategy.ids)
    n_vecs = len(exact_search_strategy.vectors)
    n_map = len(exact_search_strategy.id_to_idx)
    expected = 4

    assert n_ids == n_vecs == n_map == expected, (
        f"Sync Error! Expected {expected} records, but got: "
        f"IDs={n_ids}, Vectors={n_vecs}, Map={n_map}"
    )

    cursor = exact_search_strategy.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM records WHERE deleted = 0")
    db_count = cursor.fetchone()[0]
    assert db_count == expected, f"Database count mismatch: {db_count} != {expected}"


def test_delete(exact_search_strategy):
    """Test deleting vector records by checking if they aren't active anymore"""
    exact_search_strategy.delete(IDS_TO_DELETE)
    exact_search_strategy._ensure_cache()

    n_ids = len(exact_search_strategy.active_ids)
    n_vecs = len(exact_search_strategy.active_vectors)  
    n_mask = sum(exact_search_strategy.mask)
    expected = 2

    assert n_ids == n_vecs == n_mask == expected, (
        f"Sync Error! Expected {expected} active records, but got: "
        f"IDs={n_ids}, Vectors={n_vecs}, Mask={n_mask}"
    )

    cursor = exact_search_strategy.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM records WHERE deleted = 0")
    db_count = cursor.fetchone()[0]
    assert db_count == expected, f"Database count mismatch: {db_count} != {expected}"


@pytest.mark.parametrize("sample_records", [{"num": 4}], indirect=True)
def test_search(exact_search_strategy, sample_records):
    """Test that search retrieves, ranks, and returns correct vector fields."""
    exact_search_strategy.add(VectorRecords(dim=128, records=sample_records))

    # Use the first embedding from sample_records as query
    query_embedding = sample_records[0]["embedding"]
    results = exact_search_strategy.search(query_embedding, top_k=4, include_vectors=True)
    id_to_res = {res["id"]: res for res in results}

    assert len(results) == 4, f"Expected 4 results, but got {len(results)}"

    for record in sample_records:
        rec_id_str = exact_search_strategy._bytes_to_uuid_str(record["id"])
        
        # Only validate if the record actually made it into the top_k
        if rec_id_str in id_to_res:
            res_item = id_to_res[rec_id_str]
            assert res_item["content"] == record["content"], f"Content mismatch for {rec_id_str}"
            assert res_item["metadata"] == record["metadata"], f"Metadata mismatch for {rec_id_str}"
            assert np.allclose(res_item["embedding"], record["embedding"]), f"Embedding mismatch for {rec_id_str}"
        