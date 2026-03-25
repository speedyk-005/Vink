import time
from pathlib import Path

import numpy as np
import pytest

from vink.models import VectorRecords
from vink.sql_wrapper import SQLiteWrapper
from vink.strategies.exact_search import ExactSearch
from vink.utils.id_generation import generate_id_bytes

IDS_TO_DELETE = [generate_id_bytes() for _ in range(2)]


@pytest.fixture(scope="module")
def exact_search_strategy():
    """Create an ExactSearchStrategy instance for testing."""
    return ExactSearch(
        db=SQLiteWrapper(":memory:", index_config={}),
        dir_path=None,
        dim=128,
        in_memory=True,
        metric="euclidean",
        verbose=False,
    )


def test_add(exact_search_strategy, sample_embeddings):
    """
    Test adding vector records by checking if internal structures are synced and SQLite count.
    """
    records = [
        {
            "id": IDS_TO_DELETE[0],
            "content": "content 1",
            "metadata": {"index": 1},
            "embedding": sample_embeddings,
        },
        {
            "id": IDS_TO_DELETE[1],
            "content": "content 2",
            "metadata": {"index": 2},
            "embedding": sample_embeddings,
        },
        {
            "content": "content 3",
            "metadata": {"index": 3},
            "embedding": sample_embeddings,
        },
        {
            "content": "content 4",
            "metadata": {"index": 4},
            "embedding": sample_embeddings,
        },
    ]
    exact_search_strategy.add(VectorRecords(dim=128, metric="euclidean", records=records))

    n_ids = len(exact_search_strategy._all_ids)
    n_vecs = len(exact_search_strategy._all_vectors)
    n_map = len(exact_search_strategy._id_to_idx)
    expected = 4

    assert n_ids == n_vecs == n_map == expected, (
        f"Sync Error! Expected {expected} records, but got: "
        f"IDs={n_ids}, Vectors={n_vecs}, Map={n_map}"
    )

    active_count = exact_search_strategy.db.count("active")
    assert active_count == expected, f"Database count mismatch: {active_count} != {expected}"


def test_soft_delete(exact_search_strategy):
    """Test soft-deleting vector records by checking if they aren't active anymore"""
    exact_search_strategy.soft_delete(IDS_TO_DELETE)
    time.sleep(0.2)
    exact_search_strategy._ensure_cache()

    n_ids = len(exact_search_strategy.active_ids_arr)
    n_vecs = len(exact_search_strategy.active_vectors_arr)
    n_mask = sum(exact_search_strategy._mask)
    expected = 2

    assert n_ids == n_vecs == n_mask == expected, (
        f"Sync Error! Expected {expected} active records, but got: "
        f"IDs={n_ids}, Vectors={n_vecs}, Mask={n_mask}"
    )

    active_count = exact_search_strategy.db.count("active")
    assert active_count == expected, f"Database count mismatch: {active_count} != {expected}"


@pytest.mark.parametrize("sample_records", [{"num": 4}], indirect=True)
def test_search(exact_search_strategy, sample_records):
    """Test that search retrieves, ranks, and returns correct vector fields."""
    exact_search_strategy.add(VectorRecords(dim=128, metric="euclidean", records=sample_records))

    # Use the first embedding from sample_records as query
    query_embedding = sample_records[0]["embedding"]
    results = exact_search_strategy.search(
        query_embedding, top_k=4, include_vectors=True
    )
    id_to_res = {res["id"]: res for res in results}

    assert len(results) == 4, f"Expected 4 results, but got {len(results)}"

    for record in sample_records:
        rec_id_str = exact_search_strategy._bytes_to_uuid_str(record["id"])

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


def test_compact(exact_search_strategy):
    """Test that compact hard-deletes soft-deleted records and rebuilds in-memory structures."""
    ids_before = len(exact_search_strategy._all_ids)

    exact_search_strategy.compact()
    time.sleep(0.2)

    assert len(exact_search_strategy._all_ids) == ids_before - len(IDS_TO_DELETE), "all_ids should be slimmed by deleted count"
    assert len(exact_search_strategy._mask) == len(exact_search_strategy._all_ids), "mask length should match all_ids"
    assert all(exact_search_strategy._mask), "All entries in mask should be True"

    deleted_count = exact_search_strategy.db.count("deleted")
    assert deleted_count == 0, "All soft-deleted records should be hard-deleted from SQLite"


def test_save_load(sample_embeddings, tmp_path):
    """Test that save persists data and load restores it correctly."""
    tmp_path = Path(tmp_path)

    db = SQLiteWrapper(f"{tmp_path}/records.sqlite", index_config={})
    strategy = ExactSearch(
        db=db,
        dir_path=tmp_path,
        dim=128,
        in_memory=False,
        metric="euclidean",
        verbose=False,
    )

    records = [
        {"content": f"content {i}", "metadata": {"i": i}, "embedding": sample_embeddings}
        for i in range(3)
    ]
    original_ids = strategy.add(VectorRecords(dim=128, metric="euclidean", records=records))

    strategy.save()

    strategy2 = ExactSearch(
        db=SQLiteWrapper(f"{tmp_path}/records.sqlite", index_config={}),
        dir_path=tmp_path,
        dim=128,
        in_memory=False,
        metric="euclidean",
        verbose=False,
    )
    strategy2.load(overwrite=True)

    assert len(strategy2._all_ids) == 3, f"Expected 3 IDs, got {len(strategy2._all_ids)}"
    assert len(strategy2._all_vectors) == 3, f"Expected 3 vectors, got {len(strategy2._all_vectors)}"
    assert len(strategy2._id_to_idx) == 3, f"Expected 3 id_to_idx entries, got {len(strategy2._id_to_idx)}"
    new_ids = [strategy2._bytes_to_uuid_str(id_bytes) for id_bytes in strategy2._all_ids]
    assert set(new_ids) == set(original_ids), "Loaded IDs don't match original IDs"
