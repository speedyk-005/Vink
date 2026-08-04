import threading
import time

import pytest

from vinkra import VinkraDB
from vinkra.models import AnnConfig

DIM = 128


@pytest.fixture
def vinkdb(tmp_path, request):
    """Create a VinkraDB instance for testing."""
    params = getattr(request, "param", {})
    force_exact = params.get("force_exact", False)

    return VinkraDB(
        dir_path=tmp_path,
        dim=DIM,
        force_exact=force_exact,
        ann_config=AnnConfig(num_subspaces=4, codebook_size=4),
        verbose=False,
    )


def test_switch_triggers(vinkdb, sample_records, mocker):
    """Test that switch to ANN when _should_switch returns True."""
    assert vinkdb.strategy == "exact_search", "Should start with exact search"

    # Add first 7 records without triggering switch
    vinkdb.add(sample_records[:7])
    assert vinkdb.strategy == "exact_search", "Should still be exact after first batch"

    # Mock to trigger switch on next add
    mocker.patch.object(vinkdb, "_should_switch", return_value=True)
    vinkdb.add(sample_records[7:])

    # Poll until build completes (ANN runs in a background thread)
    timeout = 5
    start = time.time()
    while vinkdb.is_ann_building and (time.time() - start) < timeout:
        time.sleep(0.5)

    assert vinkdb.strategy == "approximate_search", (
        "Should switch to approximate search after build"
    )


@pytest.mark.parametrize("vinkdb", [{"force_exact": False}], indirect=True)
def test_recover_buffered_on_load(vinkdb, sample_records, mocker):
    """Simulate a crash mid-transition and verify recovery on reload."""
    vinkdb.add(sample_records[:7])
    assert vinkdb.strategy == "exact_search"
    assert vinkdb.has_buffered() is False

    # Trigger switch
    mocker.patch.object(vinkdb, "_should_switch", return_value=True)
    vinkdb.add(sample_records[7:])

    # Add more while building so they go to buffer
    vinkdb.add(sample_records[:2])

    # Simulate crash
    vinkdb.close()
    vinkdb.load()

    # Recovery should detect buffered records and restart the build
    timeout = 5
    start = time.time()
    while vinkdb.is_ann_building and (time.time() - start) < timeout:
        time.sleep(0.5)

    assert vinkdb.strategy == "approximate_search", (
        "Should recover and complete ANN build"
    )


@pytest.mark.parametrize("vinkdb", [{"force_exact": True}], indirect=True)
def test_force_exact(vinkdb):
    assert vinkdb.force_exact is True, "force_exact should be True"
    assert vinkdb.strategy == "exact_search", "Should stay exact when force_exact=True"
    assert vinkdb._should_switch() is False, "_should_switch should be False"


def test_close_joins_ann_build_thread(vinkdb, sample_records, mocker):
    """close() must join the in-flight ANN build before closing the DB.

    Regression test for the race where close() destroyed the SQLite
    connection while the background ANN transition thread was still
    replaying buffered records via cursor.execute().
    """
    vinkdb.add(sample_records[:7])
    assert vinkdb.strategy == "exact_search"

    # Trigger the switch, blocking the worker inside _prepare_approx_strategy
    started = threading.Event()
    release = threading.Event()

    def blocked_prepare():
        started.set()
        release.wait(timeout=5)

    mocker.patch.object(vinkdb, "_should_switch", return_value=True)
    mocker.patch.object(vinkdb, "_prepare_approx_strategy", side_effect=blocked_prepare)
    vinkdb.add(sample_records[7:])
    assert started.wait(timeout=5), "ANN build thread should have started"

    db_close = mocker.spy(vinkdb._records_db, "close")

    closer = threading.Thread(target=vinkdb.close)
    closer.start()

    # close() must block on join() while the build thread is still alive
    time.sleep(0.3)
    assert closer.is_alive(), "close() should block on join() while ANN thread is alive"
    assert db_close.call_count == 0, "DB must not close while ANN thread is alive"

    release.set()
    closer.join(timeout=5)
    assert not closer.is_alive(), "close() should finish after the ANN thread exits"
    assert db_close.call_count == 1, "DB should close after the ANN thread joins"


@pytest.mark.parametrize("vinkdb", [{"force_exact": True}], indirect=True)
def test_search_with_filter(vinkdb, sample_records):
    """Test that search with filter returns only matching records."""
    for i, record in enumerate(sample_records):
        record["metadata"]["category"] = "tech" if i % 2 == 0 else "science"

    vinkdb.add(sample_records)

    query_embedding = sample_records[0]["embedding"]
    results = vinkdb.search(query_embedding, top_k=4, filters=["category == 'tech'"])

    assert all(r["metadata"]["category"] == "tech" for r in results), (
        "All results should have category=tech"
    )
