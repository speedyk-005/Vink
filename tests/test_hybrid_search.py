import time
import pytest

from vink import VinkDB
from vink.models import ANNConfig


DIM = 128


@pytest.fixture
def vinkdb(tmp_path, request, mocker):
    """Create a VinkDB instance for testing."""
    params = getattr(request, "param", {})
    force_exact = params.get("force_exact", False)
    
    db = VinkDB(
        dir_path=tmp_path,
        dim=DIM,
        force_exact=force_exact,
        ann_config=ANNConfig(num_subspaces=4, codebook_size=4),
        verbose=False,
    )
    
    return db


def test_switch_triggers(vinkdb, sample_records, mocker):
    """Test that switch to ANN when _should_switch returns True."""
    assert vinkdb.strategy == "exact_search", "Should start with exact search"

    # Add first 7 records without triggering switch
    vinkdb.add(sample_records[:7])
    assert vinkdb.strategy == "exact_search", "Should still be exact after first batch"

    # Mock to trigger switch on next add
    mocker.patch.object(vinkdb, '_should_switch', return_value=True)
    vinkdb.add(sample_records[7:])

    assert vinkdb._ann_building is True, "ANN building should be in progress"
    
    # Poll until build completes (max 5 seconds)
    timeout = 5
    start = time.time()
    while vinkdb._ann_building and (time.time() - start) < timeout:
        time.sleep(0.5)

    assert vinkdb.strategy == "approximate_search", "Should switch to approximate search after build"


@pytest.mark.parametrize("vinkdb", [{"force_exact": True}], indirect=True)
def test_force_exact(vinkdb, sample_records, mocker):
    """Test that force_exact prevents ANN switching."""
    mocker.patch.object(vinkdb, '_should_switch', return_value=True)
    vinkdb.add(sample_records)
    
    # Stays exact despite mock
    assert vinkdb.strategy == "exact_search", "Should stay exact when force_exact=True"
