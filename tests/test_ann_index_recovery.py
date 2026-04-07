from pathlib import Path

import larch.pickle as pickle
import numpy as np
import pytest

from vink.exceptions import DatabaseCorruptedError
from vink.models import AnnConfig, VectorRecords
from vink.sql_wrapper import SQLiteWrapper
from vink.strategies.approximate_search import ApproximateSearch
from vink.utils.id_generation import generate_id_bytes

DB_PATH = "records.sqlite"


def _create_bare_approx_strategy(dir_path: Path) -> ApproximateSearch:
    """Create an ApproximateSearch instance with given path."""
    config = AnnConfig(num_subspaces=4, codebook_size=8)

    return ApproximateSearch(
        db=SQLiteWrapper(str(dir_path / DB_PATH), index_config={}),
        dir_path=dir_path,
        dim=128,
        in_memory=False,
        metric="cosine",
        verbose=False,
        ann_config=config,
    )


@pytest.fixture
def sample_records_20():
    """Generate 20 sample records for testing."""
    rng = np.random.default_rng(42)
    records = []
    for i in range(20):
        vec = rng.standard_normal(128, dtype=np.float32)
        records.append(
            {
                "id": generate_id_bytes(),
                "content": f"test document {i}",
                "metadata": {"index": i},
                "embedding": vec,
            }
        )
    return records


@pytest.fixture
def approx_search_strategy(sample_records_20, tmp_path):
    """Create an ApproximateSearchStrategy instance for testing."""
    strategy = _create_bare_approx_strategy(tmp_path)

    # N must be greater than codebook_size (8) - use first 10 for fit
    first_10 = sample_records_20[:10]
    vectors = np.array([r["embedding"] for r in first_10], dtype=np.float32)
    ids = [r["id"] for r in first_10]

    strategy.fit(vectors, np.array(ids, dtype="S16"))

    # Use wrapper insert to simulate exact search data before the switch
    strategy.db.insert(VectorRecords(dim=128, metric="cosine", records=first_10))

    strategy.save()

    # Add remaining 10 vectors after save
    remaining_10 = sample_records_20[10:]
    strategy.add(VectorRecords(dim=128, metric="cosine", records=remaining_10))

    return strategy


def test_crash_during_temp_file_save(approx_search_strategy):
    """Corrupted .wal file, load should fall back to old index."""
    db_path = approx_search_strategy.dir_path

    # Corrupt the .wal file (simulate power cut during temp file write)
    approx_search_strategy._ann_index_wal_path.write_bytes(b"corrupted")

    # Close the fixture's connection to release lock
    approx_search_strategy.db.close()

    # Load should fall back to old ann_index.pkl (N=10)
    strategy = _create_bare_approx_strategy(db_path)
    strategy.load(overwrite=True)

    assert strategy.index.N == 10, f"Expected N=10 but got {strategy.index.N}"
    assert strategy.db.count("active") == 10, f"Expected count=10 but got {strategy.db.count('active')}"


def test_crash_before_db_commit(approx_search_strategy):
    """Temp file saved but db not committed - load falls back (N=10, count=10)."""
    db_path = approx_search_strategy.dir_path
    ann_index_wal_path = approx_search_strategy._ann_index_wal_path

    with open(ann_index_wal_path, "wb") as f:
        pickle.dump(approx_search_strategy.index, f, protocol=5)

    # Skip the commit
    # approx_search_strategy.db.commit()

    # Close the fixture's connection to release lock
    approx_search_strategy.db.close()

    # Load should fall back to old ann_index.pkl (N=10)
    strategy = _create_bare_approx_strategy(db_path)
    strategy.load(overwrite=True)

    assert strategy.index.N == 10, f"Expected N=10 but got {strategy.index.N}"
    assert strategy.db.count("active") == 10, f"Expected count=10 but got {strategy.db.count('active')}"


def test_power_cut_after_commit_before_swap(approx_search_strategy):
    """Db committed, temp file saved but not swapped - recovery should work (N=20)."""
    db_path = approx_search_strategy.dir_path
    ann_index_wal_path = approx_search_strategy._ann_index_wal_path

    with open(ann_index_wal_path, "wb") as f:
        pickle.dump(approx_search_strategy.index, f, protocol=5)

    approx_search_strategy.db.commit()

    # Close the fixture's connection to release lock
    approx_search_strategy.db.close()

    # Load should recover N=20, count=20
    strategy = _create_bare_approx_strategy(db_path)
    strategy.load(overwrite=True)

    assert strategy.index.N == 20, f"Expected N=29 but got {strategy.index.N}"
    assert strategy.db.count("active") == 20, f"Expected count=20 but got {strategy.db.count('active')}"


def test_main_index_corrupted_wal_missing(approx_search_strategy):
    """Main index corrupted, WAL missing - should raise DatabaseCorruptedError."""
    db_path = approx_search_strategy.dir_path
    ann_index_path = db_path / "ann_index.pkl"

    # Corrupt main index
    ann_index_path.write_bytes(b"corrupted")

    # Close the fixture's connection to release lock
    approx_search_strategy.db.close()

    # Load should raise DatabaseCorruptedError
    strategy = _create_bare_approx_strategy(db_path)
    with pytest.raises(DatabaseCorruptedError):
        strategy.load(overwrite=True)
