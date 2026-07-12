# WAL for ApproximateSearch — Design Sketch

## Problem

`save()` serialises the entire ANN index to disk on every call. For large indices
this is expensive and wastes IO on data that hasn't changed. Meanwhile `add()`
returns immediately without any disk durability — crash between `add()` and
`save()` loses the mutation entirely.

The current shadow-file pattern (`*.pkl.tmp` → rename → fsync dir) protects
against torn writes during `save()`, but doesn't protect against crashes between
mutations. A WAL bridges that gap.

---

## Idea

Small mutations go into a **write-ahead log** (WAL). The full index is
snapshotted periodically. Recovery replays the WAL on top of the last snapshot.

```
add(vectors)
  └─ append to WAL (fsynced immediately or batched)
      └─ also insert into in-memory index (status quo)

save() / periodic threshold
  └─ write full index (shadow-file pattern)
  └─ truncate WAL

load() / crash recovery
  └─ load last snapshot
  └─ replay WAL records on top
```

---

## WAL Format (from `wal.py`)

Each frame: `[4-byte length][payload][4-byte CRC32]`

- **Length**: big-endian uint32 — number of bytes in payload
- **Payload**: pickle of the record tuple
- **CRC32**: of the payload — catches torn writes mid-frame

On replay, if a frame's length runs past EOF or the CRC doesn't match, the log
stops at the last fully-durable record. This tolerates any crash mid-append.

---

## WAL Records

Each record is a `(op, data)` tuple:

| op | data | meaning |
|---|---|---|
| `"add"` | `list[dict]` — vector records | vectors to insert into index |
| `"delete"` | `list[bytes]` — IDs | vectors to soft-delete from index |
| `"compact"` | — | hard-delete + rebuild (invalidates prior deletes) |

### Add Record Details

```python
("add", [
    {"id": b"...", "embedding": np.ndarray(...)},
    ...
])
```

Only what the ANN index needs — `id` + `embedding` (both bytes during pickle).
The full record (content, metadata) lives in SQLite, so we don't duplicate it
here.

### Delete Record Details

```python
("delete", [b"<id_bytes>", ...])
```

Soft-delete. On replay, these IDs are marked in `_mask` so the query path
skips them. A subsequent `compact` folds them permanently.

### Compact Record Details

```python
("compact", None)
```

A checkpoint marker. Everything before it has been folded into the snapshot.
On replay, stop at `compact` — no need to replay older records.

---

## Flow — Detailed

### `add(records)`

1. Compute embeddings, validate, insert into SQLite (status quo)
2. Insert into in-memory `self.index` (status quo)
3. **New**: `self.wal.append(("add", records))` — fsynced, durable

Batch path (e.g. 10 000 records in one call):

1. Same as above
2. Same as above
3. **New**: `self.wal.append_batch([("add", batch) for batch in chunks])` — one fsync

The batch path crucially means an add of 10K records requires exactly **one**
`fsync()`, not 10K of them. The trade-off: if the power dies during that single
batch write, the entire batch is lost (torn-write drops the whole last frame
on replay). That matches most user expectations — either all 10K survive or none
do.

For callers that need stricter durability, the WAL also exposes

```python
def append_nosync(self, record: Any) -> None: ...
def sync(self) -> None: ...
```

so the caller can record many small adds without fsyncing each one, then
call `sync()` once at a transaction boundary. This is the primitive
group-commit pattern.

### `delete(ids)`

1. Soft-delete in SQLite (status quo)
2. **New**: `self.wal.append(("delete", ids))`

### `soft_delete(ids)` (the bulk `WHERE id IN (...)` variant)

Same — append a single delete record with all IDs. If the power fails during the
WAL write, the delete is lost, but the IDs are still in SQLite as active, so the
next replay picks the right state. No permanent inconsistency.

### `compact()`

1. Hard-delete from SQLite (status quo)
2. Save full index, `self.wal.truncate()` — previous deletes are folded

`truncate()` empties the log file: close, reopen in write mode, fsync.
The next mutation starts a fresh WAL.

### `save()`

1. Write full index to `*.pkl.tmp` with fsync (status quo)
2. Rename → fsync dir (status quo)
3. **New**: `self.wal.truncate()` — snapshot is up to date, log is redundant

### `load()` — Recovery Path

```
load()
├── Main index exists? ─── yes ──→ try load main index
│                                      │
│                                    fail? ── yes ──→ try load shadow index
│                                      │                    │
│                                    success               fail? ──→ raise
│                                      │                    │
│                                      ↓                    ↓
│                                    loaded from main    loaded from shadow + swap
│                                      │                    │
│                                      └────────┬───────────┘
│                                               │
│                                               ↓
│                                        replay WAL on top
│                                               │
│                                        ┌──────┴──────┐
│                                        │ "add"  → index.add() + update _all_ids/_mask
│                                        │ "del"  → update _mask/_id_to_idx
│                                        │ "compact" → stop (folded)
│                                        └──────────────┘
│                                               │
│                                               ↓
│                                        truncate WAL
│                                               │
│                                               ↓
│                                        done
│
└── No index file? ──────────────→ try replaying WAL from scratch
                                        │
                                    WAL empty? ── yes ──→ nothing to load
                                        │
                                        ↓
                                    replay all records into empty index
                                        │
                                        ↓
                                    truncate WAL
                                        │
                                        ↓
                                    done — index rebuilt from mutation log
```

Key invariant: **the WAL is always a superset of the snapshot**. After a
successful `save()` + truncate they're equal. After a crash mid-`save()` the WAL
still contains the mutations that the partially-written snapshot may have missed.
Replay always brings the index to the correct state regardless of where the
crash hit.

#### Replay Algorithm — Pseudocode

```python
def _replay_wal(self) -> None:
    """Replay WAL records on top of the current index state."""
    records = self.wal.replay()
    for op, data in records:
        if op == "add":
            embeddings = np.vstack([r["embedding"] for r in data])
            self.index.add(embeddings)
            self._all_ids.extend(r["id"] for r in data)
            self._mask.extend([False] * len(data))
            for r in data:
                self._id_to_idx[r["id"]] = len(self._all_ids) - 1 - ...  # careful mapping
        elif op == "delete":
            for id_bytes in data:
                if id_bytes in self._id_to_idx:
                    self._mask[self._id_to_idx[id_bytes]] = True
        elif op == "compact":
            break  # everything before is already folded into the snapshot
    self.wal.truncate()
```

The compact break is important: after a `compact` + crash before `save()`, the
snapshot doesn't reflect the compact, but the WAL still has the compact record.
Since compact folds all prior deletes, replaying older records would double-count
them. Stopping at compact avoids that.

---

## Crash Scenarios — Exhaustive

| Crash between | State on disk | Recovery |
|---|---|---|
| `add()` SQLite insert + WAL append | SQLite has record, WAL has frame | Replay WAL → index correct |
| `add()` SQLite insert but WAL not yet flushed | SQLite has record, WAL missing frame | Replay WAL → index missing that record. Query via exact search will still find it (SQLite has it). If user re-adds, ANN gets duplicate — need dedup by ID on replay or guard in `add()`. **Risk** |
| mid-WAL-frame (torn write) | SQLite has record, WAL has broken tail | `wal.replay()` stops before torn frame. Same as above — record lost from index but present in SQLite |
| `save()` — after `.pkl.tmp` written, before rename | `.pkl.tmp` complete, `.pkl` old | Load tries `.pkl` first — gets old version, replays WAL — correct. Shadow recovery also works |
| `save()` — after rename, before dir fsync | `.pkl` = new, dir entry may not be durable | On ext4 with default options, rename is atomic. On async mounts, dir entry may be stale. If stale, next load finds old `.pkl`, replays WAL — correct. If not stale, new `.pkl` + WAL (empty after truncate) — correct |
| `save()` — after rename, before WAL truncate | `.pkl` new, WAL still has old records | Replay re-applies records already in snapshot. If add is idempotent (check by ID before inserting into index), this is safe. If not, **double-add risk** |
| `truncate()` mid-write | WAL empty or corrupt | If empty — good. If corrupt — replay stops at torn frame, likely returns 0 records, no replay. **Recovery succeeds but may miss pre-truncate records that weren't snapshotted yet** |

The dangerous case is **replaying a record that's already in the snapshot**.
Solution: make `add` replay idempotent — skip IDs already in `_id_to_idx`.

---

## Thread Safety

WAL is accessed from two threads:

| Thread | Mutates WAL? | Reads WAL? |
|---|---|---|
| Main (add/delete) | Yes — `append()` / `append_batch()` | No |
| ANN builder (save) | Yes — `truncate()` | No |
| Load (main, init) | Yes — `replay()` + `truncate()` | Yes — `replay()` |

The WAL file handle is a single `open(..., "ab")` — not thread-safe for
concurrent writes. Options:

**Option A — Lock**: a `threading.Lock` around every WAL method. Simple,
contention is negligible (main thread appends, builder thread truncates, never
at the same time in practice).

**Option B — Queue**: main thread writes into a `queue.Queue`, builder thread
drains and appends to WAL. Adds latency but decouples IO from the hot path.

**Option C — Two files**: main thread writes `wal_main.bin`, builder writes
`wal_builder.bin`. On load, replay both. Overkill.

Recommend **Option A** for now — simple, proven, matches how `_rwlock` already
works in the codebase.

---

## WAL Size Management

The WAL grows unbounded without a truncation policy. Truncation triggers:

| Trigger | Action |
|---|---|
| `save()` called explicitly | Truncate WAL after snapshot |
| WAL exceeds N bytes (configurable) | Auto-trigger `save()` in background |
| WAL exceeds M records (configurable) | Same |
| `compact()` called | Truncate WAL (deletes are folded) |

Default thresholds: 50 MB or 10 000 records, whichever hits first. Configurable
via `AnnConfig`:

```python
@dataclass
class AnnConfig:
    num_subspaces: int
    codebook_size: int
    switch_latency_ms: int = 120
    wal_max_bytes: int = 50 * 1024 * 1024   # 50 MB
    wal_max_records: int = 10_000
```

---

## Impact on Existing Code

### Files to create
- `src/vinkra/wal.py` — the `WriteAheadLog` class (already written as scratch)

### Files to modify
- `src/vinkra/strategies/approximate_search.py`
  - `__init__`: open WAL at `wal_path`
  - `add()`: append to WAL after index insert
  - `delete()`: append to WAL
  - `soft_delete()`: append to WAL
  - `save()`: truncate WAL after snapshot
  - `load()`: replay WAL after loading snapshot (or rebuild from WAL if no snapshot)
  - `_replay_wal()`: new method
  - `compact()`: truncate WAL

### Files unchanged
- `src/vinkra/sql_wrapper.py` — SQLite stays the system of record
- `src/vinkra/strategies/exact_search.py` — exact search doesn't need WAL
  (it queries SQLite directly, which has its own journal)

---

## Alternatives Considered

### 1. SQLite as the WAL (just use `records.sqlite` as the mutation log)
SQLite already has a WAL journal mode. We could re-open a second connection
in `wal_mode=True` and use it as a linear log table. Simpler (no new file
format), but couples the index-recovery logic to SQLite schema. The standalone
WAL is more portable and has zero schema dependencies.

### 2. Increase snapshot frequency (save on every N adds)
No WAL needed — just call `save()` every 1000 adds. Simple, but wastes IO for
large indices where most of the index is unchanged. WAL only writes the delta.

### 3. LMDB / RocksDB as the index backend
These have built-in crash recovery. Replacing `py_hnsw` with a transactional
key-value store would be a much larger change. The WAL approach layers crash
safety on top of the existing index without changing the index backend.

---

## Open Questions

- **Idempotent replay**: how to detect that an ID is already in `_all_ids` /
  `self.index` after a snapshot+replay overlap? Guard with `if id in _id_to_idx:
  continue`.
- **WAL format stability**: pickle is not forward-compatible across Python
  versions. For a local-only file this is fine, but worth documenting.
- **WAL vs shadow during save race**: main thread appends to WAL while builder
  thread is doing `save()`. The snapshot written by builder may be slightly
  stale (missing the last few adds). That's OK — replay fills the gap.
- **Recovery time**: for a WAL with 100K records, replay is 100K inserts into
  `self.index`. With py_hnsw this is ~seconds, not minutes. Worth measuring.
- **Should `wal.py` replace `wal.py` scratch or live in `src/vinkra/`**?
  The scratch file should graduate to the package.

---

## Migration Path

1. Move `wal.py` into `src/vinkra/wal.py` (with proper exports)
2. Add `wal_path` property to `ApproximateSearch`
3. Wire WAL into `add()` / `delete()` — mutation durability
4. Wire WAL into `load()` — recovery replay
5. Wire WAL into `save()` — truncation
6. Add size-based auto-save to prevent unbounded WAL growth
7. Update tests: add recovery tests that simulate crash mid-mutation
