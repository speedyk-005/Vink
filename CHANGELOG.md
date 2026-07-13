# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **`has_buffered` property** ([#5](https://github.com/speedyk-005/vinkra/pull/5)): Check for buffered records via `db.has_buffered`.
- **`is_empty` property** ([#5](https://github.com/speedyk-005/vinkra/pull/5)): Check if there are no active records via `db.is_empty`.
- **ANN transition recovery** ([#5](https://github.com/speedyk-005/vinkra/pull/5)): `load()` replays buffered records when stale buffer data is detected, resuming an interrupted transition.

### Changed

- **ANN transition recovery** ([#5](https://github.com/speedyk-005/vinkra/pull/5)): `load()` replays buffered records when stale buffer data is detected, resuming an interrupted transition.

## [0.2.0a2] - 2026-07-13

### Fixed

- **Close idempotency**: `close()` now flips `_closed` after save, so atexit and manual calls don't double-close.
- **Dim comparison type mismatch**: Cast both sides to `int` before comparing.

## [0.2.0a1] - 2026-07-13

### Changed

- **Auto-save on exit**: `close()` registered via `atexit`; idempotent guard.
- **Enable SQLite mmap**: Added `PRAGMA mmap_size=256MB` for reduced read overhead.
- **Switch FTS5 tokenizer to trigram**: Replaced `unicode61` with `trigram` tokenizer for substring content matching.
- **`count()` defaults to active vectors**: Added `"all"` literal for explicit total count.
- **`is_ann_building` property**: Public read-only access to ANN build status.
- **Code quality improvements** ([#2](https://github.com/speedyk-005/vinkra/pull/2)):
  - Removed `in_memory` parameter; persistence inferred from `dir_path` instead.
  - Moved `FilterToSql.translate` into `BaseStrategy`, deduplicated per-strategy implementations.
  - Replaced `VectorRecord` objects with plain `list[dict]` through the strategy chain.
  - Exposed `close()` method; suppressed noisy warnings on teardown.
  - Stripped type hints from docstrings; removed `_format_docstrings.py` and `_strip_docstring_types.py`.
  - Expanded ruff lint rules (FBT, ARG, RET, ...); fixes applied throughout.
  - Updated README with professional tone and correct usage

### Fixed

- **Clamp `top_k` to SQLite parameter limit**: Warn and clamps to 32766 to avoid hitting the SQLite max-params limit.

## [0.1.0a2] - 2026-07-10

### Changed

- **stdlib pickle**: Replaced `larch-pickle` dependency with Python's stdlib `pickle`.
- **Relaxed `pydantic` constraint**: Lowered minimum from `2.12.2` to `2.10.0` for Android/Termux compatibility.

---

## [0.1.0a1] - 2026-04-08

### Added

- **Auto-switch**: Automatic switch from exact to ANN search based on runtime latency.
- **Incremental inserts**: Add vectors without rebuilding index.
- **Soft deletes**: Mark vectors as deleted without rebuilding.
- **Compact**: Reclaim storage from soft-deleted vectors.
- **Save/Load**: Persist index and metadata to disk.
- **Dual-metric**: Euclidean and cosine similarity support.
- **Basic filtered search**: Expression operators for filtering results.
- **SQLite storage**: Metadata, content, and embeddings stored in SQLite.
