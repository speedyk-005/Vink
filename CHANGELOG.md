# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **Removed `in_memory` parameter** ([#1](https://github.com/speedyk-005/vinkra/pull/1)):
  Persistence is now inferred from `dir_path` instead of a separate boolean. Pass `None` for in-memory, or a path for persistent storage.

## [0.1.0a2] - 2026-07-10

### Changed

- **stdlib pickle**: Replaced `larch-pickle` dependency with Python's stdlib `pickle`.
- **Relaxed `pydantic` constraint**: Lowered minimum from `2.12.2` to `2.10.0` for Android/Termux compatibility.

### Added

- **Termux install guide**: Installation tip for Android users building without a Rust toolchain.

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
