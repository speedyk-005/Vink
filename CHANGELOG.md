# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0a1] - 2026-04-08

### Added

- **Auto-Switch:** Automatic switch from exact to ANN search based on runtime latency
- **Incremental Inserts:** Add vectors without rebuilding index
- **Soft Deletes:** Mark vectors as deleted without rebuilding
- **Compact:** Reclaim storage from soft-deleted vectors
- **Save/Load:** Persist index and metadata to disk
- **Dual-Metric:** Euclidean and cosine similarity support
- **Basic Filtered Search:** Expression operators for filtering results
- **SQLite Storage:** Metadata, content, and embeddings stored in SQLite
