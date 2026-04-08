# 🐦 Vink

<p align="center">
  <img src="https://github.com/speedyk-005/vink/blob/main/vink_logo.png?raw=true" alt="Vink Logo" width="300"/>
</p>

<p align="center">
  <b>V</b>ector <b>In</b>cremental <b>N</b>ano <b>K</b>it
</p>
<p align="center">
  “Vector DB that self-organize. Auto-switch, Auto-tune, Auto-scale.”
</p>

[![Python Version](https://img.shields.io/badge/Python-3.9%20--%203.14-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/vink)](https://pypi.org/project/vink)
[![CodeFactor](https://www.codefactor.io/repository/github/speedyk-005/vink/badge)](https://www.codefactor.io/repository/github/speedyk-005/vink)
[![Coverage Status](https://coveralls.io/repos/github/speedyk-005/vink/badge.svg?branch=main)](https://coveralls.io/github/speedyk-005/vink?branch=main)
[![Stability](https://img.shields.io/badge/stability-pre--alpha-yellow)](https://github.com/speedyk-005/vink)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/speedyk-005/vink/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> This project is currently in pre-alpha.

---

## Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [🤔 So What's vink Anyway? (And Why Should You Care?)](#-so-whats-vink-anyway-and-why-should-you-care)
- [📦 Installation](#-installation)
  - [The Quick & Easy Way](#the-quick--easy-way)
  - [The From-Source Way](#the-from-source-way)
- [🚀 Usage](#-usage)
  - [Initialization (VinkDB API)](#initialization-vinkdb-api)
    - [AnnConfig (API)](#annconfig-api)
  - [Add (API)](#add-api)
    - [With embedding callback](#with-embedding-callback)
    - [Without callback](#without-callback)
  - [Search (API)](#search-api)
    - [Without filters](#without-filters)
    - [With filters](#with-filters)
  - [Delete](#delete)
    - [Soft deletion (API)](#soft-deletion-api)
    - [Compaction (API)](#compaction-api)
  - [Stats (API)](#stats-api)
- [🚨 Exceptions (API)](#-exceptions-api)
- [🗺 Features & Roadmap](#-features--roadmap)
- [🔧 Core Dependencies](#-core-dependencies)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

---

## 🤔 So What's vink Anyway? (And Why Should You Care?)

Most vector databases force a trade-off: you either over-engineer for small datasets or hit a performance cliff as you scale. You’re left babysitting indices, manually tuning parameters, and praying your hardware can keep up.

**Vink** eliminates the guesswork. It automatically switches from **Exact Search** (for 100% precision) to **ANN** (for massive scale with IVF-PQ) based on dataset size and runtime latency. Whether you are running on a mobile device or a high-end server, Vink adapts its optimization strategy to your hardware and data distribution.

| Feature | Why it's awesome |
| :--- | :--- |
| ➕ **Incremental Inserts** | Add vectors anytime. Your index grows with your data, not against it. |
| 📟 **Hardware-Aware Auto-Switch** | It figures out when to ditch exact search and switch to ANN based on latency prediction. |
| ⚙️ **Self-Tuning Engine** | Background reconfiguration keeps clusters fresh as your data evolves. |
| 🎯 **Production-Ready Search** | Filtered searches, soft deletes, compact, dual-metric (Euclidean + cosine). |
| 💾 **Explicit Storage** | Disk or memory — you control where your data lives. |

Unlike enterprise solutions (Milvus, Pinecone) that require complex Docker or cloud setup, Vink runs entirely local — zero dependencies beyond pip install.

And that's just the start - there's plenty more to explore!

---

## 📦 Installation

First ensure that you have the necessary system dependencies installed.

- **Linux only**:
  Required for building [rii](https://github.com/matsui528/rii)

  ```bash
  # Debian/Ubuntu
  sudo apt-get install python3-dev

  # RedHat/Fedora/CentOS
  sudo dnf install python3-devel -y

  # CentOS 7 and older
  sudo yum install python3-devel
  ```

- **Android/Termux**:

   ```bash
   pkg install -y tur-repo
   pkg install python-scipy
   ```

### The Quick & Easy Way

The simplest way to get started is with pip:

```bash
pip install vink
```

### The From-Source Way

Prefer building from source? You can clone and install manually for full control:

```bash
git clone https://github.com/speedyk-005/vink.git
cd vink
pip install -e .
```

(But honestly, the pip way is usually way easier!)

---

## 🚀 Usage

### Initialization ([VinkDB API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB))

```python
from vink import VinkDB

# Create a database with 128-dimensional vectors
db = VinkDB("./data", dim=128)

# Or use in-memory mode (no persistence)
db = VinkDB(":memory:", dim=128)

# With custom settings
db = VinkDB(
    dir_path="./data",
    dim=384,
    metric="euclidean",       # or "cosine" (default: euclidean)
    force_exact=False,         # or True to disable ANN (default: False)
    ann_config=None,           # ANNConfig for PQ/OPQ (default: auto-generated)
    switch_latency_ms=300,    # ms threshold for ANN switch (default: 300)
    embedding_callback=None,  # fn to generate embeddings from content
    overwrite=False,          # overwrite existing index (default: False)
    verbose=False              # enable verbose output (default: False)
)
```

#### AnnConfig ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-models-AnnConfig))

Want custom ANN settings?

```python
from vink import AnnConfig

config = AnnConfig(
    num_subspaces=16,        # number of sub-vectors (default: 32)
    quantizer="pq",           # "pq" or "opq" (default: pq)
    codebook_size=128,        # centroids per subspace (default: 256)
)
db = VinkDB("./data", dim=384, ann_config=config)

# print all available options:
AnnConfig.help()
```

### Add ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB-add))

Records need:

- `content` (required): text to store
- `embedding` (required if no callback): list of floats or numpy array, shape `(d,)` or `(1, d)`
- `id` (optional): valid UUIDv7
- `metadata` (optional): dict of key-value pairs

Provide embeddings directly or use a callback to generate them on the fly.

#### With embedding callback

```python
db = VinkDB("./data", dim=384, embedding_callback=my_embedding_fn)

# Just provide content — embeddings generated automatically
db.add([
    {"content": "Hello world", "metadata": {"source": "doc1"}},
    {"content": "Another text"},
])
```

#### Without callback

Provide embeddings directly:

```python
db.add([
    {"content": "Hello world", "embedding": [0.1] * 384, "metadata": {"source": "doc1"}},
    {"content": "Another text", "embedding": [0.2] * 384}}
)]
```

### Search ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB-search))

Results include:

- `id`: vector ID
- `content`: text content
- `distance`: similarity score (lower is closer for euclidean)
- `metadata`: key-value pairs
- `embedding`: (only if `include_vectors=True`)

#### Without filters
```python
# Basic search
results = db.search(query_vec=[0.1] * 384, top_k=5)

# Include embeddings in results
results = db.search(query_vec=[0.1] * 384, include_vectors=True)
```

#### With filters

Filter syntax supports `==`, `!=`, `>`, `<`, `>=`, `<=` with strings, numbers, and booleans. More operators coming in future updates.

```
results = db.search(
    query_vec=[0.1] * 384,
    top_k=10,
    filters=["source == 'doc1'", "score >= 50", "new == True"]
)
```

### Delete

#### Soft deletion ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB-soft_delete))

Soft-delete vectors by ID without rebuilding the index — fast and efficient.

```python
# IDs come from search results or when adding
db.soft_delete(["0192a5b4-7f3c-7d6e-9a1b-2c3d4e5f6a7b", "0192a5b4-7f3c-7d6e-9a1b-2c3d4e5f6a7c"])
```

#### Compaction ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB-compact))

Actually remove soft-deleted records and reclaim storage:

```python
db.compact()
```

> [!WARNING]
> Can take 20-200+ seconds with `approximate strategy` depending on data size. Run during maintenance windows or off-peak hours. If not enough vectors remain to retrain the codec, rebuild is skipped.

### Stats ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-core-VinkDB-stats))

Get database statistics:

```python
stats = db.stats()
# {
#     "version": "...",
#     "dimension": 128,
#     "metric": "euclidean",
#     "strategy": "exact_search",
#     "last_saved_at": "...",
#     "last_deleted_at": "...",
#     "active_count": 1000,
#     "deleted_count": 5
# }
```

---

## 🚨 Exceptions ([API](https://github.com/speedyk-005/vink/blob/main/API_REFERENCES.md#vink-exceptions))

Something go wrong?

| Exception | When it hits |
| :--- | :--- |
| `InvalidInputError` | Bad data or invalid params |
| `VectorDimensionError` | Embedding dim mismatch |
| `InvalidIdError` | Malformed UUIDv7 |
| `FilterError` | Bad filter syntax |

---

## 🗺 Features & Roadmap

- [x] Incremental Inserts
- [x] Hardware-Aware Auto-Switch
- [x] Soft deletes + compact
- [x] Save/Load
- [ ] Filter DSL
  - [x] basic filters: Quick Comparison
  - [ ] Complex Filters: Content Matching, Null Checks, date/time literals, ...
- [ ] Recovery: recover soft-deleted vectors
- [ ] Collections: Multi-collection support for managing multiple indices
- [ ] CLI - command-line interface
- [ ] REST API: HTTP API for remote vector operations
- [ ] Integrations: LangChain, LlamaIndex, and other integrations

---

## 🔧 Core Dependencies

- [rii](https://github.com/matsui528/rii) — C++ ANN library with pybind11 bindings (IVF-PQ index storage)
- [nanopq](https://github.com/matsui528/nanopq) — Pure Python PQ encoding/decoding
- [scipy](https://scipy.org) — Scientific computing (distance calculations)
- [numpy](https://numpy.org) — Numerical computing
- [SQLite](https://sqlite.org) — Metadata storage (content, embeddings, metadata), filtering queries

---

## 🤝 Contributing

Bug fixes, features, docs — all welcome. Check out [CONTRIBUTING.md](https://github.com/speedyk-005/vink/blob/main/CONTRIBUTING.md) for the full details.

---

## 📜 License

Check out the [LICENSE](https://github.com/speedyk-005/vink/blob/main/LICENSE) file for all the details.

> MIT License. Use freely, modify boldly, and credit appropriately! (We're not that legendary... yet 😉)
