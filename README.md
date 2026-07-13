# 🐦 Vinkra

<p align="center">
  <img src="https://github.com/speedyk-005/vinkra/blob/main/vinkra_logo.svg?raw=true" alt="Vinkra Logo" width="300"/>
</p>

<p align="center">
  <b>V</b>ector <b>In</b>cremental <b>N</b>ano <b>K</b>it — <b>R</b>econfigurated <b>A</b>utomatically
</p>
<p align="center">
  “Vector DB that self-organizes. Auto-switch, Auto-tune, Auto-scale.”
</p>

[![Python Version](https://img.shields.io/badge/Python-3.9%20--%203.14-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/vinkra)](https://pypi.org/project/vinkra)
[![CodeFactor](https://www.codefactor.io/repository/github/speedyk-005/vinkra/badge)](https://www.codefactor.io/repository/github/speedyk-005/vinkra)
[![Coverage Status](https://coveralls.io/repos/github/speedyk-005/vinkra/badge.svg?branch=main)](https://coveralls.io/github/speedyk-005/vinkra?branch=main)
[![Stability](https://img.shields.io/badge/stability-pre--alpha-yellow)](https://github.com/speedyk-005/vinkra)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/speedyk-005/vinkra/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> This project is currently in pre-alpha.

---

## 🤔 So What's vinkra Anyway? (And Why Should You Care?)

Most vector databases force a trade-off: you either over-engineer for small datasets or hit a performance cliff as you scale. You’re left babysitting indices, manually tuning parameters, and praying your hardware can keep up.

**Vinkra** eliminates the guesswork. It automatically switches from **Exact Search** (for 100% precision) to **ANN** (for massive scale with IVF-PQ) based on dataset size and runtime latency. Whether you are running on a mobile device or a high-end server, Vinkra adapts its optimization strategy to your hardware and data distribution.

| Feature | Why it's awesome |
| :--- | :--- |
| ➕ **Incremental Inserts** | Add vectors anytime. Your index grows with your data, not against it. |
| 📟 **Hardware-Aware Auto-Switch** | It figures out when to ditch exact search and switch to ANN based on latency prediction. |
| ⚙️ **Self-Tuning Engine** | Background reconfiguration keeps clusters fresh as your data evolves. |
| 🎯 **Production-Ready Search** | Filtered searches, soft deletes, compact, dual-metric (Euclidean + cosine). |
| 💾 **Explicit Storage** | Disk or memory — you control where your data lives. |

Unlike enterprise solutions (Milvus, Pinecone) that require complex Docker or cloud setup, Vinkra runs entirely local with zero dependencies beyond pip install.

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
pip install vinkra
```

> [!TIP]
> **Termux (Android)**
>
> No Rust toolchain? Install pydantic-core pre-built wheels first, then retry:
>
> ```bash
> pip install typing-extensions
> pip install pydantic-core --index-url https://termux-user-repository.github.io/pypi/
> pip install "pydantic>=2.12.4,<2.13"
> ```

### The From-Source Way

Prefer building from source? You can clone and install manually for full control:

```bash
git clone https://github.com/speedyk-005/vinkra.git
cd vinkra
pip install -e .
```

---

## ✅ Proof It Works

Run the demo to see auto-switch in action:

```bash
# Install and run anywhere
curl -O https://raw.githubusercontent.com/speedyk-005/vinkra/main/demo_poc.py
python demo_poc.py
```

The demo uses:
- `switch_latency_ms=120` inside `AnnConfig` to trigger the switch sooner
- `dim=128`
- Batches of 10,000 vectors

The switch happens when latency exceeds `switch_latency_ms`. A Power Law model (`y = a * x^b`) continuously tunes itself from actual search latencies to predict future performance. New vectors are buffered during the switch with zero downtime.

Example output:

```
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Vectors ┃      Strategy      ┃ Avg Query (ms) ┃ Insert Time (s) ┃     Status     ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 10,000  │    exact_search    │     32.486     │      0.806      │  Exact Search  │
│ 20,000  │    exact_search    │     79.690     │      0.729      │  Exact Search  │
│ 30,000  │    exact_search    │    107.419     │      0.720      │  Exact Search  │
│ 40,000  │    exact_search    │    188.063     │      0.771      │  ⚙ Building ANN │
│ 50,000  │ approximate_search │     0.000      │     10.051      │  ✓ ANN Active  │
│ 60,000  │ approximate_search │    155.239     │      1.323      │  ✓ ANN Active  │
└─────────┴────────────────────┴────────────────┴─────────────────┴────────────────┘

✓ ANN switch successfully triggered!
```

> [!NOTE]
> Results vary by hardware and system load. Faster machines switch later, and running other programs will affect timing.

---

## 🚀 Usage

### Initialization ([VinkraDB API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB))

```python
from vinkra import VinkraDB

# Create a database with 128-dimensional vectors
db = VinkraDB(dim=128, dir_path="./data")

# Or use volatile in-memory mode (omit dir_path)
# db = VinkraDB(dim=128)

# Full configuration capabilities
db = VinkraDB(
    dim=384,
    dir_path="./data",
    metric="euclidean",         # or "cosine" (default: euclidean)
    force_exact=False,          # Set to True to completely lock it out of ANN mode
    ann_config=None,            # Provide custom AnnConfig instance (default: auto-generated)
    embedding_callback=None,    # Optional function to generate vectors from raw content text
    overwrite=False,            # Blow away existing directory index if True
    verbose=False               # Enable internal runtime diagnostic logs
)
```

#### Context Manager (Recommended)
VinkraDB fully implements Python's context manager interface. Using a `with` block guarantees your in-flight buffers flush cleanly to disk and the underlying SQLite engine closes its connections safely without dangling file locks, even if your code raises unhandled exceptions.

```python
from vinkra import VinkraDB

with VinkraDB(dim=384, dir_path="./data") as db:
    db.add([{"content": "Seamless storage context"}])

# The database saves and shuts down gracefully right here
```

#### AnnConfig ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-models-AnnConfig))

Want custom ANN configurations and custom latency thresholds? Pass them through `AnnConfig`:

```python
from vinkra import AnnConfig, VinkraDB

config = AnnConfig(
    num_subspaces=16,          # number of sub-vectors (default: 32)
    quantizer="pq",            # "pq" or "opq" (default: pq)
    codebook_size=128,         # centroids per subspace (default: 256)
    switch_latency_ms=150      # Runtime latency milestone to trigger ANN switch (default: 300)
    reconfig_threshold=100_000 # Inserts before reconfiguring the index on search performance (default: 100k)
)
db = VinkraDB(dim=384, dir_path="./data", ann_config=config)

# Print all available technical constraints and options:
AnnConfig.help()
```

### Add ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB-add))

Records accept the following schema structures:
- `content` (required): text payload to track
- `embedding` (required if no callback configured): list of floats or 1D/2D numpy array
- `id` (optional): string representation of a valid UUIDv7
- `metadata` (optional): dictionary containing scalar filtering targets

#### Without callback

```python
db.add([
    {"content": "Hello world", "embedding": [0.1] * 384, "metadata": {"source": "doc1"}},
    {"content": "Another text", "embedding": [0.2] * 384}
])
```

#### With embedding callback

```python
db = VinkraDB(dim=384, dir_path="./data", embedding_callback=my_embedding_fn)

# Omit 'embedding' keys; generated entirely under the hood
db.add([
    {"content": "Hello world", "metadata": {"source": "doc1"}},
    {"content": "Another text"},
])
```

### Search ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB-search))

Results include:

- `id`: vector ID
- `content`: text content
- `distance`: similarity score (lower is closer for euclidean)
- `metadata`: key-value pairs
- `embedding`: (only if `include_vectors=True`)

#### Without filters
```python
results = db.search(query_vec=[0.1] * 384, top_k=5)

# Include source embeddings in output mapping
results = db.search(query_vec=[0.1] * 384, include_vectors=True)
```

#### With filters

Filters are checked before similarity metrics hit vectors. Operators support `==`, `!=`, `>`, `<`, `>=`, `<=` matching against string, numeric, and boolean literals.

```python
results = db.search(
    query_vec=[0.1] * 384,
    top_k=10,
    filters=["source == 'doc1'", "score >= 50", "new == True"]
)
```

### Persistence & Index Maintenance

#### Save
If you are manually managing your resource lifetimes instead of applying the context manager pipeline, write your current index to disk space directly:

```python
db.save()
```

#### Close
Flushes the state tracker to disk parameters, flushes the runtime matrix configurations, and securely shuts down active transactional access structures in SQLite:

```python
db.close()
```

#### Soft deletion ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB-soft_delete))

Hides elements immediately from subsequent query pipelines without triggering expensive matrix reorganization:

```python
db.soft_delete(["0192a5b4-7f3c-7d6e-9a1b-2c3d4e5f6a7b"])
```

#### Compaction ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB-compact))

Purges soft-deleted tracking footprints permanently and forces an index structural rebuild:

```python
db.compact()
```

> [!WARNING]
> Running compaction on active `approximate_search` nodes can freeze workflows for 20-200+ seconds to calculate codebook states. Offload this into scheduled system maintenance hours.

### Stats ([API](https://github.com/speedyk-005/vinkra/blob/main/API_REFERENCES.md#vinkra-core-VinkraDB-stats))

```python
stats = db.stats()
# {
#     "version": "...",
#     "dim": 384,
#     "metric": "euclidean",
#     "strategy": "exact_search",
#     "last_saved_at": "...",
#     "last_deleted_at": "...",
#     "active_count": 1000,
#     "deleted_count": 5
# }
```

---

## 🗺 Features & Roadmap

- [x] Incremental Inserts
- [x] Hardware-Aware Auto-Switch
- [x] Soft deletes + compact
- [x] Save/Load + Context Manager
- [ ] Filter DSL
  - [x] basic filters: Quick Comparison
  - [ ] Complex Filters: Content Matching, Null Checks, date/time literals, ...
- [ ] Recovery: recover soft-deleted vectors
- [ ] REST API: HTTP API for remote vector operations
- [ ] Integrations: LangChain, LlamaIndex, and other integrations

---

## 🔧 Core Dependencies

[rii](https://github.com/matsui528/rii) •
[nanopq](https://github.com/matsui528/nanopq) • [scipy](https://scipy.org) • [numpy](https://numpy.org) • [SQLite](https://sqlite.org)

---

## 🤝 Contributing

Bug fixes, features, docs — all welcome. Check out [CONTRIBUTING.md](https://github.com/speedyk-005/vinkra/blob/main/CONTRIBUTING.md) for the full details.

---

## 📜 License

Check out the [LICENSE](https://github.com/speedyk-005/vinkra/blob/main/LICENSE) file for all the details.

> MIT License. Use freely, modify boldly, and credit appropriately!
