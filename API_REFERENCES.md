# `vinkra`

## Table of Contents

- 🅼 [vinkra](#vinkra)
- 🅼 [vinkra\.core](#vinkra-core)
- 🅼 [vinkra\.exceptions](#vinkra-exceptions)
- 🅼 [vinkra\.filter\_parser](#vinkra-filter_parser)
- 🅼 [vinkra\.latency\_predictor](#vinkra-latency_predictor)
- 🅼 [vinkra\.models](#vinkra-models)
- 🅼 [vinkra\.sql\_wrapper](#vinkra-sql_wrapper)
- 🅼 [vinkra\.strategies](#vinkra-strategies)
- 🅼 [vinkra\.strategies\.approximate\_search](#vinkra-strategies-approximate_search)
- 🅼 [vinkra\.strategies\.base](#vinkra-strategies-base)
- 🅼 [vinkra\.strategies\.exact\_search](#vinkra-strategies-exact_search)
- 🅼 [vinkra\.utils](#vinkra-utils)
- 🅼 [vinkra\.utils\.id\_generation](#vinkra-utils-id_generation)
- 🅼 [vinkra\.utils\.input\_validation](#vinkra-utils-input_validation)
- 🅼 [vinkra\.utils\.logging](#vinkra-utils-logging)

<a name="vinkra"></a>
## 🅼 vinkra

Vinkra: Vector Incremental Nano Kit — Reconfigurated Automatically

A lightweight vector database that incrementally switches from exact to
approximate search as your data grows — without full index rebuilds\.

Note:
    ANN switching is one-way — once switched to approximate search,
    the system will never switch back to exact search\.

Key differentiators:
    - \*\*Incremental inserts\*\*: Add vectors anytime — no rebuild per insert\.
    - \*\*Automatic reconfig\*\*: No manual tuning — exact for small datasets, ANN for large\.
    - \*\*Thread-safe\*\*: Background ANN building doesn't block new operations\.
    - \*\*Soft deletes \+ compact\*\*: Efficient deletion with explicit storage reclamation\.

Features:
    - ~100x faster using RII and Product Quantization for large datasets\.
    - Pure Python: No external services or dependencies required\.
    - Supports Euclidean \(L2\) and cosine \(dot\) product similarity\.
    - SQLite-backed persistent storage\.

Technical Background:
    Vinkra uses Reconfigurable Inverted Index \(RII\) with Product Quantization \(PQ\)
    for approximate nearest neighbor search\. The switch from exact to approximate
    happens when the normalized power-law complexity reaches 1\.0:
    \(dim \* vectors / 1M\) ^ switch\_exp \>= 1\.0\. Default switch\_exp is 1\.0\.

References:
    \.\. \[Matsui18\] Matsui et al\., "Reconfigurable Inverted Index", ACM MM 2018\.

    \.\. \[Jegou11\] Jegou et al\., "Product Quantization for Nearest Neighbor Search",
       IEEE TPAMI 2011\.

    \.\. \[Matsui15\] Matsui et al\., "Optimized Product Quantization for Nearest
       Neighbor Search", CVPR 2015\.

See Also:
    - RII: https://github\.com/matsui528/rii
    - nanopq: https://github\.com/matsui528/nanopq
<a name="vinkra-core"></a>
## 🅼 vinkra\.core

- **Classes:**
  - 🅲 [VinkraDB](#vinkra-core-VinkraDB)

### Classes

<a name="vinkra-core-VinkraDB"></a>
### 🅲 vinkra\.core\.VinkraDB

```python
class VinkraDB:
```

Vector database with hybrid exact/approximate nearest neighbor search\.

VinkraDB automatically switches from exact brute-force search to approximate
nearest neighbor \(ANN\) search based on dataset size, using Reconfigurable Inverted
Index \(RII\) and Product Quantization \(PQ\) for efficient ANN\.

Note:
    ANN switching is one-way — once switched, the system never switches back to exact search\.

Features:

    - Hybrid search: exact for small datasets, ANN for large datasets\.
    - Automatic strategy reconfig based on runtime-calibrated latency prediction\.
    - Normalized embeddings for consistent distance metrics\.
    - Supports Euclidean \(L2\) and cosine \(dot\) product similarity\.
    - Soft deletes: efficient deletion without data reorganization\.

Getting AnnConfig:

    To customize ANN behavior, create an AnnConfig instance:

    \>\>\> from vinkra import AnnConfig
    \>\>\> config = AnnConfig\(
    \.\.\.     num\_subspaces=16,
    \.\.\.     codebook\_size=128,
    \.\.\.     switch\_latency\_ms=300,
    \.\.\.     quantizer="pq"
    \.\.\. \)

    \>\>\> db = VinkraDB\(dir\_path="\./data", dim=384, ann\_config=config\)

    For help with AnnConfig parameters, call AnnConfig\.help\(\)

**Functions:**

<a name="vinkra-core-VinkraDB-__init__"></a>
#### 🅵 vinkra\.core\.VinkraDB\.\_\_init\_\_

```python
def __init__(self, dir_path: str | Path, dim: Annotated[int, Field(ge=16)], metric: Literal['euclidean', 'cosine'] = 'euclidean', force_exact: bool = False, ann_config: AnnConfig | None = None, embedding_callback: Callable | None = None, overwrite: bool = False, verbose: bool = False) -> None:
```

Initialize a VinkraDB instance\.

Note:
    The only editable attributes after initialization are:
        - ann\_config
        - embedding\_callback
        - verbose
    Everything else is read-only properties\.

**Parameters:**

- **dir_path** (`str | Path`): Directory path to store vector data\. Contains the pickled index
and SQLite database for vector records\.
Use ":memory:" for volatile in-memory storage\.
- **dim** (`int`): Dimension of the vectors\. Must be higher than 16\.
- **metric** (`Literal["euclidean", "cosine"]`): Distance metric to use\.
Defaults to "euclidean"\.
- **force_exact** (`bool`): If True, only exact calculation is used\.
If False, switches between exact and ANN based on runtime calibration\.
Defaults to False\.
- **ann_config** (`AnnConfig`): Configuration for approximate nearest neighbor search\.
Used during switching and compacting\. Defaults to ANNConfig with standard settings\.
Only applicable when force\_exact is False\.
- **embedding_callback** (`Callable`): Callback function to generate embeddings
from content\. If provided, 'embedding' key is optional in
vector records as it will be generated via this callback\. Defaults to None\.
- **overwrite** (`bool`) (default: `False`): Overwrite existing index if exists\. Defaults to False\.
- **verbose** (`bool`) (default: `False`): Enable verbose output\. Defaults to False\.
<a name="vinkra-core-VinkraDB-dir_path"></a>
#### 🅵 vinkra\.core\.VinkraDB\.dir\_path

```python
def dir_path(self) -> Path:
```
<a name="vinkra-core-VinkraDB-dim"></a>
#### 🅵 vinkra\.core\.VinkraDB\.dim

```python
def dim(self) -> int:
```
<a name="vinkra-core-VinkraDB-metric"></a>
#### 🅵 vinkra\.core\.VinkraDB\.metric

```python
def metric(self) -> str:
```
<a name="vinkra-core-VinkraDB-force_exact"></a>
#### 🅵 vinkra\.core\.VinkraDB\.force\_exact

```python
def force_exact(self) -> bool:
```
<a name="vinkra-core-VinkraDB-in_memory"></a>
#### 🅵 vinkra\.core\.VinkraDB\.in\_memory

```python
def in_memory(self) -> bool:
```
<a name="vinkra-core-VinkraDB-strategy"></a>
#### 🅵 vinkra\.core\.VinkraDB\.strategy

```python
def strategy(self) -> str:
```

The internal indexing strategy currently active, formatted in snake\_case\.
<a name="vinkra-core-VinkraDB-count"></a>
#### 🅵 vinkra\.core\.VinkraDB\.count

```python
def count(self, status: Literal['active', 'deleted'] | None = None) -> int:
```

Count vectors in the database\.

**Parameters:**

- **status** (`Literal["active", "deleted"]`): Which vectors to count\.
Count all if not provided\.

**Returns:**

- `int`: Count of vectors\.
<a name="vinkra-core-VinkraDB-stats"></a>
#### 🅵 vinkra\.core\.VinkraDB\.stats

```python
def stats(self) -> dict:
```

Return database statistics and metadata\.

**Returns:**

- `dict`: Database metadata including version, dimension, metric, strategy,
last\_saved\_at, last\_deleted\_at, active\_count, deleted\_count,
and other stored metadata\.
<a name="vinkra-core-VinkraDB-add"></a>
#### 🅵 vinkra\.core\.VinkraDB\.add

```python
def add(self, vector_records: list[dict]) -> list[str]:
```

Add vectors to the index\.

**Parameters:**

- **vector_records** (`list[dict]`): List of dicts with 'content', 'metadata',
and 'embedding' keys\. 'id' is optional
If not provided, a UUIDv7 will be auto-generated\.

**Returns:**

- `list[str]`: List of assigned UUIDv7 IDs\.

**Raises:**

- **InvalidInputError**: If validation fails or if the first batch exceeds 10,000 vectors\.
<a name="vinkra-core-VinkraDB-soft_delete"></a>
#### 🅵 vinkra\.core\.VinkraDB\.soft\_delete

```python
def soft_delete(self, ids: list[str]) -> None:
```

Soft-delete vectors from the index by their IDs \(marks as deleted\)\.

**Parameters:**

- **ids** (`list[str]`): List of UUIDv7 IDs to soft-delete\.
<a name="vinkra-core-VinkraDB-compact"></a>
#### 🅵 vinkra\.core\.VinkraDB\.compact

```python
def compact(self) -> None:
```

Hard-delete soft-deleted records and rebuild the index\.

Note:
    For ApproximateSearch, the ANN index is rebuilt from scratch which can take
    20-200\+ seconds depending on data size\. This operation should be called
    during maintenance windows or off-peak hours\.
    If not enough vectors remain to retrain the codec, rebuild is skipped\.
<a name="vinkra-core-VinkraDB-save"></a>
#### 🅵 vinkra\.core\.VinkraDB\.save

```python
def save(self) -> None:
```

Save the index to disk\.
<a name="vinkra-core-VinkraDB-load"></a>
#### 🅵 vinkra\.core\.VinkraDB\.load

```python
def load(self, overwrite: bool = False) -> None:
```

Load the index from disk\.

**Parameters:**

- **overwrite** (`bool`): If True, replace in-memory state with loaded data\.
Defaults to False\.
<a name="vinkra-core-VinkraDB-search"></a>
#### 🅵 vinkra\.core\.VinkraDB\.search

```python
def search(self, query_vec: list[float] | np.ndarray, top_k: int = 10, include_vectors: bool = False, filters: list[str] | None = None) -> list[dict]:
```

Search for k nearest neighbors using the configured metric\.

**Parameters:**

- **query_vec** (`list[float] | np.ndarray`): The query vector as a list of floats,
1D numpy array \(d,\), or 2D numpy array \(1, d\)\.
- **top_k** (`int`) (default: `10`): Number of nearest neighbors to return\. Defaults to 10\.
- **include_vectors** (`bool`): If True, include 'embedding' key in results\.
Defaults to False\.
- **filters** (`list[str] | None`): Filter expressions to apply before scoring\.
E\.g\., \["category == 'science'", "price \>= 10"\]\.

**Returns:**

- `list[dict]`: List of dicts with 'id', 'content', 'metadata', 'distance',
and optionally 'embedding' \(if include\_vectors is True\)\.
<a name="vinkra-core-VinkraDB-__enter__"></a>
#### 🅵 vinkra\.core\.VinkraDB\.\_\_enter\_\_

```python
def __enter__(self) -> 'VinkraDB':
```
<a name="vinkra-core-VinkraDB-__exit__"></a>
#### 🅵 vinkra\.core\.VinkraDB\.\_\_exit\_\_

```python
def __exit__(self, exc_type, exc_val, exc_tb) -> None:
```
<a name="vinkra-exceptions"></a>
## 🅼 vinkra\.exceptions

- **Classes:**
  - 🅲 [VinkraraDBError](#vinkra-exceptions-VinkraraDBError)
  - 🅲 [InvalidInputError](#vinkra-exceptions-InvalidInputError)
  - 🅲 [VectorDimensionError](#vinkra-exceptions-VectorDimensionError)
  - 🅲 [InvalidIdError](#vinkra-exceptions-InvalidIdError)
  - 🅲 [IndexNotFittedError](#vinkra-exceptions-IndexNotFittedError)
  - 🅲 [FilterError](#vinkra-exceptions-FilterError)
  - 🅲 [DatabaseCorruptedError](#vinkra-exceptions-DatabaseCorruptedError)

### Classes

<a name="vinkra-exceptions-VinkraraDBError"></a>
### 🅲 vinkra\.exceptions\.VinkraraDBError

```python
class VinkraraDBError(Exception):
```

Base exception for all Vinkra errors\.
<a name="vinkra-exceptions-InvalidInputError"></a>
### 🅲 vinkra\.exceptions\.InvalidInputError

```python
class InvalidInputError(VinkraraDBError):
```

Raised when one or multiple invalid input\(s\) are encountered\.
<a name="vinkra-exceptions-VectorDimensionError"></a>
### 🅲 vinkra\.exceptions\.VectorDimensionError

```python
class VectorDimensionError(InvalidInputError):
```

Raised when vector dimensions don't match the index configuration\.
<a name="vinkra-exceptions-InvalidIdError"></a>
### 🅲 vinkra\.exceptions\.InvalidIdError

```python
class InvalidIdError(InvalidInputError):
```

Raised when an invalid UUIDv7 is provided\.
<a name="vinkra-exceptions-IndexNotFittedError"></a>
### 🅲 vinkra\.exceptions\.IndexNotFittedError

```python
class IndexNotFittedError(Exception):
```

Raised when an operation requiring learned quantization is called on an unitialized index\.
<a name="vinkra-exceptions-FilterError"></a>
### 🅲 vinkra\.exceptions\.FilterError

```python
class FilterError(InvalidInputError):
```

Raised when a filter expression fails to parse\.
<a name="vinkra-exceptions-DatabaseCorruptedError"></a>
### 🅲 vinkra\.exceptions\.DatabaseCorruptedError

```python
class DatabaseCorruptedError(VinkraraDBError):
```

Raised when database files are corrupted\.
<a name="vinkra-filter_parser"></a>
## 🅼 vinkra\.filter\_parser

- **Classes:**
  - 🅲 [FilterToSql](#vinkra-filter_parser-FilterToSql)

### Classes

<a name="vinkra-filter_parser-FilterToSql"></a>
### 🅲 vinkra\.filter\_parser\.FilterToSql

```python
class FilterToSql:
```

Guide to filter expressions

Fields:
    - Names: alphanumeric/Unicode, e\.g\., category, price, last\_login
    - Values:
         - strings: 'text' or "text"
         - numbers: 42 or 3\.14
         - boolean values: True or False

Operators: \`==\`, \`\!=\`, \`\>\`, \`\<\`, \`\>=\`, \`\<=\`

**Functions:**

<a name="vinkra-filter_parser-FilterToSql-translate"></a>
#### 🅵 vinkra\.filter\_parser\.FilterToSql\.translate

```python
def translate(self, filters: list[str]) -> tuple[str, list]:
```

Convert a list of filter strings into a safe SQLite query\.

**Parameters:**

- **filters** (`list[str]`): List of filter strings\.

**Returns:**

- `tuple[str, list]`: A tuple containing:
- query \(str\): The generated SQLite condition clause\.
- params \(list\): List of parameters to safely bind to the query\.
<a name="vinkra-latency_predictor"></a>
## 🅼 vinkra\.latency\_predictor

- **Classes:**
  - 🅲 [LatencyPredictor](#vinkra-latency_predictor-LatencyPredictor)

### Classes

<a name="vinkra-latency_predictor-LatencyPredictor"></a>
### 🅲 vinkra\.latency\_predictor\.LatencyPredictor

```python
class LatencyPredictor:
```

A lean, structural predictor using only bounded Power Law fitting\.

Uses a Power Law model \(y = a \* x^b\) to predict search latency based on
the number of vectors in the index\. Initial calibration measures raw
BLAS performance, then online tuning refines parameters with actual
runtime measurements\.

The model bounds exponents between 0\.7 and 1\.5 to keep predictions
physically meaningful despite hardware jitter\.

**Functions:**

<a name="vinkra-latency_predictor-LatencyPredictor-__init__"></a>
#### 🅵 vinkra\.latency\_predictor\.LatencyPredictor\.\_\_init\_\_

```python
def __init__(self, dim: int = 128, window_size: int = 32):
```

Initialize latency predictor with Power Law model\.

**Parameters:**

- **dim** (`int`): Vector dimensionality for calibration search\.
- **window_size** (`int`): Number of \(n\_vectors, latency\) pairs to keep for online tuning\.
<a name="vinkra-latency_predictor-LatencyPredictor-predict"></a>
#### 🅵 vinkra\.latency\_predictor\.LatencyPredictor\.predict

```python
def predict(self, n_vecs: int) -> float:
```

Predict latency for a given number of vectors in milliseconds\.
<a name="vinkra-latency_predictor-LatencyPredictor-tune"></a>
#### 🅵 vinkra\.latency\_predictor\.LatencyPredictor\.tune

```python
def tune(self, n_vecs: int, actual_lat: float) -> None:
```

Update model parameters with actual latency measurement\.

**Parameters:**

- **n_vecs** (`int`): Current number of vectors in the index\.
- **actual_lat** (`float`): Actual measured latency in milliseconds\.
<a name="vinkra-models"></a>
## 🅼 vinkra\.models

- **Classes:**
  - 🅲 [AnnConfig](#vinkra-models-AnnConfig)
  - 🅲 [VectorRecord](#vinkra-models-VectorRecord)
  - 🅲 [VectorRecords](#vinkra-models-VectorRecords)

### Classes

<a name="vinkra-models-AnnConfig"></a>
### 🅲 vinkra\.models\.AnnConfig

```python
class AnnConfig(BaseModel):
```

Configuration for Approximate Nearest Neighbor \(ANN\) search settings\.

**Functions:**

<a name="vinkra-models-AnnConfig-validate_vector_dim"></a>
#### 🅵 vinkra\.models\.AnnConfig\.validate\_vector\_dim

```python
def validate_vector_dim(self, dim: int) -> None:
```

Validate ANN config against a specific vector dimension\.
<a name="vinkra-models-AnnConfig-help"></a>
#### 🅵 vinkra\.models\.AnnConfig\.help

```python
def help(cls):
```

Print configuration arguments and descriptions\.
<a name="vinkra-models-VectorRecord"></a>
### 🅲 vinkra\.models\.VectorRecord

```python
class VectorRecord(BaseModel):
```

Model for a vector record entry\.

**Functions:**

<a name="vinkra-models-VectorRecord-validate_id"></a>
#### 🅵 vinkra\.models\.VectorRecord\.validate\_id

```python
def validate_id(cls, v):
```

Validate an ID or generate a new UUIDv7\. Always returns 16 bytes\.
<a name="vinkra-models-VectorRecords"></a>
### 🅲 vinkra\.models\.VectorRecords

```python
class VectorRecords(BaseModel):
```

Container for multiple vector records with dimension enforcement\.

**Functions:**

<a name="vinkra-models-VectorRecords-validate_dimensions"></a>
#### 🅵 vinkra\.models\.VectorRecords\.validate\_dimensions

```python
def validate_dimensions(self) -> 'VectorRecords':
```

Ensure all embeddings match the specified dimension and normalize if needed\.
<a name="vinkra-sql_wrapper"></a>
## 🅼 vinkra\.sql\_wrapper

- **Classes:**
  - 🅲 [SQLiteWrapper](#vinkra-sql_wrapper-SQLiteWrapper)

### Classes

<a name="vinkra-sql_wrapper-SQLiteWrapper"></a>
### 🅲 vinkra\.sql\_wrapper\.SQLiteWrapper

```python
class SQLiteWrapper:
```

Central SQLite connection and schema management for VinkraDB\.

**Functions:**

<a name="vinkra-sql_wrapper-SQLiteWrapper-__init__"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.\_\_init\_\_

```python
def __init__(self, path: str, index_config: dict):
```

Initialize SQLite wrapper\.

**Parameters:**

- **path**: Path to SQLite database file\.
- **index_config**: Optional dict with index metadata \(dimension, metric, strategy\)\.
Used to initialize db\_meta table on first creation\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-conn"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.conn

```python
def conn(self):
```

Expose the raw connection
<a name="vinkra-sql_wrapper-SQLiteWrapper-close"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.close

```python
def close(self) -> None:
```

Close the database connection\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-commit"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.commit

```python
def commit(self) -> None:
```

Explicitly commit the current transaction\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-insert"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.insert

```python
def insert(self, vec_records: VectorRecords, is_buffer: bool = False) -> None:
```

Insert vec\_records into SQLite\.

**Parameters:**

- **vec_records**: VectorRecords object\.
- **is_buffer**: If True, marks all vec\_records as buffered vec\_records\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-soft_delete"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.soft\_delete

```python
def soft_delete(self, ids: list[bytes]) -> None:
```

Soft-delete vec\_records from SQLite \(marks as deleted\)\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-fetch"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.fetch

```python
def fetch(self, where: str | None = None, params: list | tuple | None = None, include_vectors: bool = False):
```

Fetch vec\_records from SQLite\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-count"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.count

```python
def count(self, status: Literal['active', 'deleted'] | None = None) -> int:
```

Count vectors in the database\.

**Parameters:**

- **status** (`Literal["active", "deleted"]`): Which vectors to count\.
Count all if not provided\.

**Returns:**

- `int`: Count of vectors\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-clear_buffer"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.clear\_buffer

```python
def clear_buffer(self) -> None:
```

Set all buffer flags to False\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-compact"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.compact

```python
def compact(self) -> None:
```

Hard-delete all soft-deleted records from SQLite\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-iter_embeddings"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.iter\_embeddings

```python
def iter_embeddings(self, batch_size: int = 50000) -> Generator[list, None, None]:
```

Iterate over embeddings in batches\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-__getitem__"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.\_\_getitem\_\_

```python
def __getitem__(self, key: str) -> str | None:
```

Get a metadata value from db\_meta table\.
<a name="vinkra-sql_wrapper-SQLiteWrapper-__setitem__"></a>
#### 🅵 vinkra\.sql\_wrapper\.SQLiteWrapper\.\_\_setitem\_\_

```python
def __setitem__(self, key: str, value: str) -> None:
```

Set a metadata value in db\_meta table\.
<a name="vinkra-strategies"></a>
## 🅼 vinkra\.strategies
<a name="vinkra-strategies-approximate_search"></a>
## 🅼 vinkra\.strategies\.approximate\_search

- **Classes:**
  - 🅲 [ApproximateSearch](#vinkra-strategies-approximate_search-ApproximateSearch)

### Classes

<a name="vinkra-strategies-approximate_search-ApproximateSearch"></a>
### 🅲 vinkra\.strategies\.approximate\_search\.ApproximateSearch

```python
class ApproximateSearch(BaseStrategy):
```

Approximate search strategy using Product Quantization \(PQ\) or Optimized PQ\.

This strategy compresses high-dimensional vectors into compact codes using
subspace quantization\. It provides significantly faster search performance
and reduced memory usage compared to exact methods, making it suitable
for large-scale vector datasets where sub-millisecond latency is required\.

Note:
    This strategy relies on codebooks generated during the 'fit'
    initialization step\. Precision is subject to the number of subspaces
    and the quantization method employed\.

**Functions:**

<a name="vinkra-strategies-approximate_search-ApproximateSearch-__init__"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.\_\_init\_\_

```python
def __init__(self, db: SQLiteWrapper, dir_path: Path | None, dim: int, in_memory: bool, metric: Literal['euclidean', 'cosine'], verbose: bool, ann_config: AnnConfig) -> None:
```

Initialize the ApproximateSearch\.

**Parameters:**

- **db** (`SQLiteWrapper`): SQLite wrapper for database operations\.
- **dir_path** (`Path | None`) (default: `None`): Path to store vector data\. Defaults to None\.
- **dim** (`int`): Dimension of the vectors\.
- **in_memory** (`bool`): Whether using in-memory storage\.
- **metric** (`Literal["euclidean", "dot"]`): Distance metric to use\.
- **verbose** (`bool`): Enable verbose output\.
- **ann_config** (`AnnConfig`): ANN configuration\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-fit"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.fit

```python
def fit(self, vectors: np.ndarray, active_ids_arr: np.ndarray) -> None:
```

Initialize the Approximate Search index by training the Quantizer\.

It processes all currently indexed vectors to generate the subspace codebooks
required for approximate search\.
The quantizer is trained with randomly sampled vectors\.

**Parameters:**

- **vectors** (`np.ndarray`): A 2D array of shape \(N, D\) representing the N vectors
of dimensionality D to be indexed\.
- **active_ids_arr** (`np.ndarray`): Array of active IDs corresponding to the vectors\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-add"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.add

```python
def add(self, vector_records, is_buffer: bool = False) -> list[str]:
```

Add vectors to the index\.

**Parameters:**

- **vector_records** (`VectorRecords`): Container with list of vector records\.

**Returns:**

- `list[str]`: List of assigned UUIDv7 IDs\.

**Raises:**

- **IndexNotFittedError**: If called on an index that has not been fitted yet\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-soft_delete"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.soft\_delete

```python
def soft_delete(self, ids: list[bytes]) -> None:
```

Soft-delete vectors from the index by their IDs \(marks as deleted\)\.

**Parameters:**

- **ids** (`list[bytes]`): List of UUIDv7 IDs to soft-delete\.

**Raises:**

- **IndexNotFittedError**: If called on an index that has not been fitted yet\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-search"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.search

```python
def search(self, query_vec: np.ndarray, top_k: int = 10, include_vectors: bool = False, filters: list[str] | None = None) -> list[dict]:
```

Search for k nearest neighbors using the configured metric\.

**Parameters:**

- **query_vec** (`np.ndarray`): The query vector as a 2D numpy array with shape \(1, d\)\.
- **top_k** (`int`) (default: `10`): Number of nearest neighbors to return\. Defaults to 10\.
- **include_vectors** (`bool`): If True, include 'embedding' key in results\.
Defaults to False\.
- **filters** (`list[str] | None`): Filter expressions to apply before scoring\.

**Returns:**

- `list[dict]`: List of dicts with 'id', 'content', 'metadata', 'distance',
and optionally 'embedding' \(if include\_vectors is True\)\.

**Raises:**

- **IndexNotFittedError**: If called on an index that has not been fitted yet\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-compact"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.compact

```python
def compact(self) -> None:
```

Hard-delete soft-deleted records and rebuild the index\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-save"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.save

```python
def save(self) -> None:
```

Save the index to disk using double-write strategy for tight syncing\.
<a name="vinkra-strategies-approximate_search-ApproximateSearch-load"></a>
#### 🅵 vinkra\.strategies\.approximate\_search\.ApproximateSearch\.load

```python
def load(self, overwrite: bool) -> None:
```

Load the index from disk\.

**Parameters:**

- **overwrite** (`bool`): If True, replace in-memory state with loaded data\.
<a name="vinkra-strategies-base"></a>
## 🅼 vinkra\.strategies\.base

- **Classes:**
  - 🅲 [BaseStrategy](#vinkra-strategies-base-BaseStrategy)

### Classes

<a name="vinkra-strategies-base-BaseStrategy"></a>
### 🅲 vinkra\.strategies\.base\.BaseStrategy

```python
class BaseStrategy(ABC):
```

Base class for search strategies\.

Provides abstract interface for exact and approximate search implementations\.

**Functions:**

<a name="vinkra-strategies-base-BaseStrategy-__init__"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.\_\_init\_\_

```python
def __init__(self, db: SQLiteWrapper, dir_path: Path | None, dim: int, is_exact: bool, in_memory: bool, metric: Literal['euclidean', 'cosine'], verbose: bool, **kwargs) -> None:
```

Initialize the strategy\.

**Parameters:**

- **db** (`SQLiteWrapper`): SQLite wrapper for database operations\.
- **dir_path** (`Path | None`) (default: `None`): Path to store vector data for querying\. Defaults to None\.
- **dim** (`int`): Dimension of the vectors\.
- **is_exact** (`bool`): Whether this strategy uses exact search\.
- **in_memory** (`bool`): Whether using in-memory storage\.
- **metric** (`Literal["euclidean", "cosine"]`): Distance metric to use\.
- **verbose** (`bool`): Enable verbose output\.
- ****kwargs**: Additional keyword arguments for subclasses\.
<a name="vinkra-strategies-base-BaseStrategy-add"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.add

```python
def add(self, vector_records: VectorRecords, is_buffer: bool = False) -> list[str]:
```

Add vectors to the index\.

**Parameters:**

- **vector_records** (`VectorRecords`): Container with list of vector records\.
- **is_buffer** (`bool`): If True, records are already in SQLite \(buffer replay\)\.
Subclasses should skip re-inserting to avoid duplicate key errors\.

**Returns:**

- `list[str]`: List of assigned UUIDv7 IDs\.
<a name="vinkra-strategies-base-BaseStrategy-soft_delete"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.soft\_delete

```python
def soft_delete(self, ids: list[str]) -> None:
```

Soft-delete vectors from the index by their IDs \(marks as deleted\)\.

**Parameters:**

- **ids** (`list[bytes]`): List of UUIDv7 IDs to soft-delete\.
<a name="vinkra-strategies-base-BaseStrategy-compact"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.compact

```python
def compact(self) -> None:
```

Hard-delete soft-deleted records and rebuild the index\.
<a name="vinkra-strategies-base-BaseStrategy-save"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.save

```python
def save(self) -> None:
```

Save the index to disk\.
<a name="vinkra-strategies-base-BaseStrategy-load"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.load

```python
def load(self, overwrite: bool) -> None:
```

Load the index from disk\.

**Parameters:**

- **overwrite** (`bool`): If True, replace in-memory state with loaded data\.
<a name="vinkra-strategies-base-BaseStrategy-search"></a>
#### 🅵 vinkra\.strategies\.base\.BaseStrategy\.search

```python
def search(self, query_vec: np.ndarray, top_k: int = 10, include_vectors: bool = False, filters: list[str] | None = None) -> list[dict]:
```

Search for k nearest neighbors using the configured metric\.

**Parameters:**

- **query_vec** (`np.ndarray`): The query vector as a 2D numpy array with shape \(1, d\)\.
- **top_k** (`int`) (default: `10`): Number of nearest neighbors to return\. Defaults to 10\.
- **include_vectors** (`bool`): If True, include 'embedding' key in results\.
Defaults to False\.
- **filters** (`list[str] | None`): Filter expressions to apply before scoring\.

**Returns:**

- `list[dict]`: List of dicts with 'id', 'content', 'metadata', 'distance',
and optionally 'embedding' \(if include\_vectors is True\)\.
<a name="vinkra-strategies-exact_search"></a>
## 🅼 vinkra\.strategies\.exact\_search

- **Classes:**
  - 🅲 [ExactSearch](#vinkra-strategies-exact_search-ExactSearch)

### Classes

<a name="vinkra-strategies-exact_search-ExactSearch"></a>
### 🅲 vinkra\.strategies\.exact\_search\.ExactSearch

```python
class ExactSearch(BaseStrategy):
```

Exact search strategy using brute-force distance computation\.

This strategy computes exact nearest neighbors by calculating distances
to all stored vectors\. It is suitable for smaller datasets or when
maximum recall is required\.

**Functions:**

<a name="vinkra-strategies-exact_search-ExactSearch-__init__"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.\_\_init\_\_

```python
def __init__(self, db: SQLiteWrapper, dir_path: Path | None, dim: int, in_memory: bool, metric: Literal['euclidean', 'cosine'], verbose: bool) -> None:
```

Initialize the ExactSearch\.

**Parameters:**

- **db** (`SQLiteWrapper`): SQLite wrapper for database operations\.
- **dir_path** (`Path | None`) (default: `None`): Path to store vector data\. Defaults to None\.
- **dim** (`int`): Dimension of the vectors\.
- **in_memory** (`bool`): Whether using in-memory storage\.
- **metric** (`Literal["euclidean", "cosine"]`): Distance metric to use\.
- **verbose** (`bool`): Enable verbose output\.
<a name="vinkra-strategies-exact_search-ExactSearch-add"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.add

```python
def add(self, vector_records, is_buffer: bool = False) -> list[str]:
```

Add vectors to the index\.

**Parameters:**

- **vector_records** (`VectorRecords`): Container with list of vector records\.
- **is_buffer** (`bool`) (default: `False`): If True, records are already in SQLite\. Defaults to False\.

**Returns:**

- `list[str]`: List of assigned UUIDv7 IDs\.
<a name="vinkra-strategies-exact_search-ExactSearch-soft_delete"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.soft\_delete

```python
def soft_delete(self, ids: list[bytes]) -> None:
```

Soft-delete vectors from the index by their IDs \(marks as deleted\)\.

**Parameters:**

- **ids** (`list[bytes]`): List of UUIDv7 IDs to soft-delete\.
<a name="vinkra-strategies-exact_search-ExactSearch-search"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.search

```python
def search(self, query_vec: np.ndarray, top_k: int = 10, include_vectors: bool = False, filters: list[str] | None = None) -> list[dict]:
```

Search for k nearest neighbors using the configured metric\.

**Parameters:**

- **query_vec** (`np.ndarray`): The query vector as a 2D numpy array with shape \(1, d\)\.
- **top_k** (`int`) (default: `10`): Number of nearest neighbors to return\. Defaults to 10\.
- **include_vectors** (`bool`): If True, include 'embedding' key in results\.
Defaults to False\.
- **filters** (`list[str] | None`): Filter expressions to apply before scoring\.

**Returns:**

- `list[dict]`: List of dicts with 'id', 'content', 'metadata', 'distance',
and optionally 'embedding' \(if include\_vectors is True\)\.
<a name="vinkra-strategies-exact_search-ExactSearch-compact"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.compact

```python
def compact(self) -> None:
```

Hard-delete soft-deleted records and rebuild the index\.
<a name="vinkra-strategies-exact_search-ExactSearch-save"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.save

```python
def save(self) -> None:
```

Save the index to disk by committing the database\.
<a name="vinkra-strategies-exact_search-ExactSearch-load"></a>
#### 🅵 vinkra\.strategies\.exact\_search\.ExactSearch\.load

```python
def load(self, overwrite: bool) -> None:
```

Load the index from SQLite\.

**Parameters:**

- **overwrite** (`bool`): If True, replace in-memory state with loaded data\.
<a name="vinkra-utils"></a>
## 🅼 vinkra\.utils
<a name="vinkra-utils-id_generation"></a>
## 🅼 vinkra\.utils\.id\_generation

- **Functions:**
  - 🅵 [generate\_id](#vinkra-utils-id_generation-generate_id)
  - 🅵 [generate\_id\_bytes](#vinkra-utils-id_generation-generate_id_bytes)

### Functions

<a name="vinkra-utils-id_generation-generate_id"></a>
### 🅵 vinkra\.utils\.id\_generation\.generate\_id

```python
def generate_id() -> str:
```

Generate a UUIDv7 as a string\.

**Returns:**

- `str`: RFC 9562 UUIDv7 in standard string format\.
<a name="vinkra-utils-id_generation-generate_id_bytes"></a>
### 🅵 vinkra\.utils\.id\_generation\.generate\_id\_bytes

```python
def generate_id_bytes() -> bytes:
```

Generate a UUIDv7 as 16 bytes\.

**Returns:**

- `bytes`: UUIDv7 in 16-byte binary form\.
<a name="vinkra-utils-input_validation"></a>
## 🅼 vinkra\.utils\.input\_validation

- **Functions:**
  - 🅵 [pretty\_errors](#vinkra-utils-input_validation-pretty_errors)
  - 🅵 [validate\_arguments](#vinkra-utils-input_validation-validate_arguments)
  - 🅵 [validate\_embedding](#vinkra-utils-input_validation-validate_embedding)
  - 🅵 [validate\_id](#vinkra-utils-input_validation-validate_id)

### Functions

<a name="vinkra-utils-input_validation-pretty_errors"></a>
### 🅵 vinkra\.utils\.input\_validation\.pretty\_errors

```python
def pretty_errors(error: ValidationError) -> str:
```

Formats Pydantic validation errors into a clean, human-readable summary\.

**Parameters:**

- **error** (`ValidationError`): The Pydantic error object to format\.

**Returns:**

- `str`: A scannable string containing error counts, locations, and input types\.
<a name="vinkra-utils-input_validation-validate_arguments"></a>
### 🅵 vinkra\.utils\.input\_validation\.validate\_arguments

```python
def validate_arguments(fn):
```

Decorator that enforces type safety on function inputs and outputs\.

**Parameters:**

- **fn** (`Callable`): The function to be validated\.

**Returns:**

- `Callable`: A wrapped function that re-raises Pydantic errors as InvalidInputError\.
<a name="vinkra-utils-input_validation-validate_embedding"></a>
### 🅵 vinkra\.utils\.input\_validation\.validate\_embedding

```python
def validate_embedding(vecs: list[float] | np.ndarray, dim: int, metric: str) -> np.ndarray:
```

Validate and optionally normalize input vectors\.

Ensures the input is a valid numeric array and enforces a 2D \(1, d\) shape\.
Normalization is only applied for cosine metric; euclidian skips normalization\.

**Parameters:**

- **vecs** (`list[float] | np.ndarray`): Input embedding\. Accepts 1D arrays of
shape \(d,\) or 2D row vectors of shape \(1, d\)\.
- **dim** (`int`): The required dimension for the embedding\.
- **metric** (`str`): Distance metric\.

**Returns:**

- `np.ndarray`: A float32 row vector of shape \(1, d\)\. Normalized for cosine,
raw for euclidean\.

**Raises:**

- **InvalidInputError**: If the input contains non-numeric values or \(for cosine\)
has a zero-magnitude \(null\) norm\.
- **VectorDimensionError**: If the input dimensionality or shape is incompatible
with a single-vector representation\.
<a name="vinkra-utils-input_validation-validate_id"></a>
### 🅵 vinkra\.utils\.input\_validation\.validate\_id

```python
def validate_id(id: str | bytes) -> bytes:
```

Validate an ID or generate a new UUIDv7\. Always returns 16 bytes\.

**Parameters:**

- **id** (`str | bytes`): UUIDv7 as string or bytes\.

**Returns:**

- `bytes`: 16-byte binary UUIDv7\.
<a name="vinkra-utils-logging"></a>
## 🅼 vinkra\.utils\.logging

- **Functions:**
  - 🅵 [log\_info](#vinkra-utils-logging-log_info)

### Functions

<a name="vinkra-utils-logging-log_info"></a>
### 🅵 vinkra\.utils\.logging\.log\_info

```python
def log_info(verbose: bool, *args, **kwargs) -> None:
```

Log an info message if verbose is enabled\.

This is a convenience function that only logs when verbose mode is enabled,
avoiding unnecessary log output in production\.

**Parameters:**

- **verbose**: If True, logs the message; if False, does nothing\.
- ***args**: Positional arguments passed to logger\.info\(\)\.
- ****kwargs**: Keyword arguments passed to logger\.info\(\)\.
