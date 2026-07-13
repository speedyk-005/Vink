Code quality pass: ruff compliance, internal refactors, CI setup

### Summary
Clean up the entire codebase to pass an expanded set of ruff lint rules, remove legacy abstractions that added indirection without value, and set up vinkra-specific CI workflows. No behavioural changes to the public API.

### What Changed
- Unified filter translation in `BaseStrategy` instead of duplicating it per-strategy.
- Replaced `VectorRecord` Pydantic models with plain `list[dict]` through the entire strategy chain.
- Stripped type annotations from docstrings to reduce drift; removed the two post-processing scripts.
- Expanded ruff to cover FBT, ARG, RET, PLR, E501, PTH123, A, F821, N806, SIM, INP001 — all violations fixed.
- Replaced FTS5 `unicode61` tokenizer with `trigram` for substring content matching.
- Switched from `open()` to `Path.open()` per `PTH123`.
- Added `.github/` workflows, Dependabot config, and issue template adapted for vinkra.

### Testing
- Full `pytest` suite passes.
- `ruff check src/ tests/` and `ruff format src/ tests/ --check` both clean.

### Risk / Rollout
Not needed — no migrations, no feature flags, no schema changes. Revert by rolling back the branch.
