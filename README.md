# IDBI Gift Voucher Recommendation Engine

**CI/CD Implementation | Major Project | Manipal University Jaipur**

A production-grade Hybrid Recommendation System combining ALS collaborative filtering with semantic embeddings, wrapped in a fully automated 3-phase CI/CD pipeline.

---

## Architecture

```
PUSH → Phase 1: CI Gauntlet          Phase 2: Training Core       Phase 3: CD Post-Flight
       ├─ Ruff lint (F-type)    →    ├─ Load interactions         ├─ Validate 6 .pkl files
       ├─ 28 smoke tests              ├─ Build sparse matrix        └─ Container teardown
       └─ Env/Module/Model            ├─ Train ALS (64 factors)
          readiness checks            ├─ TF-IDF embeddings
                                      ├─ Hybrid score (65/35)
                                      └─ Deliver → ab_idbi_recommendations_uat
```

## Tool Mapping

| Report | This Project |
|---|---|
| TFS | GitHub |
| Jenkins | GitHub Actions |
| `pipelines.yaml` | `.github/workflows/pipeline.yml` |
| UV | UV (exact) |
| Ruff | Ruff v0.15.x (exact) |
| Docker + Compose | Docker + Compose (exact) |
| pytest | pytest (exact) |
| Oracle DB | SQLite (`ab_idbi_recommendations_uat`) |

## Quick Start

```bash
pip install uv
uv venv .venv && source .venv/bin/activate
uv add -r requirements.txt && uv sync
uv run python data/generate_dataset.py
uv run ruff check . --select F --ignore E402 --output-format grouped
uv run pytest tests/ -v
uv run python train_pipeline.py
```

**→ See [SETUP.md](SETUP.md) for full VS Code instructions.**

## Results

| Metric | Value |
|---|---|
| Tests | **50 / 50 passed** |
| Lint errors | **0** |
| Recommendation rows | **~15,798** |
| Artifacts validated | **6 .pkl files** |
| Runtime | **~14 seconds** |
