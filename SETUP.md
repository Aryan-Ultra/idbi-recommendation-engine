# SETUP GUIDE — IDBI Gift Voucher Recommendation Engine

> Step-by-step instructions for getting this project running in VS Code from scratch.

---

## Prerequisites

Install these before you start:

| Tool | Download | Why |
|---|---|---|
| VS Code | https://code.visualstudio.com | IDE |
| Python 3.9+ | https://python.org/downloads | Runtime |
| Docker Desktop | https://docker.com/products/docker-desktop | Containerisation |
| Git | https://git-scm.com | Version control |

---

## Step 1 — Open in VS Code

```
File → Open Folder → select the idbi-recommendation-engine folder
```

VS Code will detect `.vscode/` automatically and apply all settings.

---

## Step 2 — Install Recommended Extensions

VS Code will prompt: **"Do you want to install the recommended extensions?"**
Click **Install All**. These are defined in `.vscode/extensions.json`:

- **ms-python.python** — Python IntelliSense
- **charliermarsh.ruff** — Ruff linter (as per report)
- **ms-azuretools.vscode-docker** — Docker support
- **github.vscode-github-actions** — Pipeline YAML support

---

## Step 3 — Set Up the UV Environment

Open the integrated terminal (`Ctrl+`` `) and run:

```bash
# Install UV (Rust-based package manager — mirrors report Section 3.1)
pip install uv

# Create virtual environment
uv venv .venv

# Activate it
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Initialise project and install all dependencies
uv init --no-workspace
uv add -r requirements.txt
uv sync
```

VS Code will detect the `.venv` automatically and switch the interpreter.
If not, press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

---

## Step 4 — Generate the Dataset

```bash
uv run python data/generate_dataset.py
```

This creates three CSV files in `data/`:
- `bank_voucher_interactions.csv` — customer × voucher interactions
- `vouchers_metadata.csv` — 20 gift voucher descriptions
- `customers.csv` — 2,000 bank customer profiles

---

## Step 5 — Run the Ruff Linter (CI Gate 1)

```bash
uv run ruff --version
uv run ruff check . --select F --ignore E402 --output-format grouped
```

Expected output: `All checks passed!`

In VS Code, Ruff runs automatically on save (configured in `.vscode/settings.json`).

---

## Step 6 — Run the Smoke Tests (CI Gate 2)

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: **28 passed**.

You can also run tests from the **Testing** sidebar (flask icon on the left).

---

## Step 7 — Run the Training Pipeline

```bash
uv run python train_pipeline.py
```

Or press **Ctrl+Shift+B** and select **"7: Train Pipeline"** from the task menu.

The 8-phase pipeline will:
1. Load 21,933 customer-voucher interactions
2. Build a 2,000 × 20 sparse matrix
3. Train the ALS model (20 iterations, 64 factors)
4. Generate TF-IDF semantic embeddings (20 × 189)
5. Blend scores: 65% ALS + 35% Semantic
6. Generate top-10 recommendations for all 2,000 customers
7. Write ~15,798 rows to `ab_idbi_recommendations_uat` (SQLite)
8. Validate all 6 `.pkl` artifacts

---

## Step 8 — Post-Training Artifact Checks

```bash
uv run pytest tests/post_training/ -v -k pkl
```

Expected: **22 passed** — all `.pkl` files exist, are non-empty, and load cleanly.

---

## Step 9 — View Recommendations in the DB

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from src.database import RecommendationDatabase
import pandas as pd
db = RecommendationDatabase()
print(db.read_recommendations('CUST00001').to_string(index=False))
"
```

---

## VS Code Task Runner (Quick Reference)

Press **Ctrl+Shift+B** to access all pipeline tasks:

| Task | What it does |
|---|---|
| `0: Full Pipeline` | data → lint → test → train (complete run) |
| `1: Setup UV Environment` | Install all dependencies |
| `2: Generate Dataset` | Create synthetic bank-voucher data |
| `3: Ruff Lint` | Run `--select F --ignore E402` check |
| `4: Smoke Tests` | Run 28 CI-phase tests |
| `5: Post-Training Tests` | Validate .pkl artifacts |
| `6: All Tests (50)` | Full test suite |
| `7: Train Pipeline` | Run `train_pipeline.py` |
| `8: Docker — Build Image` | Build `recommendation-engine-idbi` |
| `9: Docker — Run Tests` | Tests inside container |
| `10: Docker — Run Training` | Training inside container (libgomp1) |
| `11: Docker — Destroy` | Stop + remove container |
| `12: Clean Artifacts` | Delete .pkl, .log, .db files |

---

## VS Code Debug Launcher (F5)

Press **F5** or go to **Run & Debug** (Ctrl+Shift+D) and pick a configuration:

| Configuration | Description |
|---|---|
| `▶ Run: train_pipeline.py` | Full training with breakpoints |
| `▶ Run: Generate Dataset` | Debug dataset generation |
| `▶ Test: Smoke Tests` | Debug smoke tests |
| `▶ Test: Post-Training Artifacts` | Debug .pkl checks |
| `▶ Test: All Tests (50)` | Run entire test suite |
| `▶ Debug: Current File` | Run whatever file is open |

---

## Docker Workflow (Optional)

Mirrors the report's Section 3.4 containerisation exactly:

```bash
# Build image (installs libgomp1 — resolves ALS parallelism)
docker build -t recommendation-engine-idbi:latest .

# Check container status
docker-compose ps

# Run smoke tests inside container
docker-compose run --rm app bash -c 'uv run pytest tests/test_smoke.py -v'

# Run training inside container
docker-compose run --rm app bash -c 'uv run python train_pipeline.py'

# Teardown (zero memory footprint)
docker stop recommendation-engine-idbi && docker rm recommendation-engine-idbi
```

---

## CI/CD Pipeline (GitHub)

1. Create a new GitHub repository
2. Push this folder:
   ```bash
   git init
   git add .
   git commit -m "feat: initial IDBI recommendation engine"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin ci-cd-final
   ```
3. The pipeline in `.github/workflows/pipeline.yml` triggers automatically.
4. Watch **Actions** tab on GitHub — all 3 phases run sequentially.

---

## Project Structure Quick Reference

```
idbi-recommendation-engine/
│
├── .vscode/
│   ├── settings.json    ← Python interpreter, Ruff, pytest config
│   ├── extensions.json  ← Recommended extensions (auto-prompted)
│   ├── launch.json      ← F5 debug configurations
│   └── tasks.json       ← Ctrl+Shift+B task menu
│
├── .github/workflows/
│   └── pipeline.yml     ← 3-phase CI/CD (GitHub Actions)
│
├── src/                 ← All source modules
│   ├── config.py        ← Central config (paths, hyperparams, DB URL)
│   ├── data_loader.py   ← CSV loader + validation
│   ├── preprocessor.py  ← LabelEncoder + sparse matrix builder
│   ├── als_model.py     ← ALS collaborative filtering (implicit 0.7+)
│   ├── embedding_model.py ← TF-IDF semantic embeddings
│   ├── hybrid_scorer.py ← 65% ALS + 35% Semantic blender
│   ├── recommender.py   ← Top-N generation + DB delivery
│   ├── evaluator.py     ← Precision@K + Catalogue Coverage metrics
│   └── database.py      ← SQLite → ab_idbi_recommendations_uat
│
├── tests/
│   ├── test_smoke.py         ← 28 tests (Env + Modules + Model)
│   └── post_training/
│       └── test_artifacts.py ← 22 tests (.pkl validation)
│
├── data/
│   └── generate_dataset.py  ← Synthetic dataset generator
│
├── model_artifacts/     ← .pkl files saved here after training
├── logs/                ← pipeline.log written here
│
├── train_pipeline.py    ← 8-phase training orchestrator
├── Dockerfile           ← python:3.9-slim + libgomp1 + UV
├── docker-compose.yml   ← Bind mounts + named volumes
├── Makefile             ← CLI shortcuts for all commands
├── pyproject.toml       ← Project metadata + Ruff + pytest config
├── requirements.txt     ← Pinned dependencies
├── uv.lock              ← Deterministic dependency graph
└── .gitignore
```
