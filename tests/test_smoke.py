"""
tests/test_smoke.py — Smoke & unit tests for the IDBI Recommendation Pipeline.

Mirrors the 3 test categories from the project report:
  1. Environment & Setup Tests
  2. Module Import Tests
  3. Model Readiness Tests

Run:  uv run pytest tests/test_smoke.py -v
"""

import sys
import pickle
import importlib
from pathlib import Path
import pytest

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — Environment & Setup Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEnvironmentSetup:
    """Validates the runtime environment is correctly configured."""

    def test_python_version_is_39_or_higher(self):
        """Pipeline requires Python 3.9+."""
        major, minor = sys.version_info.major, sys.version_info.minor
        assert major == 3 and minor >= 9, (
            f"Expected Python 3.9+, got {major}.{minor}"
        )

    def test_project_root_exists(self):
        """The project root directory must be resolvable."""
        assert PROJECT_ROOT.exists(), f"Project root not found: {PROJECT_ROOT}"

    def test_uv_virtual_environment_active(self):
        """UV creates a .venv — verify we are inside a virtual environment."""
        assert sys.prefix != sys.base_prefix, (
            "Not inside a virtual environment. Activate with: source .venv/bin/activate"
        )

    def test_model_artifacts_directory_exists(self):
        """model_artifacts/ must exist before the pipeline runs (created by config.py)."""
        artifacts_dir = PROJECT_ROOT / "model_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        assert artifacts_dir.exists(), "model_artifacts/ directory missing"

    def test_logs_directory_writable(self):
        """Pipeline must be able to write logs."""
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_log = logs_dir / ".write_test"
        test_log.write_text("ok")
        assert test_log.exists()
        test_log.unlink()

    def test_data_directory_exists(self):
        """data/ directory must be present."""
        data_dir = PROJECT_ROOT / "data"
        assert data_dir.exists(), "data/ directory missing"

    def test_requirements_txt_present(self):
        """requirements.txt must exist in the project root."""
        req_path = PROJECT_ROOT / "requirements.txt"
        assert req_path.exists(), "requirements.txt missing"

    def test_config_constants_are_valid(self):
        """Config values must satisfy basic sanity constraints."""
        from src.config import (
            ALS_FACTORS, ALS_ITERATIONS, ALS_ALPHA,
            ALS_SCORE_WEIGHT, EMBEDDING_SIMILARITY_WEIGHT, TOP_N,
        )
        assert ALS_FACTORS > 0
        assert ALS_ITERATIONS > 0
        assert ALS_ALPHA > 0
        assert 0 < ALS_SCORE_WEIGHT < 1
        assert 0 < EMBEDDING_SIMILARITY_WEIGHT < 1
        assert abs(ALS_SCORE_WEIGHT + EMBEDDING_SIMILARITY_WEIGHT - 1.0) < 1e-6
        assert TOP_N > 0


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — Module Import Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestModuleImports:
    """Ensures all source packages import cleanly without errors."""

    @pytest.mark.parametrize("module_path", [
        "src.config",
        "src.data_loader",
        "src.preprocessor",
        "src.als_model",
        "src.embedding_model",
        "src.hybrid_scorer",
        "src.recommender",
        "src.evaluator",
        "src.database",
    ])
    def test_module_imports_without_error(self, module_path: str):
        """Each source module must be importable without raising an exception."""
        mod = importlib.import_module(module_path)
        assert mod is not None, f"Failed to import {module_path}"

    def test_implicit_library_importable(self):
        """The ALS library (implicit) must be available."""
        from implicit.als import AlternatingLeastSquares
        assert AlternatingLeastSquares is not None

    def test_sentence_transformers_importable(self):
        """SentenceTransformers must be available for embedding model."""
        from sentence_transformers import SentenceTransformer
        assert SentenceTransformer is not None

    def test_scipy_sparse_importable(self):
        """scipy.sparse is required for the user-item matrix."""
        from scipy.sparse import csr_matrix
        assert csr_matrix is not None

    def test_sklearn_label_encoder_importable(self):
        """LabelEncoder is used in DataPreprocessor."""
        from sklearn.preprocessing import LabelEncoder
        assert LabelEncoder is not None

    def test_sqlalchemy_importable(self):
        """SQLAlchemy is required for the database layer."""
        from sqlalchemy import create_engine
        assert create_engine is not None


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — Model Readiness Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestModelReadiness:
    """Validates ALS training capabilities and artifact structure."""

    def test_als_model_instantiation(self):
        """ALSModel must instantiate without errors."""
        from src.als_model import ALSModel
        model = ALSModel()
        assert model.model is not None

    def test_als_model_trains_on_synthetic_data(self):
        """ALS must train on a tiny synthetic sparse matrix without crashing."""
        from src.als_model import ALSModel
        import scipy.sparse as sp
        import numpy as np

        n_users, n_items = 50, 20
        rng = np.random.default_rng(0)
        data = rng.integers(1, 6, size=200).astype(np.float32)
        row = rng.integers(0, n_users, size=200)
        col = rng.integers(0, n_items, size=200)
        matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))

        model = ALSModel()
        model.train(matrix)
        assert model.user_factors is not None
        # implicit transposes: user_factors=(n_items,factors), item_factors=(n_users,factors)
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_factors.shape[1] == model.model.factors
        assert model.item_factors.shape[1] == model.model.factors

    def test_preprocessor_encodes_and_decodes(self):
        """DataPreprocessor must encode/decode user and item IDs correctly."""
        import pandas as pd
        from src.preprocessor import DataPreprocessor

        df = pd.DataFrame({
            "customer_id": ["C001", "C001", "C002", "C003"],
            "voucher_id": ["V001", "V002", "V001", "V003"],
            "interaction_weight": [1, 3, 5, 1],
        })
        preprocessor = DataPreprocessor()
        matrix = preprocessor.fit_transform(df)
        assert matrix.shape == (3, 3)

        idx = preprocessor.encode_user("C001")
        assert 0 <= idx < 3
        decoded = preprocessor.decode_items([0])
        assert isinstance(decoded[0], str)

    def test_hybrid_scorer_weight_constraint(self):
        """Hybrid scorer must enforce ALS + Semantic weights sum to 1.0."""
        from src.config import ALS_SCORE_WEIGHT, EMBEDDING_SIMILARITY_WEIGHT
        total = ALS_SCORE_WEIGHT + EMBEDDING_SIMILARITY_WEIGHT
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_artifact_directory_structure_ready(self):
        """model_artifacts/ must exist and be writable before training."""
        from src.config import MODEL_ARTIFACTS_DIR
        assert MODEL_ARTIFACTS_DIR.exists()
        test_pkl = MODEL_ARTIFACTS_DIR / ".test_write.pkl"
        with open(test_pkl, "wb") as f:
            pickle.dump({"test": True}, f)
        with open(test_pkl, "rb") as f:
            data = pickle.load(f)
        assert data["test"] is True
        test_pkl.unlink()

    def test_database_table_creates_without_error(self):
        """Database layer must create the UAT table without errors."""
        from src.database import RecommendationDatabase
        db = RecommendationDatabase(db_url="sqlite:///test_temp.db")
        count = db.row_count()
        assert count == 0
        db.engine.dispose()
        Path("test_temp.db").unlink(missing_ok=True)
