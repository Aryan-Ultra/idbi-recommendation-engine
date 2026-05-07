"""
tests/post_training/test_artifacts.py — Post-training artifact validation.

Per the project report (Phase 3 / CD phase):
  Verifies .pkl files exist in model_artifacts/, are non-empty, and are loadable.

Run:  uv run pytest tests/post_training/ -v -k pkl
"""

import pickle
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import (
    ALS_MODEL_PKL,
    USER_FACTORS_PKL,
    ITEM_FACTORS_PKL,
    EMBEDDINGS_PKL,
    USER_ENCODER_PKL,
    ITEM_ENCODER_PKL,
)

PKL_ARTIFACTS = [
    ("als_model.pkl", ALS_MODEL_PKL),
    ("user_factors.pkl", USER_FACTORS_PKL),
    ("item_factors.pkl", ITEM_FACTORS_PKL),
    ("voucher_embeddings.pkl", EMBEDDINGS_PKL),
    ("user_encoder.pkl", USER_ENCODER_PKL),
    ("item_encoder.pkl", ITEM_ENCODER_PKL),
]


class TestPklArtifacts:
    """Validates all post-training .pkl artifacts."""

    @pytest.mark.parametrize("name,path", PKL_ARTIFACTS)
    def test_pkl_file_exists(self, name: str, path: Path):
        """Each artifact .pkl file must exist after training."""
        assert path.exists(), (
            f"{name} not found at {path}. "
            "Run train_pipeline.py first."
        )

    @pytest.mark.parametrize("name,path", PKL_ARTIFACTS)
    def test_pkl_file_is_not_empty(self, name: str, path: Path):
        """Each artifact .pkl file must have non-zero byte size."""
        if not path.exists():
            pytest.skip(f"{name} does not exist — skipping size check")
        size = path.stat().st_size
        assert size > 0, f"{name} is empty (0 bytes)"

    @pytest.mark.parametrize("name,path", PKL_ARTIFACTS)
    def test_pkl_file_is_loadable(self, name: str, path: Path):
        """Each artifact must be unpicklable without corruption errors."""
        if not path.exists():
            pytest.skip(f"{name} does not exist — skipping load check")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            assert obj is not None, f"{name} loaded as None"
        except (pickle.UnpicklingError, EOFError, Exception) as exc:
            pytest.fail(f"Failed to load {name}: {exc}")

    def test_user_factors_shape(self):
        """user_factors.pkl must be a 2D numpy array (n_users × n_factors)."""
        if not USER_FACTORS_PKL.exists():
            pytest.skip("user_factors.pkl not present")
        import numpy as np
        with open(USER_FACTORS_PKL, "rb") as f:
            factors = pickle.load(f)
        assert isinstance(factors, np.ndarray), "user_factors must be ndarray"
        assert factors.ndim == 2, f"Expected 2D, got {factors.ndim}D"
        assert factors.shape[0] > 0, "user_factors has zero users"

    def test_item_factors_shape(self):
        """item_factors.pkl must be a 2D numpy array (n_items × n_factors)."""
        if not ITEM_FACTORS_PKL.exists():
            pytest.skip("item_factors.pkl not present")
        import numpy as np
        with open(ITEM_FACTORS_PKL, "rb") as f:
            factors = pickle.load(f)
        assert isinstance(factors, np.ndarray), "item_factors must be ndarray"
        assert factors.ndim == 2
        assert factors.shape[0] > 0

    def test_embeddings_shape_and_normalized(self):
        """Embeddings must be unit-normalized (cosine sim ready)."""
        if not EMBEDDINGS_PKL.exists():
            pytest.skip("voucher_embeddings.pkl not present")
        import numpy as np
        with open(EMBEDDINGS_PKL, "rb") as f:
            payload = pickle.load(f)
        embeddings = payload["embeddings"]
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        # Check unit norm (±0.01 tolerance)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01), "Embeddings are not unit-normalized"

    def test_encoder_classes_non_empty(self):
        """LabelEncoders must contain classes_ after fitting."""
        if not USER_ENCODER_PKL.exists():
            pytest.skip("user_encoder.pkl not present")
        with open(USER_ENCODER_PKL, "rb") as f:
            user_enc = pickle.load(f)
        with open(ITEM_ENCODER_PKL, "rb") as f:
            item_enc = pickle.load(f)
        assert len(user_enc.classes_) > 0, "user_encoder has no classes"
        assert len(item_enc.classes_) > 0, "item_encoder has no classes"


def run_artifact_checks():
    """Called directly from train_pipeline.py for inline post-training validation."""
    import logging
    logger = logging.getLogger(__name__)

    failed = []
    for name, path in PKL_ARTIFACTS:
        if not path.exists():
            failed.append(f"MISSING: {name}")
            continue
        if path.stat().st_size == 0:
            failed.append(f"EMPTY: {name}")
            continue
        try:
            with open(path, "rb") as f:
                pickle.load(f)
            logger.info("  ✓ %s  (%d bytes)", name, path.stat().st_size)
        except Exception as exc:
            failed.append(f"CORRUPT: {name} — {exc}")

    if failed:
        raise RuntimeError(
            "Post-training artifact validation FAILED:\n" + "\n".join(failed)
        )
    logger.info("All %d .pkl artifacts validated successfully.", len(PKL_ARTIFACTS))
