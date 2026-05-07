"""
train_pipeline.py — Main training orchestrator for the IDBI Gift Voucher
                    Hybrid Recommendation System.

Mirrors the exact pipeline described in the project report:
  1. Load data
  2. Preprocess → sparse matrix
  3. Train ALS model
  4. Train Embedding model
  5. Build HybridScorer
  6. Generate top-10 recommendations for all customers
  7. Deliver to database (ab_idbi_recommendations_uat)
  8. Post-training artifact validation

Run inside Docker:  uv run python train_pipeline.py
Run locally:        uv run python train_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import LOG_FILE, LOG_LEVEL, TOP_N
from src.data_loader import load_interactions, load_vouchers_metadata, validate_interactions
from src.preprocessor import DataPreprocessor
from src.als_model import ALSModel
from src.embedding_model import EmbeddingModel
from src.hybrid_scorer import HybridScorer
from src.recommender import GiftVoucherRecommender
from src.database import RecommendationDatabase

# ─── Logging Setup ────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def banner(title: str):
    logger.info("─" * 60)
    logger.info("  %s", title)
    logger.info("─" * 60)


def main():
    t0 = time.time()
    banner("IDBI Gift Voucher Recommendation Engine — Training Pipeline")

    # ── Phase 1: Data Loading ─────────────────────────────────────────────────
    banner("Phase 1 | Data Loading")
    interactions_df = load_interactions()
    vouchers_df = load_vouchers_metadata()
    validate_interactions(interactions_df)
    logger.info(
        "Dataset loaded: %d interactions | %d customers | %d vouchers",
        len(interactions_df),
        interactions_df["customer_id"].nunique(),
        interactions_df["voucher_id"].nunique(),
    )

    # ── Phase 2: Preprocessing → Sparse Matrix ───────────────────────────────
    banner("Phase 2 | Preprocessing")
    preprocessor = DataPreprocessor()
    user_item_matrix = preprocessor.fit_transform(interactions_df)
    preprocessor.save()
    logger.info("Sparse user-item matrix: %s", user_item_matrix.shape)

    # ── Phase 3: ALS Model Training ──────────────────────────────────────────
    banner("Phase 3 | ALS Collaborative Filtering")
    als_model = ALSModel()
    als_model.train(user_item_matrix)
    als_model.save()
    logger.info("ALS model trained and saved.")

    # ── Phase 4: Semantic Embedding Model ────────────────────────────────────
    banner("Phase 4 | Semantic Embedding Model")
    embedding_model = EmbeddingModel()
    embedding_model.fit(vouchers_df)
    embedding_model.save()
    logger.info("Embedding model trained and saved.")

    # ── Phase 5: Hybrid Scoring ──────────────────────────────────────────────
    banner("Phase 5 | Hybrid Scoring Engine")
    hybrid_scorer = HybridScorer(als_model=als_model, embedding_model=embedding_model)
    logger.info("HybridScorer ready (ALS=%.0f%%, Semantic=%.0f%%)",
                hybrid_scorer._alpha * 100, hybrid_scorer._beta * 100)

    # ── Phase 6: Generate Recommendations ───────────────────────────────────
    banner("Phase 6 | Generating Recommendations (Top-10 per Customer)")
    db = RecommendationDatabase()
    recommender = GiftVoucherRecommender(
        hybrid_scorer=hybrid_scorer,
        preprocessor=preprocessor,
        vouchers_df=vouchers_df,
        db=db,
    )
    recommendations_df = recommender.generate_all_recommendations(
        user_item_matrix=user_item_matrix
    )
    logger.info("Recommendations DataFrame shape: %s", recommendations_df.shape)

    # ── Phase 7: Deliver to Database ─────────────────────────────────────────
    banner("Phase 7 | Delivering to ab_idbi_recommendations_uat")
    recommender.deliver(recommendations_df)
    row_count = db.row_count()
    logger.info("Database row count (post-delivery): %d", row_count)

    # ── Phase 8: Post-Training Artifact Validation ───────────────────────────
    banner("Phase 8 | Post-Training Artifact Validation")
    from tests.post_training.test_artifacts import run_artifact_checks
    run_artifact_checks()

    elapsed = time.time() - t0
    banner(f"Pipeline Complete in {elapsed:.1f}s")
    logger.info(
        "Delivered top-%d gift voucher recommendations for %d customers.",
        TOP_N,
        recommendations_df["customer_id"].nunique(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
