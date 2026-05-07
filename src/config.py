"""
config.py — Central configuration for the IDBI Gift Voucher Recommendation Engine.
All tuneable parameters, paths, and database settings live here.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist at import time
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "bank_voucher_interactions.csv"
VOUCHERS_METADATA_PATH = DATA_DIR / "vouchers_metadata.csv"

# ─── ALS Hyperparameters ──────────────────────────────────────────────────────
ALS_FACTORS = 64           # Latent factor dimensions
ALS_ITERATIONS = 20        # Training epochs
ALS_REGULARIZATION = 0.01  # L2 regularisation
ALS_ALPHA = 40             # Confidence scaling for implicit feedback
ALS_RANDOM_STATE = 42

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim; fast & free on HuggingFace
EMBEDDING_SIMILARITY_WEIGHT = 0.35           # Weight of semantic score in hybrid
ALS_SCORE_WEIGHT = 0.65                      # Weight of ALS score in hybrid

# ─── Recommendation Output ────────────────────────────────────────────────────
TOP_N = 10                  # Top-N gift vouchers to recommend per customer

# ─── Database (SQLite — free substitute for IDBI Oracle) ──────────────────────
DB_PATH = PROJECT_ROOT / "idbi_recommendations.db"
DB_URL = f"sqlite:///{DB_PATH}"
DB_TABLE = "ab_idbi_recommendations_uat"
DB_SCHEMA = "analytics_public"   # logical schema tag (SQLite ignores schemas)

# ─── Artifact Filenames ───────────────────────────────────────────────────────
ALS_MODEL_PKL = MODEL_ARTIFACTS_DIR / "als_model.pkl"
USER_FACTORS_PKL = MODEL_ARTIFACTS_DIR / "user_factors.pkl"
ITEM_FACTORS_PKL = MODEL_ARTIFACTS_DIR / "item_factors.pkl"
EMBEDDINGS_PKL = MODEL_ARTIFACTS_DIR / "voucher_embeddings.pkl"
USER_ENCODER_PKL = MODEL_ARTIFACTS_DIR / "user_encoder.pkl"
ITEM_ENCODER_PKL = MODEL_ARTIFACTS_DIR / "item_encoder.pkl"
SPARSE_MATRIX_PKL = MODEL_ARTIFACTS_DIR / "user_item_sparse.pkl"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
