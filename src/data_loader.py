"""
data_loader.py — Loads and validates the bank voucher interaction dataset.
"""

import logging
import pandas as pd

from src.config import DATASET_PATH, VOUCHERS_METADATA_PATH

logger = logging.getLogger(__name__)


def load_interactions() -> pd.DataFrame:
    """Load the customer-voucher interaction dataset from CSV."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run: python data/generate_dataset.py"
        )
    df = pd.read_csv(DATASET_PATH)
    required_cols = {"customer_id", "voucher_id", "interaction_weight"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")
    logger.info("Loaded interactions: %d rows, %d customers, %d vouchers",
                len(df), df["customer_id"].nunique(), df["voucher_id"].nunique())
    return df


def load_vouchers_metadata() -> pd.DataFrame:
    """Load the gift voucher catalogue metadata."""
    if not VOUCHERS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Vouchers metadata not found at {VOUCHERS_METADATA_PATH}. "
            "Run: python data/generate_dataset.py"
        )
    df = pd.read_csv(VOUCHERS_METADATA_PATH)
    logger.info("Loaded voucher catalogue: %d vouchers across %d categories",
                len(df), df["category"].nunique())
    return df


def validate_interactions(df: pd.DataFrame) -> bool:
    """Run basic quality checks on the interactions dataframe."""
    assert df["customer_id"].notna().all(), "Null customer_ids found"
    assert df["voucher_id"].notna().all(), "Null voucher_ids found"
    assert (df["interaction_weight"] > 0).all(), "Non-positive interaction weights found"
    assert df[["customer_id", "voucher_id"]].duplicated().sum() == 0, "Duplicate interactions found"
    return True
