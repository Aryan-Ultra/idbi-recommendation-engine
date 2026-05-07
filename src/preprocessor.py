"""
preprocessor.py — Encodes users/items and builds the sparse user-item matrix.
"""

import logging
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from src.config import (
    USER_ENCODER_PKL,
    ITEM_ENCODER_PKL,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Encodes raw IDs into integer indices and builds a sparse confidence matrix."""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.n_users: int = 0
        self.n_items: int = 0

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Fit encoders on interaction data and return a sparse user-item matrix.
        Matrix entries are the interaction weights (implicit confidence values).
        """
        user_idx = self.user_encoder.fit_transform(df["customer_id"])
        item_idx = self.item_encoder.fit_transform(df["voucher_id"])
        weights = df["interaction_weight"].values.astype(np.float32)

        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)

        matrix = sp.csr_matrix(
            (weights, (user_idx, item_idx)),
            shape=(self.n_users, self.n_items),
        )
        logger.info(
            "Sparse matrix: %d users × %d items | density=%.4f%%",
            self.n_users,
            self.n_items,
            matrix.nnz / (self.n_users * self.n_items) * 100,
        )
        return matrix

    def save(self):
        """Persist encoders and sparse matrix to disk."""
        with open(USER_ENCODER_PKL, "wb") as f:
            pickle.dump(self.user_encoder, f)
        with open(ITEM_ENCODER_PKL, "wb") as f:
            pickle.dump(self.item_encoder, f)
        logger.info("Encoders saved to %s and %s", USER_ENCODER_PKL, ITEM_ENCODER_PKL)

    @classmethod
    def load(cls) -> "DataPreprocessor":
        """Load a previously fitted preprocessor from disk."""
        preprocessor = cls()
        with open(USER_ENCODER_PKL, "rb") as f:
            preprocessor.user_encoder = pickle.load(f)
        with open(ITEM_ENCODER_PKL, "rb") as f:
            preprocessor.item_encoder = pickle.load(f)
        preprocessor.n_users = len(preprocessor.user_encoder.classes_)
        preprocessor.n_items = len(preprocessor.item_encoder.classes_)
        return preprocessor

    def encode_user(self, customer_id: str) -> int:
        return int(self.user_encoder.transform([customer_id])[0])

    def encode_item(self, voucher_id: str) -> int:
        return int(self.item_encoder.transform([voucher_id])[0])

    def decode_items(self, item_indices: list) -> list:
        return list(self.item_encoder.inverse_transform(item_indices))
