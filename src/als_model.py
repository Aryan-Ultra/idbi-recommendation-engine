"""
als_model.py — ALS Collaborative Filtering model (implicit 0.7+ API).
Filters out padding sentinels returned by implicit when fewer than N unseen items exist.
"""

import logging
import pickle
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.config import (
    ALS_FACTORS, ALS_ITERATIONS, ALS_REGULARIZATION, ALS_ALPHA,
    ALS_RANDOM_STATE, ALS_MODEL_PKL, USER_FACTORS_PKL, ITEM_FACTORS_PKL,
)

logger = logging.getLogger(__name__)

# implicit pads with float32 negative min when fewer than N unseen items exist
_ALS_SCORE_FLOOR = -1e30


class ALSModel:
    """Wrapper around implicit ALS for training and inference."""

    def __init__(self):
        self.model = AlternatingLeastSquares(
            factors=ALS_FACTORS,
            iterations=ALS_ITERATIONS,
            regularization=ALS_REGULARIZATION,
            random_state=ALS_RANDOM_STATE,
            use_gpu=False,
        )
        self.user_factors = None
        self.item_factors = None

    def train(self, user_item_matrix: sp.csr_matrix):
        """Train ALS on the user-item confidence matrix (user × item, no transpose)."""
        confidence_matrix = (user_item_matrix * ALS_ALPHA).astype(np.float32)
        logger.info("Training ALS: factors=%d, iterations=%d", ALS_FACTORS, ALS_ITERATIONS)
        self.model.fit(confidence_matrix)
        self.user_factors = self.model.user_factors   # (n_users, factors)
        self.item_factors = self.model.item_factors   # (n_items, factors)
        logger.info(
            "ALS training complete | user_factors=%s, item_factors=%s",
            self.user_factors.shape, self.item_factors.shape,
        )

    def recommend(self, user_idx: int, user_item_matrix: sp.csr_matrix, n: int = 10):
        """
        Return top-N valid (item_idx, score) tuples for a given user.
        Filters out implicit's padding sentinels (-3.4e38) and zero-padding.
        """
        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=True,
        )
        # Filter sentinels: keep only positive scores
        valid = [(int(iid), float(s)) for iid, s in zip(item_ids, scores)
                 if float(s) > _ALS_SCORE_FLOOR and float(s) > 0.0]
        return valid

    def save(self):
        with open(ALS_MODEL_PKL, "wb") as f:
            pickle.dump(self.model, f)
        with open(USER_FACTORS_PKL, "wb") as f:
            pickle.dump(self.user_factors, f)
        with open(ITEM_FACTORS_PKL, "wb") as f:
            pickle.dump(self.item_factors, f)
        logger.info("ALS artifacts saved to %s", ALS_MODEL_PKL.parent)

    @classmethod
    def load(cls) -> "ALSModel":
        als = cls()
        with open(ALS_MODEL_PKL, "rb") as f:
            als.model = pickle.load(f)
        with open(USER_FACTORS_PKL, "rb") as f:
            als.user_factors = pickle.load(f)
        with open(ITEM_FACTORS_PKL, "rb") as f:
            als.item_factors = pickle.load(f)
        return als
