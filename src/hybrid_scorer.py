"""
hybrid_scorer.py — Combines ALS collaborative filtering scores with semantic
embedding scores to produce a unified hybrid ranking for gift voucher recommendations.

Hybrid Score = α × ALS_score + β × Semantic_score
where α + β = 1.0  (configured in config.py)
"""

import logging
import numpy as np
import scipy.sparse as sp

from src.config import ALS_SCORE_WEIGHT, EMBEDDING_SIMILARITY_WEIGHT, TOP_N
from src.als_model import ALSModel
from src.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class HybridScorer:
    """
    Merges ALS and semantic embedding scores.
    For each user:
      1. Get top-K ALS candidates (collaborative signal).
      2. Compute semantic similarity of those candidates to user's history.
      3. Blend the two scores and return the final top-N ranked list.
    """

    def __init__(self, als_model: ALSModel, embedding_model: EmbeddingModel):
        self.als_model = als_model
        self.embedding_model = embedding_model
        self._alpha = ALS_SCORE_WEIGHT
        self._beta = EMBEDDING_SIMILARITY_WEIGHT
        assert abs(self._alpha + self._beta - 1.0) < 1e-6, (
            f"ALS and Embedding weights must sum to 1.0, got {self._alpha + self._beta}"
        )

    def recommend(
        self,
        user_idx: int,
        user_item_matrix: sp.csr_matrix,
        n: int = TOP_N,
        als_pool_size: int = 50,
    ) -> List[dict]:
        """
        Generate top-N hybrid recommendations for a single user.

        Returns a list of dicts:
          [{"rank": 1, "item_idx": X, "als_score": ..., "sem_score": ..., "hybrid_score": ...}, ...]
        """
        # Step 1: ALS candidates (larger pool for re-ranking)
        als_candidates = self.als_model.recommend(
            user_idx=user_idx,
            user_item_matrix=user_item_matrix,
            n=als_pool_size,
        )
        if not als_candidates:
            logger.warning("No ALS candidates for user_idx=%d", user_idx)
            return []

        candidate_item_indices = [item_idx for item_idx, _ in als_candidates]
        als_scores_raw = np.array([score for _, score in als_candidates])

        # Normalise ALS scores to [0, 1]
        als_min, als_max = als_scores_raw.min(), als_scores_raw.max()
        if als_max > als_min:
            als_scores_norm = (als_scores_raw - als_min) / (als_max - als_min)
        else:
            als_scores_norm = np.ones_like(als_scores_raw)

        # Step 2: Get user's interaction history for semantic profiling
        user_row = user_item_matrix[user_idx]
        interacted_item_indices = user_row.indices.tolist()

        # Step 3: Semantic scores for all ALS candidates
        semantic_scores = self.embedding_model.score_items_for_user(
            user_interacted_item_indices=interacted_item_indices,
            candidate_item_indices=candidate_item_indices,
        )

        sem_scores_raw = np.array(
            [semantic_scores.get(idx, 0.0) for idx in candidate_item_indices]
        )

        # Step 4: Hybrid blend
        hybrid_scores = self._alpha * als_scores_norm + self._beta * sem_scores_raw
        ranked_order = np.argsort(hybrid_scores)[::-1][:n]

        results = []
        for rank, order_idx in enumerate(ranked_order, start=1):
            results.append({
                "rank": rank,
                "item_idx": candidate_item_indices[order_idx],
                "als_score": float(als_scores_norm[order_idx]),
                "semantic_score": float(sem_scores_raw[order_idx]),
                "hybrid_score": float(hybrid_scores[order_idx]),
            })
        return results
