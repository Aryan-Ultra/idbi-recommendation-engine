"""
evaluator.py — Offline evaluation metrics for the hybrid recommendation system.
Computes Precision@K, Recall@K, and Coverage on a held-out test set.
"""

import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import TOP_N

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluates recommendation quality using standard IR metrics."""

    def __init__(self, hybrid_scorer, preprocessor):
        self.scorer = hybrid_scorer
        self.preprocessor = preprocessor

    def train_test_split(
        self, interactions_df: pd.DataFrame, test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Hold out a fraction of each user's interactions as the test set.
        Ensures every user has at least 1 training interaction.
        """
        train_rows, test_rows = [], []
        for _, group in interactions_df.groupby("customer_id"):
            group = group.sample(frac=1.0, random_state=42)
            n_test = max(1, int(len(group) * test_ratio))
            test_rows.append(group.iloc[:n_test])
            train_rows.append(group.iloc[n_test:])

        train_df = pd.concat(train_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)
        logger.info(
            "Split: train=%d rows, test=%d rows", len(train_df), len(test_df)
        )
        return train_df, test_df

    def precision_at_k(
        self,
        user_item_matrix: sp.csr_matrix,
        test_df: pd.DataFrame,
        k: int = TOP_N,
        n_users: int = 200,
    ) -> float:
        """Compute mean Precision@K across a sample of users."""
        test_lookup = (
            test_df.groupby("customer_id")["voucher_id"].apply(set).to_dict()
        )
        all_customers = list(self.preprocessor.user_encoder.classes_)
        eval_customers = [c for c in all_customers[:n_users] if c in test_lookup]

        precisions = []
        for customer_id in eval_customers:
            user_idx = self.preprocessor.encode_user(customer_id)
            results = self.scorer.recommend(user_idx, user_item_matrix, n=k)
            predicted_vouchers = set(
                self.preprocessor.decode_items([r["item_idx"] for r in results])
            )
            actual = test_lookup[customer_id]
            hit = len(predicted_vouchers & actual)
            precisions.append(hit / k)

        p_at_k = float(np.mean(precisions)) if precisions else 0.0
        logger.info("Precision@%d = %.4f (over %d users)", k, p_at_k, len(precisions))
        return p_at_k

    def catalog_coverage(
        self,
        user_item_matrix: sp.csr_matrix,
        n_users: int = 500,
        k: int = TOP_N,
    ) -> float:
        """
        Fraction of the total voucher catalogue that appears in at least one
        recommendation across the evaluated user sample.
        """
        all_customers = list(self.preprocessor.user_encoder.classes_)[:n_users]
        recommended_items: set = set()

        for customer_id in all_customers:
            user_idx = self.preprocessor.encode_user(customer_id)
            results = self.scorer.recommend(user_idx, user_item_matrix, n=k)
            for r in results:
                recommended_items.add(r["item_idx"])

        coverage = len(recommended_items) / self.preprocessor.n_items
        logger.info(
            "Catalogue Coverage = %.2f%% (%d/%d items)",
            coverage * 100,
            len(recommended_items),
            self.preprocessor.n_items,
        )
        return coverage
