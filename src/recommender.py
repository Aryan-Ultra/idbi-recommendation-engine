"""
recommender.py — High-level recommendation facade.
"""

import logging
import pandas as pd
import scipy.sparse as sp
from datetime import datetime, timezone

from src.config import TOP_N

logger = logging.getLogger(__name__)


class GiftVoucherRecommender:
    """Orchestrates end-to-end recommendation generation for all bank customers."""

    def __init__(self, hybrid_scorer, preprocessor, vouchers_df, db):
        self.scorer = hybrid_scorer
        self.preprocessor = preprocessor
        self.vouchers_df = vouchers_df.set_index("voucher_id")
        self.db = db

    def recommend_for_user(self, customer_id: str, user_item_matrix: sp.csr_matrix) -> pd.DataFrame:
        user_idx = self.preprocessor.encode_user(customer_id)
        hybrid_results = self.scorer.recommend(
            user_idx=user_idx,
            user_item_matrix=user_item_matrix,
            n=TOP_N,
        )
        if not hybrid_results:
            return pd.DataFrame()

        records = []
        for result in hybrid_results:
            voucher_id = self.preprocessor.decode_items([result["item_idx"]])[0]
            meta = self.vouchers_df.loc[voucher_id] if voucher_id in self.vouchers_df.index else {}
            records.append({
                "customer_id": customer_id,
                "rank": result["rank"],
                "voucher_id": voucher_id,
                "voucher_name": meta.get("name", ""),
                "category": meta.get("category", ""),
                "brand": meta.get("brand", ""),
                "face_value_inr": meta.get("face_value", 0),
                "als_score": round(result["als_score"], 6),
                "semantic_score": round(result["semantic_score"], 6),
                "hybrid_score": round(result["hybrid_score"], 6),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
        return pd.DataFrame(records)

    def generate_all_recommendations(self, user_item_matrix: sp.csr_matrix, sample_size=None):
        all_customers = list(self.preprocessor.user_encoder.classes_)
        if sample_size:
            all_customers = all_customers[:sample_size]

        logger.info("Generating recommendations for %d customers...", len(all_customers))
        all_recs = []
        for i, customer_id in enumerate(all_customers):
            user_recs = self.recommend_for_user(customer_id, user_item_matrix)
            if not user_recs.empty:
                all_recs.append(user_recs)
            if (i + 1) % 500 == 0:
                logger.info("  Processed %d/%d customers", i + 1, len(all_customers))

        if not all_recs:
            return pd.DataFrame()

        final_df = pd.concat(all_recs, ignore_index=True)
        logger.info("Generated %d recommendations for %d customers",
                    len(final_df), final_df["customer_id"].nunique())
        return final_df

    def deliver(self, recommendations_df: pd.DataFrame):
        self.db.write_recommendations(recommendations_df)
        logger.info("Delivered %d recommendation rows to database", len(recommendations_df))
