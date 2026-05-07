"""
database.py — Database layer for delivering recommendations.
Uses SQLite (free substitute for IDBI's Oracle/PostgreSQL).
Maps logically to the analytics_public schema / ab_idbi_recommendations_uat table.
"""

import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.config import DB_URL, DB_TABLE

logger = logging.getLogger(__name__)


class RecommendationDatabase:
    """Wraps SQLAlchemy engine for reading/writing the recommendations UAT table."""

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url, echo=False)
        self._ensure_table()

    def _ensure_table(self):
        """Create the recommendations table if it does not exist."""
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     TEXT NOT NULL,
            rank            INTEGER NOT NULL,
            voucher_id      TEXT NOT NULL,
            voucher_name    TEXT,
            category        TEXT,
            brand           TEXT,
            face_value_inr  REAL,
            als_score       REAL,
            semantic_score  REAL,
            hybrid_score    REAL,
            generated_at    TEXT
        )
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
        logger.info("UAT table '%s' is ready.", DB_TABLE)

    def write_recommendations(self, df: pd.DataFrame, replace: bool = True):
        """
        Write recommendations to the UAT table.
        By default, replaces all previous data (clean-run semantics matching the pipeline).
        """
        try:
            if replace:
                with self.engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM {DB_TABLE}"))
                    conn.commit()
            df.to_sql(DB_TABLE, con=self.engine, if_exists="append", index=False)
            logger.info("Wrote %d rows to '%s'.", len(df), DB_TABLE)
        except SQLAlchemyError as exc:
            logger.error("Database write failed: %s", exc)
            raise

    def read_recommendations(self, customer_id: str | None = None) -> pd.DataFrame:
        """Read back recommendations; optionally filter by customer_id."""
        query = f"SELECT * FROM {DB_TABLE}"
        if customer_id:
            query += f" WHERE customer_id = '{customer_id}' ORDER BY rank"
        return pd.read_sql(query, con=self.engine)

    def row_count(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {DB_TABLE}"))
            return result.scalar()
