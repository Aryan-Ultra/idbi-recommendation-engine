"""
preview_db.py — Prints a summary of the recommendations database.
Called by GitHub Actions Step 6b to show database results in the Actions log.
"""

import sys
sys.path.insert(0, '.')

from sqlalchemy import text
import pandas as pd
from src.database import RecommendationDatabase

db = RecommendationDatabase()

print()
print("============================================================")
print("  ab_idbi_recommendations_uat — TABLE SUMMARY")
print("============================================================")
print("  Total rows     :", db.row_count())

with db.engine.connect() as conn:
    customers = conn.execute(
        text("SELECT COUNT(DISTINCT customer_id) FROM ab_idbi_recommendations_uat")
    ).scalar()
    print("  Total customers:", customers)

    cats = pd.read_sql(
        "SELECT category, COUNT(*) as count FROM ab_idbi_recommendations_uat GROUP BY category ORDER BY count DESC",
        conn
    )
    print()
    print("  Recommendations by Category:")
    print(cats.to_string(index=False))

print()
print("  Sample — Top 10 for CUST00001:")
print("------------------------------------------------------------")
df1 = db.read_recommendations("CUST00001")
print(df1[["rank", "voucher_id", "voucher_name", "category", "hybrid_score"]].to_string(index=False))

print()
print("  Sample — Top 10 for CUST00050:")
print("------------------------------------------------------------")
df2 = db.read_recommendations("CUST00050")
print(df2[["rank", "voucher_id", "voucher_name", "category", "hybrid_score"]].to_string(index=False))

print()
print("  Sample — Top 10 for CUST00100:")
print("------------------------------------------------------------")
df3 = db.read_recommendations("CUST00100")
print(df3[["rank", "voucher_id", "voucher_name", "category", "hybrid_score"]].to_string(index=False))

print()
print("============================================================")
print("  DATABASE PREVIEW COMPLETE")
print("============================================================")
