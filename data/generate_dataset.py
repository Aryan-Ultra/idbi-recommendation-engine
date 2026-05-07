"""
generate_dataset.py — Generates a realistic synthetic bank gift-voucher interaction dataset.

Dataset mirrors a real bank loyalty programme:
  - 5,000 bank customers
  - 80 gift vouchers across 10 product categories
  - ~120,000 implicit feedback interactions (views, clicks, redemptions)

Run:  python data/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# ─── Gift Voucher Catalogue ───────────────────────────────────────────────────
VOUCHER_CATALOGUE = [
    # (voucher_id, name, category, brand, face_value_inr, description)
    # Electronics
    ("V001", "Amazon 500 Gift Card", "Electronics", "Amazon", 500, "Shop electronics, gadgets, and more on Amazon India"),
    ("V002", "Flipkart 1000 Voucher", "Electronics", "Flipkart", 1000, "Electronics and appliances at Flipkart"),
    ("V003", "Croma 2000 Gift Card", "Electronics", "Croma", 2000, "Premium electronics at Croma retail stores"),
    ("V004", "Samsung SmartBuy 1500", "Electronics", "Samsung", 1500, "Samsung official products and accessories"),
    ("V005", "Apple Store 3000 Card", "Electronics", "Apple", 3000, "Apple products and accessories"),
    ("V006", "Mi Store 500 Voucher", "Electronics", "Xiaomi", 500, "Xiaomi smartphones and IoT devices"),
    ("V007", "Sony Centre 1000", "Electronics", "Sony", 1000, "Sony audio, camera, and entertainment"),
    ("V008", "Reliance Digital 2500", "Electronics", "Reliance Digital", 2500, "All electronics under one roof"),
    # Fashion
    ("V009", "Myntra 500 Gift Card", "Fashion", "Myntra", 500, "Trending fashion for men and women"),
    ("V010", "Ajio 750 Voucher", "Fashion", "Ajio", 750, "Branded clothing and footwear"),
    ("V011", "Nykaa Fashion 500", "Fashion", "Nykaa", 500, "Curated fashion and lifestyle brands"),
    ("V012", "H&M 1000 Gift Card", "Fashion", "H&M", 1000, "International fashion at affordable prices"),
    ("V013", "Zara 1500 Gift Card", "Fashion", "Zara", 1500, "Premium international fashion brand"),
    ("V014", "Westside 500 Voucher", "Fashion", "Westside", 500, "Tata Group fashion destination"),
    ("V015", "FabIndia 750 Voucher", "Fashion", "FabIndia", 750, "Handcrafted Indian clothing and home decor"),
    ("V016", "Bata 500 Gift Card", "Fashion", "Bata", 500, "Quality footwear for the whole family"),
    # Dining & Food
    ("V017", "Swiggy 300 Gift Card", "Dining", "Swiggy", 300, "Food delivery from top restaurants"),
    ("V018", "Zomato 500 Voucher", "Dining", "Zomato", 500, "Restaurant delivery and dine-out offers"),
    ("V019", "Dominos 250 Gift Card", "Dining", "Dominos", 250, "Pizza and fast food delivery"),
    ("V020", "McDonald's 200 Card", "Dining", "McDonald's", 200, "Burgers and meals at McDonald's"),
    ("V021", "Starbucks 500 Card", "Dining", "Starbucks", 500, "Premium coffee and beverages"),
    ("V022", "KFC 300 Gift Card", "Dining", "KFC", 300, "Fried chicken and fast food"),
    ("V023", "BigBasket 500 Voucher", "Grocery", "BigBasket", 500, "Groceries and daily essentials delivered"),
    ("V024", "Blinkit 300 Voucher", "Grocery", "Blinkit", 300, "10-minute grocery delivery"),
    # Travel
    ("V025", "MakeMyTrip 2000 Card", "Travel", "MakeMyTrip", 2000, "Flights, hotels, and holiday packages"),
    ("V026", "Yatra 1500 Voucher", "Travel", "Yatra", 1500, "Domestic and international travel booking"),
    ("V027", "IRCTC eCatering 500", "Travel", "IRCTC", 500, "Train tickets and station food"),
    ("V028", "Cleartrip 1000 Card", "Travel", "Cleartrip", 1000, "Flights and hotel bookings"),
    ("V029", "OYO 1000 Gift Card", "Travel", "OYO", 1000, "Budget and premium hotel stays"),
    ("V030", "Airbnb 2500 Card", "Travel", "Airbnb", 2500, "Unique home stays across India"),
    # Health & Wellness
    ("V031", "Netmeds 500 Voucher", "Healthcare", "Netmeds", 500, "Medicines and healthcare products"),
    ("V032", "PharmEasy 300 Card", "Healthcare", "PharmEasy", 300, "Online pharmacy and diagnostics"),
    ("V033", "1mg Health Card 500", "Healthcare", "1mg", 500, "Medicines, lab tests, and consultations"),
    ("V034", "Cult.fit 1000 Voucher", "Health & Fitness", "Cult.fit", 1000, "Gym classes and fitness coaching"),
    ("V035", "HealthKart 500 Card", "Health & Fitness", "HealthKart", 500, "Supplements and fitness nutrition"),
    ("V036", "Apollo 1000 Card", "Healthcare", "Apollo", 1000, "Apollo pharmacy and diagnostics"),
    # Entertainment
    ("V037", "BookMyShow 500 Card", "Entertainment", "BookMyShow", 500, "Movie tickets and live events"),
    ("V038", "Netflix 499 Card", "Entertainment", "Netflix", 499, "One month Netflix Premium subscription"),
    ("V039", "Amazon Prime 299", "Entertainment", "Amazon", 299, "Prime Video and shopping benefits"),
    ("V040", "Spotify 119 Card", "Entertainment", "Spotify", 119, "Music streaming subscription"),
    ("V041", "SonyLIV 299 Card", "Entertainment", "SonyLIV", 299, "Sports, movies, and web series"),
    ("V042", "Hotstar 499 Card", "Entertainment", "Disney+ Hotstar", 499, "Cricket, movies, and Disney content"),
    # Fuel & Mobility
    ("V043", "Indian Oil 500 Card", "Fuel", "Indian Oil", 500, "Fuel at Indian Oil petrol stations"),
    ("V044", "HP Petro Card 500", "Fuel", "Hindustan Petroleum", 500, "Fuel at HP petrol pumps"),
    ("V045", "Bharat Petroleum 500", "Fuel", "BPCL", 500, "Fuel at Bharat Petroleum outlets"),
    ("V046", "Ola 300 Gift Card", "Mobility", "Ola", 300, "Cab rides across India"),
    ("V047", "Uber 300 Gift Card", "Mobility", "Uber", 300, "Cab rides with Uber India"),
    ("V048", "Rapido 200 Card", "Mobility", "Rapido", 200, "Bike taxi and auto rides"),
    # Education
    ("V049", "Udemy 499 Voucher", "Education", "Udemy", 499, "Online courses and professional skills"),
    ("V050", "Coursera 999 Card", "Education", "Coursera", 999, "University-grade online certification"),
    ("V051", "BYJU's 1000 Card", "Education", "BYJU's", 1000, "K-12 learning and competitive exams"),
    ("V052", "Unacademy 500 Card", "Education", "Unacademy", 500, "Exam prep for UPSC, JEE, NEET"),
    # Home & Lifestyle
    ("V053", "IKEA 1000 Gift Card", "Home", "IKEA", 1000, "Furniture and home furnishings"),
    ("V054", "Pepperfry 1500 Card", "Home", "Pepperfry", 1500, "Furniture and home decor online"),
    ("V055", "Urban Ladder 2000", "Home", "Urban Ladder", 2000, "Premium furniture and home accessories"),
    ("V056", "HomeCentre 1000", "Home", "HomeCentre", 1000, "Home furniture and decor at Lifestyle"),
    # Banking & Financial
    ("V057", "PVR Cinemas 500", "Entertainment", "PVR", 500, "Movie tickets at PVR multiplex chain"),
    ("V058", "INOX 500 Card", "Entertainment", "INOX", 500, "Movie entertainment at INOX cinemas"),
    ("V059", "Paytm Mall 500", "Electronics", "Paytm", 500, "Electronics and fashion on Paytm Mall"),
    ("V060", "Nykaa Beauty 500", "Beauty", "Nykaa", 500, "Cosmetics and beauty products"),
    ("V061", "Lakme 400 Card", "Beauty", "Lakme", 400, "Lakme beauty products and salon services"),
    ("V062", "Forest Essentials 750", "Beauty", "Forest Essentials", 750, "Luxury Ayurvedic skincare"),
    ("V063", "Mamaearth 300 Card", "Beauty", "Mamaearth", 300, "Natural and toxin-free beauty products"),
    ("V064", "Boat 500 Gift Card", "Electronics", "Boat", 500, "Earphones, headphones, and accessories"),
    ("V065", "Noise 500 Card", "Electronics", "Noise", 500, "Smartwatches and fitness bands"),
    ("V066", "Lenskart 750 Card", "Healthcare", "Lenskart", 750, "Prescription glasses and sunglasses"),
    ("V067", "MedLife 300 Card", "Healthcare", "MedLife", 300, "Online pharmacy and wellness"),
    ("V068", "EaseMyTrip 1000", "Travel", "EaseMyTrip", 1000, "Discounted flights and hotels"),
    ("V069", "RedBus 500 Card", "Travel", "RedBus", 500, "Bus ticket booking across India"),
    ("V070", "Meesho 300 Card", "Fashion", "Meesho", 300, "Affordable fashion and lifestyle"),
    ("V071", "Tata CLiQ 1000", "Electronics", "Tata CLiQ", 1000, "Premium and luxury brand shopping"),
    ("V072", "Shoppers Stop 1000", "Fashion", "Shoppers Stop", 1000, "Department store for branded fashion"),
    ("V073", "Big Bazaar 500", "Grocery", "Big Bazaar", 500, "Supermarket for groceries and essentials"),
    ("V074", "Spencer's 500 Card", "Grocery", "Spencer's", 500, "Fresh produce and packaged food"),
    ("V075", "JioMart 300 Card", "Grocery", "JioMart", 300, "Grocery delivery from Reliance"),
    ("V076", "Decathlon 1000", "Sports", "Decathlon", 1000, "Sports gear and outdoor equipment"),
    ("V077", "Nike Gift Card 1500", "Sports", "Nike", 1500, "Athletic footwear and apparel"),
    ("V078", "Adidas 1000 Card", "Sports", "Adidas", 1000, "Sports and lifestyle products"),
    ("V079", "Puma 750 Card", "Sports", "Puma", 750, "Athletic footwear and casual wear"),
    ("V080", "Titan 2000 Card", "Lifestyle", "Titan", 2000, "Watches and jewellery from Titan"),
]
def generate_customer_segments(n_customers: int) -> pd.DataFrame:
    """Generate customer demographic and segment data."""
    age_groups = rng.choice(
        ["18-25", "26-35", "36-45", "46-60", "60+"],
        size=n_customers,
        p=[0.15, 0.30, 0.28, 0.18, 0.09],
    )
    income_segments = rng.choice(
        ["Mass", "Affluent", "HNI", "UHNI"],
        size=n_customers,
        p=[0.45, 0.35, 0.15, 0.05],
    )
    cities = rng.choice(
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
         "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"],
        size=n_customers,
        p=[0.18, 0.17, 0.15, 0.10, 0.10, 0.09, 0.08, 0.06, 0.04, 0.03],
    )
    return pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
        "age_group": age_groups,
        "income_segment": income_segments,
        "city": cities,
    })
def generate_interactions(customers_df: pd.DataFrame, n_interactions: int) -> pd.DataFrame:
    """
    Generate implicit feedback interactions (views=1, clicks=3, redemptions=5).
    Customers are biased toward categories matching their income/age profile.
    """
    vouchers_df = pd.DataFrame(VOUCHER_CATALOGUE,
                                columns=["voucher_id", "name", "category", "brand",
                                         "face_value", "description"])

    # Category affinity per income segment
    category_affinity = {
        "Mass": {"Grocery": 4, "Dining": 3, "Fashion": 3, "Fuel": 3, "Mobility": 2,
                 "Entertainment": 2, "Healthcare": 2, "Electronics": 1},
        "Affluent": {"Electronics": 4, "Fashion": 4, "Travel": 3, "Dining": 3,
                     "Entertainment": 3, "Health & Fitness": 2, "Beauty": 2, "Home": 2},
        "HNI": {"Travel": 5, "Electronics": 4, "Home": 4, "Beauty": 3,
                "Fashion": 3, "Health & Fitness": 3, "Entertainment": 2, "Lifestyle": 2},
        "UHNI": {"Travel": 5, "Lifestyle": 5, "Electronics": 4, "Home": 4,
                 "Beauty": 4, "Health & Fitness": 3, "Fashion": 3, "Entertainment": 3},
    }

    all_interactions = []
    interaction_types = {1: "view", 3: "click", 5: "redemption"}

    for _, customer in customers_df.iterrows():
        segment = customer["income_segment"]
        affinity = category_affinity[segment]

        # Determine how many interactions this customer has
        n_cust_interactions = int(rng.integers(10, 80))

        for _ in range(n_cust_interactions):
            # Pick a category with affinity weighting
            categories = list(affinity.keys())
            weights = np.array([affinity.get(c, 1) for c in categories], dtype=float)
            weights = weights / weights.sum()
            chosen_cat = rng.choice(categories, p=weights)

            # Get vouchers in that category
            cat_vouchers = vouchers_df[vouchers_df["category"] == chosen_cat]
            if len(cat_vouchers) == 0:
                cat_vouchers = vouchers_df
            voucher = cat_vouchers.sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]

            # Interaction type weighted toward views
            interaction_weight = rng.choice([1, 3, 5], p=[0.60, 0.28, 0.12])

            all_interactions.append({
                "customer_id": customer["customer_id"],
                "voucher_id": voucher["voucher_id"],
                "interaction_weight": interaction_weight,
                "interaction_type": interaction_types[interaction_weight],
            })

    interactions_df = pd.DataFrame(all_interactions)

    # Aggregate duplicate (customer, voucher) pairs by taking max weight
    interactions_df = (
        interactions_df
        .groupby(["customer_id", "voucher_id"], as_index=False)
        .agg({"interaction_weight": "max", "interaction_type": "last"})
    )

    return interactions_df
def main():
    print("═" * 60)
    print("  IDBI Gift Voucher Dataset Generator")
    print("═" * 60)

    N_CUSTOMERS = 5000
    

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Generating {N_CUSTOMERS} bank customers...")
    customers_df = generate_customer_segments(N_CUSTOMERS)
    customers_df.to_csv(output_dir / "customers.csv", index=False)
    print(f"      ✓ Saved customers.csv  ({len(customers_df)} rows)")

    print(f"\n[2/3] Generating voucher catalogue ({len(VOUCHER_CATALOGUE)} vouchers)...")
    vouchers_df = pd.DataFrame(VOUCHER_CATALOGUE,
                                columns=["voucher_id", "name", "category", "brand",
                                         "face_value", "description"])
    vouchers_df.to_csv(output_dir / "vouchers_metadata.csv", index=False)
    print(f"      ✓ Saved vouchers_metadata.csv  ({len(vouchers_df)} rows)")

    print("\n[3/3] Generating interaction dataset...")
    interactions_df = generate_interactions(customers_df, 0)
    interactions_df.to_csv(output_dir / "bank_voucher_interactions.csv", index=False)
    print(f"      ✓ Saved bank_voucher_interactions.csv  ({len(interactions_df)} rows)")

    print("\n─── Summary ───────────────────────────────────────────")
    print(f"  Customers       : {len(customers_df):,}")
    print(f"  Vouchers        : {len(vouchers_df):,}")
    print(f"  Interactions    : {len(interactions_df):,}")
    print(f"  Sparsity        : {1 - len(interactions_df)/(len(customers_df)*len(vouchers_df)):.2%}")
    print("  Categories      :", interactions_df.merge(vouchers_df, on="voucher_id")["category"].nunique())
    print("═" * 60)
if __name__ == "__main__":
    main()
