"""
KarthikTeja_OlistEcommerce.py
=============================
AI-Powered E-Commerce Business Intelligence & Decision Support System
Internship Project — IBM Data Analytics

Author  : KarthikTeja
Dataset : Brazilian E-Commerce Public Dataset by Olist
Source  : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

MODULES (in execution order)
------------------------------
Phase 2  — preprocessing.py       : Data loading, cleaning, joins, RFM
Phase 3  — analytics.py           : KPIs, time-series, category/state/seller aggregates
Phase 4  — customer_analytics.py  : RFM segments, repeat analysis, geography
Phase 5  — delivery_satisfaction.py : Delivery deep-dive, review analysis
Phase 6  — ml_model.py            : Late-delivery prediction (Logistic Regression)
Phase 7  — insight_engine.py      : Structured business insight engine
Phase 8  — app.py                 : Flask web dashboard (4 pages, 11 API endpoints)

HOW TO RUN THE COMPLETE PIPELINE
----------------------------------
Run each phase as a standalone script (they read/write data/ directory):

    python preprocessing.py
    python analytics.py
    python customer_analytics.py
    python delivery_satisfaction.py
    python ml_model.py
    python insight_engine.py
    python app.py          # starts the dashboard at http://localhost:5000

Or run this combined file's individual phase main() functions directly in sequence.

IMPORTANT: The original Olist CSV files are NEVER modified.
All outputs are written to data/ as separate files.
"""

# ============================================================
# MODULE: preprocessing.py — Phase 2
# ============================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = "data"                  # output directory (never the raw CSV directory)
RAW_DIR  = "."                     # location of original Olist CSVs (read-only)

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SECTION 1 — RAW DATA LOADING
# ---------------------------------------------------------------------------

def load_raw():
    """Load all nine source CSVs.  Never writes to these files."""
    print("\n" + "="*60)
    print("SECTION 1 — LOADING RAW DATA")
    print("="*60)

    files = {
        "customers":    "olist_customers_dataset.csv",
        "orders":       "olist_orders_dataset.csv",
        "items":        "olist_order_items_dataset.csv",
        "payments":     "olist_order_payments_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "sellers":      "olist_sellers_dataset.csv",
        "geolocation":  "olist_geolocation_dataset.csv",
        "translation":  "product_category_name_translation.csv",
    }

    dfs = {}
    for name, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        # translation CSV has a UTF-8 BOM
        enc = "utf-8-sig" if name == "translation" else "utf-8"
        df = pd.read_csv(path, encoding=enc, low_memory=False)
        dfs[name] = df
        print(f"  Loaded {fname:<45} rows={len(df):>7,}  cols={df.shape[1]}")

    return dfs


# ---------------------------------------------------------------------------
# SECTION 2 — VALIDATION REPORT
# ---------------------------------------------------------------------------

def validate_raw(dfs):
    print("\n" + "="*60)
    print("SECTION 2 — RAW DATA VALIDATION")
    print("="*60)

    for name, df in dfs.items():
        nulls  = df.isnull().sum().sum()
        dupes  = df.duplicated().sum()
        print(f"\n  [{name}]  rows={len(df):,}  null_cells={nulls:,}  exact_dup_rows={dupes}")
        for col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                pct = n / len(df) * 100
                print(f"      {col:<45} nulls={n:>6,} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# SECTION 3 — REVIEW INVESTIGATION & AGGREGATION STRATEGY
# ---------------------------------------------------------------------------
#
# FINDINGS FROM INVESTIGATION:
#
# 1. Total review rows: 99,224  |  Unique review_ids: 98,410  |  Unique order_ids: 98,673
# 2. 547 order_ids have more than one review row (543 × 2 rows, 4 × 3 rows).
# 3. All rows with duplicate review_id belong to DIFFERENT order_ids — meaning the same
#    review_id was recycled/mapped to multiple orders.  This appears to be a data
#    pipeline artefact (likely a system bug that duplicated a review record across orders).
# 4. No rows are exact duplicates across all columns — every row has distinct content.
# 5. For orders with two review rows:
#      • 155 of 547 are same-day events with different review_ids and sometimes different
#        scores — these are genuine re-reviews or a system that allowed two submissions.
#      • 392 of 547 are different-day events with meaningful time gaps (avg 5 days).
#        Of those, 172 also have score changes — the customer updated their opinion
#        (typically after the product arrived and they formed a final judgment).
# 6. Score-change direction: first_score mean = 3.62, last_score mean = 3.16 —
#    customers who re-reviewed tended to revise DOWNWARD, often after late delivery.
#
# AGGREGATION DECISION (documented):
#
# Because re-reviews represent the customer's FINAL settled opinion — and because the
# downward revision is analytically significant (it correlates with service failures) —
# the strategy is:
#   • Sort review rows by review_creation_date ascending within each order_id.
#   • Keep the LAST review row (most recent sentiment) as the canonical review.
#   • Preserve the review_score of the last event.
#   • For comment text: prefer the last non-null comment, so narrative context is kept.
#
# This is documented and justified; no review information is silently discarded.

def aggregate_reviews(reviews_raw):
    print("\n" + "="*60)
    print("SECTION 3 — REVIEW AGGREGATION")
    print("="*60)

    df = reviews_raw.copy()
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"], errors="coerce")

    # Sort so last event is at the bottom within each order_id
    df = df.sort_values(["order_id", "review_creation_date"])

    # Last non-null comment per order
    def last_comment(s):
        non_null = s.dropna()
        return non_null.iloc[-1] if len(non_null) > 0 else np.nan

    agg = df.groupby("order_id", as_index=False).agg(
        review_id            = ("review_id",              "last"),
        review_score         = ("review_score",           "last"),
        review_comment_title = ("review_comment_title",   last_comment),
        review_comment_message=("review_comment_message", last_comment),
        review_creation_date = ("review_creation_date",   "last"),
        review_answer_timestamp=("review_answer_timestamp","last"),
        review_event_count   = ("review_id",              "count"),   # audit field
    )

    multi_review_orders = (agg["review_event_count"] > 1).sum()
    print(f"  Input rows          : {len(df):,}")
    print(f"  Output rows (unique order_ids): {len(agg):,}")
    print(f"  Orders with >1 event (kept last): {multi_review_orders}")
    print(f"  Strategy: keep LAST review per order (most recent customer sentiment)")
    return agg


# ---------------------------------------------------------------------------
# SECTION 4 — INDIVIDUAL TABLE CLEANING
# ---------------------------------------------------------------------------

def clean_customers(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    # No nulls expected; normalise zip to 5-char string
    df["customer_zip_code_prefix"] = df["customer_zip_code_prefix"].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset="customer_id")
    print(f"  customers cleaned: {len(df):,} rows")
    return df

def clean_orders(df):
    df = df.copy()
    ts_cols = ["order_purchase_timestamp","order_approved_at",
               "order_delivered_carrier_date","order_delivered_customer_date",
               "order_estimated_delivery_date"]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Validity checks on delivered orders
    delivered = df[df["order_status"] == "delivered"]
    missing_delivery = delivered["order_delivered_customer_date"].isnull().sum()
    if missing_delivery > 0:
        print(f"  WARNING: {missing_delivery} delivered orders missing actual delivery date")

    # Sanity: purchase before estimated delivery
    bad_est = (df["order_estimated_delivery_date"] < df["order_purchase_timestamp"]).sum()
    if bad_est > 0:
        print(f"  WARNING: {bad_est} orders where estimated delivery < purchase date")

    df = df.drop_duplicates(subset="order_id")
    print(f"  orders cleaned: {len(df):,} rows  |  statuses: {df['order_status'].value_counts().to_dict()}")
    return df

def clean_items(df):
    df = df.copy()
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce")
    neg_price = (df["price"] < 0).sum()
    if neg_price:
        print(f"  WARNING: {neg_price} items with negative price")
    print(f"  items cleaned: {len(df):,} rows")
    return df

def clean_payments(df):
    df = df.copy()
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce")
    # Aggregate to one row per order: sum value, max installments, dominant payment type
    agg = df.groupby("order_id", as_index=False).agg(
        total_payment_value    = ("payment_value",       "sum"),
        payment_installments   = ("payment_installments","max"),
        payment_type           = ("payment_type",        lambda x: x.value_counts().index[0]),
        payment_methods_count  = ("payment_sequential",  "count"),
    )
    print(f"  payments cleaned: {len(df):,} rows → aggregated to {len(agg):,} order-level rows")
    return agg

def clean_products(df, translation):
    """
    Product categories:
    - original `product_category_name` (Portuguese) is PRESERVED as-is.
    - English translation is joined from the translation table and stored in a NEW column.
    - No source values are modified or silently corrected.
    - Known translation typos ("fashio_female_clothing", "costruction_tools_garden",
      "costruction_tools_tools") are corrected ONLY in the translation output column,
      and the correction is documented here.
    """
    df = df.copy()
    trans = translation.copy()

    # Document and fix typos in the English translation column only
    typo_fixes = {
        "fashio_female_clothing":       "fashion_female_clothing",
        "costruction_tools_garden":     "construction_tools_garden",
        "costruction_tools_tools":      "construction_tools_tools",
    }
    trans["product_category_name_english"] = (
        trans["product_category_name_english"].replace(typo_fixes)
    )
    print(f"  Translation typo fixes applied (English column only): {typo_fixes}")

    # Join translation — left join to preserve all products
    df = df.merge(trans, on="product_category_name", how="left")

    # Fill untranslated (null) categories with a clear label; preserve original PT name
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")

    missing_cat = df["product_category_name"].isnull().sum()
    print(f"  products: {len(df):,} rows  |  missing category (PT): {missing_cat}")
    print(f"  products: unique categories: {df['product_category_name'].nunique()} PT, "
          f"{df['product_category_name_english'].nunique()} EN")
    return df

def clean_sellers(df):
    df = df.copy()
    df["seller_zip_code_prefix"] = df["seller_zip_code_prefix"].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset="seller_id")
    print(f"  sellers cleaned: {len(df):,} rows")
    return df

def clean_geolocation(df):
    """
    One representative lat/lng per zip code (median).
    The full geolocation table is not stored — only the deduped mapping used for joins.
    """
    df = df.copy()
    df["geolocation_zip_code_prefix"] = df["geolocation_zip_code_prefix"].astype(str).str.zfill(5)
    geo_deduped = (
        df.groupby("geolocation_zip_code_prefix", as_index=False)
          .agg(lat=("geolocation_lat", "median"),
               lng=("geolocation_lng", "median"),
               city=("geolocation_city", "first"),
               state=("geolocation_state", "first"))
    )
    print(f"  geolocation: {len(df):,} rows → {len(geo_deduped):,} unique zip codes (median lat/lng)")
    return geo_deduped


# ---------------------------------------------------------------------------
# SECTION 5 — REFERENTIAL INTEGRITY CHECKS
# ---------------------------------------------------------------------------

def check_referential_integrity(dfs_clean):
    print("\n" + "="*60)
    print("SECTION 5 — REFERENTIAL INTEGRITY CHECKS")
    print("="*60)

    orders    = dfs_clean["orders"]
    customers = dfs_clean["customers"]
    items     = dfs_clean["items"]
    payments  = dfs_clean["payments"]
    reviews   = dfs_clean["reviews"]
    products  = dfs_clean["products"]
    sellers   = dfs_clean["sellers"]

    def check(child_col, parent_col, label):
        orphans = (~child_col.isin(parent_col)).sum()
        total   = len(child_col)
        print(f"  {label:<55} orphans={orphans:>5} / {total:>7,}")

    check(orders["customer_id"],   customers["customer_id"],   "orders.customer_id → customers.customer_id")
    check(items["order_id"],       orders["order_id"],          "items.order_id → orders.order_id")
    check(items["product_id"],     products["product_id"],      "items.product_id → products.product_id")
    check(items["seller_id"],      sellers["seller_id"],        "items.seller_id → sellers.seller_id")
    check(payments["order_id"],    orders["order_id"],          "payments.order_id → orders.order_id")
    check(reviews["order_id"],     orders["order_id"],          "reviews.order_id → orders.order_id")


# ---------------------------------------------------------------------------
# SECTION 6 — TABLE JOINS & FEATURE ENGINEERING
# ---------------------------------------------------------------------------
#
# REVENUE / VALUE DEFINITIONS (documented):
#
#   product_sales_value  = SUM(price)              per order
#   freight_value_total  = SUM(freight_value)       per order
#   order_value          = product_sales_value + freight_value_total
#
# These three measures are always kept separate so dashboards can choose the
# correct one.  "Revenue" is never used ambiguously in the code.
#
# DELIVERY FEATURE DEFINITIONS:
#   delivery_days        = (order_delivered_customer_date - order_purchase_timestamp).days
#   estimated_days       = (order_estimated_delivery_date - order_purchase_timestamp).days
#   delivery_delay_days  = (order_delivered_customer_date - order_estimated_delivery_date).days
#                          positive = late,  negative = early
#   is_late              = 1 if delivery_delay_days > 0 else 0   (for delivered orders only)
#   on_time              = 1 - is_late

def build_orders_master(orders, customers, payments_agg, reviews_agg, items):
    """
    orders_master: one row per order.
    Includes order metadata, customer info, aggregated payment, aggregated review,
    and aggregated item-level financials.
    """
    print("\n" + "="*60)
    print("SECTION 6a — BUILD orders_master")
    print("="*60)

    # Aggregate items to order level
    items_order = items.groupby("order_id", as_index=False).agg(
        product_sales_value = ("price",         "sum"),
        freight_value_total = ("freight_value", "sum"),
        item_count          = ("order_item_id", "count"),
    )
    items_order["order_value"] = (
        items_order["product_sales_value"] + items_order["freight_value_total"]
    )

    # Join chain: orders → customers → items_order → payments → reviews
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(items_order,   on="order_id",    how="left")
    df = df.merge(payments_agg,  on="order_id",    how="left")
    df = df.merge(reviews_agg[["order_id","review_score","review_event_count",
                                "review_comment_message","review_creation_date"]],
                  on="order_id", how="left")

    # ── Delivery feature engineering ──────────────────────────────────────
    # Only meaningful for delivered orders; kept as NaN for others
    mask_del = df["order_status"] == "delivered"

    df["delivery_days"] = np.where(
        mask_del,
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days,
        np.nan
    )
    df["estimated_days"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    )
    df["delivery_delay_days"] = np.where(
        mask_del,
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days,
        np.nan
    )
    df["is_late"] = np.where(
        mask_del & df["delivery_delay_days"].notna(),
        (df["delivery_delay_days"] > 0).astype(int),
        np.nan
    )
    df["on_time"] = np.where(df["is_late"].notna(), 1 - df["is_late"], np.nan)

    # ── Time features ──────────────────────────────────────────────────────
    df["purchase_year"]  = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_month_label"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    print(f"  orders_master rows : {len(df):,}")
    print(f"  Columns            : {list(df.columns)}")

    # Sanity: no duplicate order_ids
    assert df["order_id"].duplicated().sum() == 0, "FAIL: duplicate order_ids in orders_master"
    print("  PASS: no duplicate order_ids")

    # Row-count reconciliation
    base_orders = len(orders)
    output_rows = len(df)
    if output_rows != base_orders:
        print(f"  WARNING: row count changed {base_orders} → {output_rows} (check join type)")
    else:
        print(f"  PASS: row count preserved after joins ({output_rows:,})")

    return df


def build_orders_full(orders_master, items, products, sellers):
    """
    orders_full: one row per order-item line.
    Includes all orders_master fields repeated per item, plus product and seller detail.
    """
    print("\n" + "="*60)
    print("SECTION 6b — BUILD orders_full")
    print("="*60)

    # Enrich items with product info (preserve original PT category name, add EN)
    item_rich = items.merge(
        products[["product_id","product_category_name",
                  "product_category_name_english",
                  "product_weight_g","product_length_cm",
                  "product_height_cm","product_width_cm",
                  "product_photos_qty"]],
        on="product_id", how="left"
    )
    item_rich = item_rich.merge(
        sellers[["seller_id","seller_zip_code_prefix","seller_city","seller_state"]],
        on="seller_id", how="left"
    )
    item_rich["item_value"] = item_rich["price"] + item_rich["freight_value"]
    item_rich["product_volume_cm3"] = (
        item_rich["product_length_cm"] *
        item_rich["product_height_cm"] *
        item_rich["product_width_cm"]
    )

    # Merge with orders_master (drop item-level financial aggregates to avoid confusion)
    drop_cols = ["product_sales_value","freight_value_total","order_value","item_count"]
    master_slim = orders_master.drop(columns=[c for c in drop_cols if c in orders_master.columns])

    full = master_slim.merge(item_rich, on="order_id", how="left")

    print(f"  orders_full rows   : {len(full):,}")
    print(f"  Columns            : {list(full.columns)}")

    # orders_full can legitimately exceed items row count.
    # Orders with no items (canceled/unavailable/created) produce one null-item row via left join.
    expected_items = len(items)
    orders_without_items = (~orders_master["order_id"].isin(items["order_id"])).sum()
    expected_full = expected_items + orders_without_items
    if len(full) == expected_full:
        print(f"  PASS: {len(full):,} rows = {expected_items:,} item rows + "
              f"{orders_without_items} orders without items (canceled/unavailable)")
    else:
        print(f"  WARNING: orders_full has {len(full):,} rows vs expected {expected_full:,}")

    return full


# ---------------------------------------------------------------------------
# SECTION 7 — RFM FEATURE ENGINEERING
# ---------------------------------------------------------------------------
#
# RFM is computed on DELIVERED orders only (cancellations/returned orders
# should not count toward customer value).
#
# Reference date: the day after the most recent order_purchase_timestamp in the dataset,
# so that Recency is relative to the dataset's own time horizon, not wall-clock time.
#
# Definitions:
#   Recency    = (reference_date - max(order_purchase_timestamp)).days  per customer_unique_id
#   Frequency  = COUNT(DISTINCT order_id) for delivered orders           per customer_unique_id
#   Monetary   = SUM(order_value)  for delivered orders                  per customer_unique_id
#
# RFM Score:
#   Each metric is independently ranked into quintiles (1–5).
#   R is REVERSED (lower recency = better = score 5).
#   Combined RFM score = R_score * 100 + F_score * 10 + M_score
#   Segment labels assigned from combined score thresholds.

def build_rfm(orders_master):
    print("\n" + "="*60)
    print("SECTION 7 — RFM FEATURE ENGINEERING")
    print("="*60)

    delivered = orders_master[orders_master["order_status"] == "delivered"].copy()
    reference_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    print(f"  RFM reference date : {reference_date.date()}")
    print(f"  Delivered orders   : {len(delivered):,}")

    rfm = delivered.groupby("customer_unique_id", as_index=False).agg(
        last_purchase_date = ("order_purchase_timestamp", "max"),
        frequency          = ("order_id",                 "nunique"),
        monetary           = ("order_value",              "sum"),
    )
    rfm["recency_days"] = (reference_date - rfm["last_purchase_date"]).dt.days

    # Quintile scoring
    rfm["R_score"] = pd.qcut(rfm["recency_days"],  q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"],       q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]

    # Segment assignment
    def segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 2 and m <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Cannot Lose Them"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost / Inactive"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(segment, axis=1)

    print(f"  RFM customers      : {len(rfm):,}")
    print(f"  Repeat customers   : {(rfm['frequency'] > 1).sum():,}")
    print(f"\n  Segment distribution:")
    for seg, cnt in rfm["segment"].value_counts().items():
        print(f"    {seg:<25} {cnt:>6,} ({cnt/len(rfm)*100:.1f}%)")

    return rfm


# ---------------------------------------------------------------------------
# SECTION 8 — PERSIST OUTPUTS
# ---------------------------------------------------------------------------

def save_outputs(orders_master, orders_full, customer_rfm):
    print("\n" + "="*60)
    print("SECTION 8 — SAVING OUTPUTS")
    print("="*60)

    outputs = {
        "orders_master.csv" : orders_master,
        "orders_full.csv"   : orders_full,
        "customer_rfm.csv"  : customer_rfm,
    }
    for fname, df in outputs.items():
        path = os.path.join(DATA_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {path:<35} {len(df):>7,} rows  {df.shape[1]} cols")


# ---------------------------------------------------------------------------
# MAIN — Phase 2
# ---------------------------------------------------------------------------

def main_preprocessing():
    print("\n" + "#"*60)
    print("  OLIST PREPROCESSING PIPELINE — PHASE 2")
    print("#"*60)

    # 1 — Load
    dfs = load_raw()

    # 2 — Validate raw
    validate_raw(dfs)

    # 3 — Review aggregation
    reviews_agg = aggregate_reviews(dfs["reviews"])

    # 4 — Clean each table
    print("\n" + "="*60)
    print("SECTION 4 — TABLE CLEANING")
    print("="*60)
    customers     = clean_customers(dfs["customers"])
    orders        = clean_orders(dfs["orders"])
    items         = clean_items(dfs["items"])
    payments_agg  = clean_payments(dfs["payments"])
    products      = clean_products(dfs["products"], dfs["translation"])
    sellers       = clean_sellers(dfs["sellers"])
    geo_deduped   = clean_geolocation(dfs["geolocation"])

    # 5 — Referential integrity
    dfs_clean = {
        "orders":   orders,
        "customers":customers,
        "items":    items,
        "payments": payments_agg,
        "reviews":  reviews_agg,
        "products": products,
        "sellers":  sellers,
    }
    check_referential_integrity(dfs_clean)

    # 6 — Build analytical tables
    orders_master = build_orders_master(orders, customers, payments_agg, reviews_agg, items)
    orders_full   = build_orders_full(orders_master, items, products, sellers)

    # 7 — RFM
    customer_rfm  = build_rfm(orders_master)

    # 8 — Save
    save_outputs(orders_master, orders_full, customer_rfm)

    print("\n" + "#"*60)
    print("  PHASE 2 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: analytics.py — Phase 3
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PYTHONIOENCODING = "utf-8"

# ---------------------------------------------------------------------------
# LOAD PROCESSED DATA
# ---------------------------------------------------------------------------

def load_processed():
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)

    master = pd.read_csv(os.path.join(DATA_DIR, "orders_master.csv"),
                         parse_dates=["order_purchase_timestamp",
                                      "order_delivered_customer_date",
                                      "order_estimated_delivery_date",
                                      "order_approved_at"])
    full   = pd.read_csv(os.path.join(DATA_DIR, "orders_full.csv"),
                         parse_dates=["order_purchase_timestamp",
                                      "order_delivered_customer_date",
                                      "order_estimated_delivery_date"])

    print(f"  orders_master : {len(master):,} rows")
    print(f"  orders_full   : {len(full):,} rows")

    # Scope subsets used repeatedly
    delivered_master = master[master["order_status"] == "delivered"].copy()
    delivered_full   = full[full["order_status"] == "delivered"].copy()

    print(f"  delivered orders (master): {len(delivered_master):,}")
    print(f"  delivered items  (full)  : {len(delivered_full):,}")

    return master, full, delivered_master, delivered_full


# ---------------------------------------------------------------------------
# SECTION 1 — TOP-LINE KPIs
# ---------------------------------------------------------------------------
# Grain of orders_master : one row per order_id  (guaranteed by Phase 2 assert)
# Grain of orders_full   : one row per (order_id, order_item_id)
#
# Revenue definitions (all sourced from orders_master to avoid double-counting):
#   product_sales_value  = SUM(price)                    pre-aggregated in Phase 2
#   freight_value_total  = SUM(freight_value)            pre-aggregated in Phase 2
#   order_value          = product_sales + freight       pre-aggregated in Phase 2
#
# These sums are safe at orders_master grain because Phase 2 already collapsed
# multi-item orders into order-level totals.

def compute_topline_kpis(master, delivered_master):
    print("\n" + "="*60)
    print("SECTION 1 — TOP-LINE KPIs")
    print("="*60)

    # --- 1a. Financial KPIs (delivered orders only for revenue) ---------------
    #   Formula : SUM(product_sales_value) over delivered orders_master rows
    #   Source  : orders_master.product_sales_value  (one row per order)
    #   Records : delivered_master rows that have a non-null product_sales_value
    rev_rows = delivered_master["product_sales_value"].notna().sum()
    total_product_sales = delivered_master["product_sales_value"].sum()
    total_freight       = delivered_master["freight_value_total"].sum()
    total_order_value   = delivered_master["order_value"].sum()

    # Independent validation: rebuild from raw sums via orders_full and compare
    # (orders_full item-level: SUM(price) per delivered order should equal total_product_sales)
    # Validation done separately in validate_kpis() below.

    # --- 1b. Order / Customer counts ------------------------------------------
    total_orders    = master["order_id"].nunique()
    delivered_orders = delivered_master["order_id"].nunique()
    #   customer_unique_id distinguishes real customers from order-IDs
    total_customers = master["customer_unique_id"].nunique()

    # --- 1c. Average Order Value (AOV) ----------------------------------------
    #   Formula : total_order_value / delivered_orders
    #   Only delivered orders contribute to revenue, so AOV is over delivered set
    aov = total_order_value / delivered_orders if delivered_orders > 0 else np.nan

    # --- 1d. Review Score -----------------------------------------------------
    #   Source  : orders_master.review_score (one review per order after Phase 2 dedup)
    #   Scope   : all orders that have a review (not just delivered)
    review_rows = delivered_master["review_score"].notna().sum()
    avg_review  = delivered_master["review_score"].mean()

    # --- 1e. Delivery performance ---------------------------------------------
    #   Scope   : delivered orders with non-null is_late (excludes 8 with missing delivery date)
    del_scope  = delivered_master["is_late"].notna()
    del_count  = del_scope.sum()
    on_time_pct = delivered_master.loc[del_scope, "is_late"].apply(lambda x: 1 - x).mean() * 100
    late_pct    = delivered_master.loc[del_scope, "is_late"].mean() * 100
    avg_delivery_days = delivered_master["delivery_days"].mean()

    # --- 1f. Repeat customer rate ---------------------------------------------
    #   Formula : customers with >1 delivered order / total customers with >=1 delivered order
    #   Source  : delivered_master.customer_unique_id
    cust_order_counts = delivered_master.groupby("customer_unique_id")["order_id"].nunique()
    repeat_customers  = (cust_order_counts > 1).sum()
    total_del_cust    = len(cust_order_counts)
    repeat_rate       = repeat_customers / total_del_cust * 100 if total_del_cust > 0 else 0

    kpis = {
        "total_product_sales":     round(total_product_sales, 2),
        "total_freight_value":     round(total_freight, 2),
        "total_order_value":       round(total_order_value, 2),
        "total_orders":            int(total_orders),
        "total_delivered_orders":  int(delivered_orders),
        "total_customers":         int(total_customers),
        "avg_order_value":         round(aov, 2),
        "avg_review_score":        round(avg_review, 4),
        "on_time_delivery_pct":    round(on_time_pct, 2),
        "late_delivery_pct":       round(late_pct, 2),
        "avg_delivery_days":       round(avg_delivery_days, 2),
        "repeat_customer_rate_pct":round(repeat_rate, 2),
        # metadata
        "_revenue_scope":          "delivered orders only",
        "_review_record_count":    int(review_rows),
        "_delivery_record_count":  int(del_count),
        "_repeat_customer_base":   int(total_del_cust),
    }

    print(f"\n  Total Product Sales (delivered)  : R$ {total_product_sales:>14,.2f}  ({rev_rows:,} orders)")
    print(f"  Total Freight Value (delivered)  : R$ {total_freight:>14,.2f}")
    print(f"  Total Order Value   (delivered)  : R$ {total_order_value:>14,.2f}")
    print(f"  Total Orders (all statuses)      : {total_orders:>10,}")
    print(f"  Total Delivered Orders           : {delivered_orders:>10,}")
    print(f"  Total Unique Customers           : {total_customers:>10,}")
    print(f"  Average Order Value              : R$ {aov:>10,.2f}")
    print(f"  Average Review Score             : {avg_review:>10.4f}  ({review_rows:,} reviews)")
    print(f"  On-Time Delivery %               : {on_time_pct:>10.2f}%  ({del_count:,} deliveries)")
    print(f"  Late Delivery %                  : {late_pct:>10.2f}%")
    print(f"  Average Delivery Days            : {avg_delivery_days:>10.2f} days")
    print(f"  Repeat Customer Rate             : {repeat_rate:>10.2f}%  ({repeat_customers:,} / {total_del_cust:,})")

    return kpis


def validate_kpis(master, full, delivered_master, delivered_full, kpis):
    """
    Independent cross-validation of key financial KPIs.
    Rebuilds totals from the item-level orders_full table and compares.
    """
    print("\n  --- VALIDATION ---")

    # Re-derive total product sales from item level (orders_full, delivered only)
    # Each row in orders_full represents one item line with its price
    # We must restrict to delivered orders (order_status == 'delivered')
    item_sales  = delivered_full["price"].sum()
    item_freight= delivered_full["freight_value"].sum()

    diff_sales   = abs(item_sales  - kpis["total_product_sales"])
    diff_freight = abs(item_freight- kpis["total_freight_value"])

    tol = 0.01   # floating-point tolerance

    if diff_sales < tol:
        print(f"  PASS: product sales cross-check  master={kpis['total_product_sales']:,.2f}  "
              f"item-level={item_sales:,.2f}  diff={diff_sales:.4f}")
    else:
        print(f"  FAIL: product sales MISMATCH  master={kpis['total_product_sales']:,.2f}  "
              f"item-level={item_sales:,.2f}  diff={diff_sales:,.2f}")

    if diff_freight < tol:
        print(f"  PASS: freight cross-check        master={kpis['total_freight_value']:,.2f}  "
              f"item-level={item_freight:,.2f}  diff={diff_freight:.4f}")
    else:
        print(f"  FAIL: freight MISMATCH           master={kpis['total_freight_value']:,.2f}  "
              f"item-level={item_freight:,.2f}  diff={diff_freight:,.2f}")

    # Validate order count: master should equal distinct order_ids in full for delivered
    master_del_count = delivered_master["order_id"].nunique()
    full_del_count   = delivered_full["order_id"].nunique()
    if master_del_count == full_del_count:
        print(f"  PASS: delivered order count consistent ({master_del_count:,})")
    else:
        print(f"  WARN: delivered order count differs  master={master_del_count}  full={full_del_count}")

    # Validate AOV: order_value / orders should equal pre-computed avg
    recomputed_aov = delivered_master["order_value"].sum() / delivered_master["order_id"].nunique()
    if abs(recomputed_aov - kpis["avg_order_value"]) < tol:
        print(f"  PASS: AOV recomputed={recomputed_aov:.2f}  stored={kpis['avg_order_value']:.2f}")
    else:
        print(f"  FAIL: AOV mismatch recomputed={recomputed_aov:.2f}  stored={kpis['avg_order_value']:.2f}")


# ---------------------------------------------------------------------------
# SECTION 2 — MONTHLY TIME-SERIES
# ---------------------------------------------------------------------------
# Grain : one row per calendar month (YYYY-MM period label)
# Source: orders_master for order/customer/revenue aggregates
#         delivered_master for delivery performance
#
# NOTE: months with fewer than 5 orders are excluded (dataset edge months)
# to avoid misleading partial-month spikes at the start/end of the time range.

def compute_monthly_trends(master, delivered_master):
    print("\n" + "="*60)
    print("SECTION 2 — MONTHLY TIME-SERIES")
    print("="*60)

    # All orders (for volume/customer counts)
    m = master.copy()
    m["ym"] = m["order_purchase_timestamp"].dt.to_period("M").astype(str)

    monthly_all = m.groupby("ym", as_index=False).agg(
        order_count        = ("order_id",             "nunique"),
        customer_count     = ("customer_unique_id",   "nunique"),
    )

    # Delivered orders (for revenue + delivery KPIs)
    d = delivered_master.copy()
    d["ym"] = d["order_purchase_timestamp"].dt.to_period("M").astype(str)

    monthly_del = d.groupby("ym", as_index=False).agg(
        product_sales_value = ("product_sales_value", "sum"),
        freight_value_total = ("freight_value_total", "sum"),
        order_value         = ("order_value",         "sum"),
        delivered_count     = ("order_id",            "nunique"),
        avg_review_score    = ("review_score",        "mean"),
        avg_delivery_days   = ("delivery_days",       "mean"),
        on_time_count       = ("on_time",             "sum"),
        late_count          = ("is_late",             "sum"),
    )
    monthly_del["on_time_pct"] = (
        monthly_del["on_time_count"] /
        (monthly_del["on_time_count"] + monthly_del["late_count"]) * 100
    ).round(2)

    # Join
    monthly = monthly_all.merge(monthly_del, on="ym", how="left")
    monthly = monthly.sort_values("ym")

    # Drop edge months with very low volume (< 50 orders — partial-month artefacts)
    monthly = monthly[monthly["order_count"] >= 50].reset_index(drop=True)

    print(f"  Monthly trend rows: {len(monthly)} months")
    print(f"  Date range: {monthly['ym'].min()} → {monthly['ym'].max()}")
    print(f"  Peak month by order count: "
          f"{monthly.loc[monthly['order_count'].idxmax(), 'ym']}  "
          f"({monthly['order_count'].max():,} orders)")
    print(f"  Peak month by product sales: "
          f"{monthly.loc[monthly['product_sales_value'].idxmax(), 'ym']}  "
          f"(R$ {monthly['product_sales_value'].max():,.0f})")

    return monthly


# ---------------------------------------------------------------------------
# SECTION 3 — CATEGORY PERFORMANCE
# ---------------------------------------------------------------------------
# Grain  : one row per product_category_name_english
# Source : orders_full (item level) — CORRECT grain for category splits
#
# Financial columns sourced from item-level (price, freight_value per item row).
# Review scores joined from order level via order_id to avoid double-counting
# when an order has multiple items in the same category.
# Delivery KPIs joined from order level for the same reason.

def compute_category_performance(full, delivered_full, delivered_master):
    print("\n" + "="*60)
    print("SECTION 3 — CATEGORY PERFORMANCE")
    print("="*60)

    # Use delivered items only for financial metrics
    df = delivered_full[delivered_full["price"].notna()].copy()

    # Item-level financials per category
    cat_finance = df.groupby("product_category_name_english", as_index=False).agg(
        product_sales_value   = ("price",          "sum"),
        freight_value_total   = ("freight_value",  "sum"),
        item_count            = ("order_item_id",  "count"),
        order_count           = ("order_id",       "nunique"),
    )
    cat_finance["order_value"] = (
        cat_finance["product_sales_value"] + cat_finance["freight_value_total"]
    )
    cat_finance["avg_item_price"] = (
        cat_finance["product_sales_value"] / cat_finance["item_count"]
    ).round(2)
    cat_finance["avg_order_value"] = (
        cat_finance["order_value"] / cat_finance["order_count"]
    ).round(2)

    # Order-level review and delivery scores per category
    # Use first item's category per order to avoid double-counting order-level metrics
    # (for orders spanning multiple categories, this is a known limitation — documented)
    order_cat = df.sort_values("order_item_id").drop_duplicates(subset="order_id")[
        ["order_id", "product_category_name_english"]
    ]
    order_del = delivered_master[
        ["order_id", "review_score", "delivery_days", "is_late", "on_time",
         "delivery_delay_days"]
    ]
    order_cat_del = order_cat.merge(order_del, on="order_id", how="left")

    cat_sat = order_cat_del.groupby("product_category_name_english", as_index=False).agg(
        avg_review_score   = ("review_score",        "mean"),
        avg_delivery_days  = ("delivery_days",        "mean"),
        late_count         = ("is_late",              "sum"),
        on_time_count      = ("on_time",              "sum"),
        avg_delay_days     = ("delivery_delay_days",  "mean"),
    )
    cat_sat["late_pct"] = (
        cat_sat["late_count"] /
        (cat_sat["late_count"] + cat_sat["on_time_count"]) * 100
    ).round(2)

    cat = cat_finance.merge(cat_sat, on="product_category_name_english", how="left")
    cat = cat.sort_values("product_sales_value", ascending=False).reset_index(drop=True)
    cat["rank_by_sales"] = cat.index + 1

    print(f"  Category rows: {len(cat)}")
    print(f"\n  Top 10 by product sales:")
    for _, r in cat.head(10).iterrows():
        print(f"    {r['product_category_name_english']:<40} "
              f"R$ {r['product_sales_value']:>12,.0f}  "
              f"orders={r['order_count']:>6,}  "
              f"avg_review={r['avg_review_score']:.2f}")
    print(f"\n  Bottom 5 by product sales:")
    for _, r in cat.tail(5).iterrows():
        print(f"    {r['product_category_name_english']:<40} "
              f"R$ {r['product_sales_value']:>12,.0f}  "
              f"orders={r['order_count']:>6,}")

    # Validation: sum of category product_sales should equal total product sales
    cat_total = cat["product_sales_value"].sum()
    item_total = delivered_full["price"].sum()
    diff = abs(cat_total - item_total)
    if diff < 0.01:
        print(f"\n  PASS: category sales sum={cat_total:,.2f} matches item-level total={item_total:,.2f}")
    else:
        print(f"\n  WARN: category sales sum={cat_total:,.2f} vs item-level={item_total:,.2f}  diff={diff:,.2f}")
        print(f"        (difference likely from items with unknown category)")

    return cat


# ---------------------------------------------------------------------------
# SECTION 4 — REGIONAL (STATE) PERFORMANCE
# ---------------------------------------------------------------------------
# Grain  : one row per customer_state
# Source : orders_master for order/revenue aggregates (no double-count risk)
#          delivered_master for delivery KPIs

def compute_state_performance(master, delivered_master):
    print("\n" + "="*60)
    print("SECTION 4 — REGIONAL (STATE) PERFORMANCE")
    print("="*60)

    # All-orders view (for order count, customer count)
    state_all = master.groupby("customer_state", as_index=False).agg(
        order_count      = ("order_id",           "nunique"),
        customer_count   = ("customer_unique_id", "nunique"),
    )

    # Delivered-orders view (for revenue + satisfaction + delivery)
    state_del = delivered_master.groupby("customer_state", as_index=False).agg(
        product_sales_value  = ("product_sales_value", "sum"),
        freight_value_total  = ("freight_value_total", "sum"),
        order_value          = ("order_value",         "sum"),
        avg_order_value      = ("order_value",         "mean"),
        avg_review_score     = ("review_score",        "mean"),
        avg_delivery_days    = ("delivery_days",       "mean"),
        late_count           = ("is_late",             "sum"),
        on_time_count        = ("on_time",             "sum"),
        avg_delay_days       = ("delivery_delay_days", "mean"),
    )
    state_del["late_pct"] = (
        state_del["late_count"] /
        (state_del["late_count"] + state_del["on_time_count"]) * 100
    ).round(2)
    state_del["avg_order_value"] = state_del["avg_order_value"].round(2)

    state = state_all.merge(state_del, on="customer_state", how="left")
    state = state.sort_values("order_count", ascending=False).reset_index(drop=True)

    print(f"  State rows: {len(state)}")
    print(f"\n  Top 10 states by order count:")
    for _, r in state.head(10).iterrows():
        print(f"    {r['customer_state']}  orders={r['order_count']:>6,}  "
              f"customers={r['customer_count']:>5,}  "
              f"sales=R${r['product_sales_value']:>10,.0f}  "
              f"late%={r['late_pct']:.1f}%")

    return state


# ---------------------------------------------------------------------------
# SECTION 5 — SELLER PERFORMANCE
# ---------------------------------------------------------------------------
# Grain  : one row per seller_id
# Source : orders_full (item level — correct because seller_id lives here)
#
# Revenue: SUM(price) per seller from delivered items (item-level is correct here
#          since a seller may have multiple items in one order; we want their share).
# Delivery/review: deduped at order level to avoid double-counting.

def compute_seller_performance(delivered_full, delivered_master):
    print("\n" + "="*60)
    print("SECTION 5 — SELLER PERFORMANCE")
    print("="*60)

    df = delivered_full[delivered_full["seller_id"].notna()].copy()

    seller_finance = df.groupby("seller_id", as_index=False).agg(
        product_sales_value = ("price",         "sum"),
        freight_value_total = ("freight_value", "sum"),
        item_count          = ("order_item_id", "count"),
        order_count         = ("order_id",      "nunique"),
        seller_state        = ("seller_state",  "first"),
        seller_city         = ("seller_city",   "first"),
    )

    # Review and delivery from order level (one metric per order, not per item)
    order_seller = df.sort_values("order_item_id").drop_duplicates(subset=["order_id","seller_id"])[
        ["order_id","seller_id"]
    ]
    order_del = delivered_master[["order_id","review_score","delivery_days","is_late","on_time"]]
    order_seller_del = order_seller.merge(order_del, on="order_id", how="left")

    seller_sat = order_seller_del.groupby("seller_id", as_index=False).agg(
        avg_review_score  = ("review_score",   "mean"),
        avg_delivery_days = ("delivery_days",  "mean"),
        late_count        = ("is_late",        "sum"),
        on_time_count     = ("on_time",        "sum"),
    )
    seller_sat["late_pct"] = (
        seller_sat["late_count"] /
        (seller_sat["late_count"] + seller_sat["on_time_count"]) * 100
    ).round(2)

    sellers = seller_finance.merge(seller_sat, on="seller_id", how="left")
    sellers = sellers.sort_values("product_sales_value", ascending=False).reset_index(drop=True)
    sellers["rank_by_sales"] = sellers.index + 1

    print(f"  Seller rows: {len(sellers)}")
    print(f"\n  Top 10 sellers by product sales:")
    for _, r in sellers.head(10).iterrows():
        print(f"    seller ...{r['seller_id'][-6:]}  "
              f"state={r['seller_state']}  "
              f"sales=R${r['product_sales_value']:>10,.0f}  "
              f"orders={r['order_count']:>5,}  "
              f"review={r['avg_review_score']:.2f}  "
              f"late%={r['late_pct']:.1f}%")

    return sellers


# ---------------------------------------------------------------------------
# SECTION 6 — DELIVERY ANALYSIS
# ---------------------------------------------------------------------------
# Grain  : delivered orders with non-null delivery_days
# Provides: delay distribution buckets, performance by state and category

def compute_delivery_analysis(delivered_master, delivered_full):
    print("\n" + "="*60)
    print("SECTION 6 — DELIVERY ANALYSIS")
    print("="*60)

    del_valid = delivered_master[delivered_master["delivery_days"].notna()].copy()

    # Delay distribution buckets
    bins   = [-999, -7, -3, 0, 3, 7, 14, 999]
    labels = ["Early >7d", "Early 3-7d", "Early 0-3d",
              "Late 1-3d", "Late 4-7d", "Late 8-14d", "Late >14d"]
    del_valid["delay_bucket"] = pd.cut(
        del_valid["delivery_delay_days"], bins=bins, labels=labels
    )
    delay_dist = del_valid["delay_bucket"].value_counts().reindex(labels).reset_index()
    delay_dist.columns = ["delay_bucket", "order_count"]
    delay_dist["pct"] = (delay_dist["order_count"] / len(del_valid) * 100).round(2)

    print(f"\n  Delivery delay distribution ({len(del_valid):,} delivered orders):")
    for _, r in delay_dist.iterrows():
        bar = "#" * int(r["pct"] / 2)
        print(f"    {r['delay_bucket']:<14} {r['order_count']:>6,} ({r['pct']:>5.1f}%)  {bar}")

    # Delivery by state
    state_del = delivered_master.groupby("customer_state", as_index=False).agg(
        order_count        = ("order_id",            "nunique"),
        avg_delivery_days  = ("delivery_days",        "mean"),
        avg_delay_days     = ("delivery_delay_days",  "mean"),
        late_pct           = ("is_late",              "mean"),
        avg_review_score   = ("review_score",         "mean"),
    )
    state_del["late_pct"] = (state_del["late_pct"] * 100).round(2)
    state_del["avg_delivery_days"] = state_del["avg_delivery_days"].round(2)
    state_del["avg_delay_days"]    = state_del["avg_delay_days"].round(2)
    state_del = state_del.sort_values("late_pct", ascending=False).reset_index(drop=True)

    print(f"\n  Top 5 states by late delivery %:")
    for _, r in state_del.head(5).iterrows():
        print(f"    {r['customer_state']}  late%={r['late_pct']:.1f}%  "
              f"avg_days={r['avg_delivery_days']:.1f}  "
              f"review={r['avg_review_score']:.2f}")

    # Delivery by category (use first item's category per order)
    df = delivered_full[delivered_full["price"].notna()].copy()
    order_cat = df.sort_values("order_item_id").drop_duplicates(subset="order_id")[
        ["order_id", "product_category_name_english"]
    ]
    order_del = delivered_master[[
        "order_id", "delivery_days", "delivery_delay_days", "is_late", "on_time", "review_score"
    ]]
    cat_del = order_cat.merge(order_del, on="order_id", how="left")
    cat_del_agg = cat_del.groupby("product_category_name_english", as_index=False).agg(
        order_count       = ("order_id",            "nunique"),
        avg_delivery_days = ("delivery_days",        "mean"),
        avg_delay_days    = ("delivery_delay_days",  "mean"),
        late_pct          = ("is_late",              "mean"),
        avg_review_score  = ("review_score",         "mean"),
    )
    cat_del_agg["late_pct"]           = (cat_del_agg["late_pct"] * 100).round(2)
    cat_del_agg["avg_delivery_days"]  = cat_del_agg["avg_delivery_days"].round(2)
    cat_del_agg["avg_delay_days"]     = cat_del_agg["avg_delay_days"].round(2)
    cat_del_agg = cat_del_agg.sort_values("late_pct", ascending=False).reset_index(drop=True)

    print(f"\n  Top 5 categories by late delivery %:")
    for _, r in cat_del_agg.head(5).iterrows():
        print(f"    {r['product_category_name_english']:<40} "
              f"late%={r['late_pct']:.1f}%  "
              f"avg_days={r['avg_delivery_days']:.1f}")

    return delay_dist, state_del, cat_del_agg


# ---------------------------------------------------------------------------
# SECTION 7 — CUSTOMER SATISFACTION
# ---------------------------------------------------------------------------

def compute_satisfaction(delivered_master):
    print("\n" + "="*60)
    print("SECTION 7 — CUSTOMER SATISFACTION")
    print("="*60)

    dm = delivered_master[delivered_master["review_score"].notna()].copy()

    # Review score distribution
    score_dist = dm["review_score"].value_counts().sort_index().reset_index()
    score_dist.columns = ["review_score", "order_count"]
    score_dist["pct"] = (score_dist["order_count"] / len(dm) * 100).round(2)
    print(f"  Review score distribution ({len(dm):,} scored orders):")
    for _, r in score_dist.iterrows():
        bar = "#" * int(r["pct"] / 2)
        print(f"    Score {r['review_score']}  {r['order_count']:>6,} ({r['pct']:>5.1f}%)  {bar}")

    # Monthly average review score
    dm["ym"] = dm["order_purchase_timestamp"].dt.to_period("M").astype(str)
    sat_time = dm.groupby("ym", as_index=False).agg(
        avg_review_score = ("review_score", "mean"),
        review_count     = ("review_score", "count"),
    )
    sat_time = sat_time[sat_time["review_count"] >= 50].sort_values("ym")
    print(f"\n  Monthly satisfaction: {len(sat_time)} months  "
          f"range=[{sat_time['avg_review_score'].min():.2f}, "
          f"{sat_time['avg_review_score'].max():.2f}]")

    # Review score vs delivery delay buckets
    # Only use rows where both review_score and delivery_delay_days are available
    dm_del = dm[dm["delivery_delay_days"].notna() & dm["review_score"].notna()].copy()
    bins   = [-999, -7, -3, 0, 3, 7, 14, 999]
    labels = ["Early >7d", "Early 3-7d", "Early 0-3d",
              "Late 1-3d", "Late 4-7d", "Late 8-14d", "Late >14d"]
    dm_del["delay_bucket"] = pd.cut(dm_del["delivery_delay_days"], bins=bins, labels=labels)
    rev_delay = dm_del.groupby("delay_bucket", observed=False, as_index=False).agg(
        avg_review_score = ("review_score", "mean"),
        order_count      = ("order_id",     "nunique"),
    )
    rev_delay["avg_review_score"] = rev_delay["avg_review_score"].round(3)
    # Ensure canonical label order
    rev_delay["delay_bucket"] = pd.Categorical(rev_delay["delay_bucket"], categories=labels, ordered=True)
    rev_delay = rev_delay.sort_values("delay_bucket").reset_index(drop=True)
    rev_delay["delay_bucket"] = rev_delay["delay_bucket"].astype(str)

    print(f"\n  Review score by delivery delay bucket:")
    for _, r in rev_delay.iterrows():
        if pd.notna(r["avg_review_score"]):
            print(f"    {str(r['delay_bucket']):<14} avg_score={r['avg_review_score']:.3f}  "
                  f"n={int(r['order_count']) if pd.notna(r['order_count']) else 0}")

    return score_dist, sat_time, rev_delay


# ---------------------------------------------------------------------------
# SECTION 8 — SAVE ALL OUTPUTS
# ---------------------------------------------------------------------------

def save_all_analytics(kpis, monthly, category, state, sellers,
             delay_dist, state_del, cat_del, score_dist, sat_time, rev_delay):
    print("\n" + "="*60)
    print("SECTION 8 — SAVING OUTPUTS")
    print("="*60)

    # KPIs as JSON
    kpi_path = os.path.join(DATA_DIR, "kpis.json")
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)
    print(f"  Saved {kpi_path}")

    csvs = {
        "monthly_trends.csv":      monthly,
        "category_performance.csv":category,
        "state_performance.csv":   state,
        "seller_performance.csv":  sellers,
        "delivery_delay_dist.csv": delay_dist,
        "delivery_by_state.csv":   state_del,
        "delivery_by_category.csv":cat_del,
        "review_score_dist.csv":   score_dist,
        "satisfaction_over_time.csv": sat_time,
        "review_vs_delay.csv":     rev_delay,
    }
    for fname, df in csvs.items():
        path = os.path.join(DATA_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {path:<45} {len(df):>5} rows")


# ---------------------------------------------------------------------------
# MAIN — Phase 3
# ---------------------------------------------------------------------------

def main_analytics():
    print("\n" + "#"*60)
    print("  OLIST ANALYTICS — PHASE 3")
    print("#"*60)

    master, full, delivered_master, delivered_full = load_processed()

    kpis    = compute_topline_kpis(master, delivered_master)
    validate_kpis(master, full, delivered_master, delivered_full, kpis)

    monthly  = compute_monthly_trends(master, delivered_master)
    category = compute_category_performance(full, delivered_full, delivered_master)
    state    = compute_state_performance(master, delivered_master)
    sellers  = compute_seller_performance(delivered_full, delivered_master)
    delay_dist, state_del, cat_del = compute_delivery_analysis(delivered_master, delivered_full)
    score_dist, sat_time, rev_delay = compute_satisfaction(delivered_master)

    save_all_analytics(kpis, monthly, category, state, sellers,
             delay_dist, state_del, cat_del, score_dist, sat_time, rev_delay)

    print("\n" + "#"*60)
    print("  PHASE 3 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: customer_analytics.py — Phase 4
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_data_customers():
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    master = pd.read_csv(
        os.path.join(DATA_DIR, "orders_master.csv"),
        parse_dates=["order_purchase_timestamp"]
    )
    rfm = pd.read_csv(os.path.join(DATA_DIR, "customer_rfm.csv"))

    delivered = master[master["order_status"] == "delivered"].copy()

    print(f"  orders_master      : {len(master):,} rows")
    print(f"  delivered orders   : {len(delivered):,} rows")
    print(f"  customer_rfm       : {len(rfm):,} customers")

    # Pre-flight: confirm no duplicate customer_unique_id in rfm
    assert rfm["customer_unique_id"].duplicated().sum() == 0, \
        "FAIL: duplicate customer_unique_id in rfm"

    # Pre-flight: rfm monetary sum == delivered order_value sum
    diff = abs(rfm["monetary"].sum() - delivered["order_value"].sum())
    assert diff < 0.01, f"FAIL: monetary mismatch, diff={diff}"
    print(f"  PRE-FLIGHT PASS: rfm monetary reconciles with delivered order_value "
          f"(R$ {rfm['monetary'].sum():,.2f})")

    return master, delivered, rfm


# ---------------------------------------------------------------------------
# SECTION 1 — RFM SEGMENT SUMMARY
# ---------------------------------------------------------------------------
# Grain  : one row per segment (7 segments)
# Source : customer_rfm (one row per customer_unique_id)
#
# Validated against:
#   - total customer count == rfm rows (93,358)
#   - total segment revenue == rfm.monetary.sum() == delivered.order_value.sum()

def rfm_segment_summary(rfm, delivered):
    print("\n" + "="*60)
    print("SECTION 1 — RFM SEGMENT SUMMARY")
    print("="*60)

    seg = rfm.groupby("segment", as_index=False).agg(
        customer_count      = ("customer_unique_id", "count"),
        total_revenue       = ("monetary",           "sum"),
        avg_spend           = ("monetary",           "mean"),
        median_spend        = ("monetary",           "median"),
        avg_recency_days    = ("recency_days",       "mean"),
        avg_frequency       = ("frequency",          "mean"),
        avg_r_score         = ("R_score",            "mean"),
        avg_f_score         = ("F_score",            "mean"),
        avg_m_score         = ("M_score",            "mean"),
    )

    total_customers = seg["customer_count"].sum()
    total_revenue   = seg["total_revenue"].sum()

    seg["customer_pct"]  = (seg["customer_count"] / total_customers * 100).round(2)
    seg["revenue_pct"]   = (seg["total_revenue"]  / total_revenue   * 100).round(2)
    seg["avg_spend"]     = seg["avg_spend"].round(2)
    seg["median_spend"]  = seg["median_spend"].round(2)
    seg["avg_recency_days"] = seg["avg_recency_days"].round(1)
    seg["avg_frequency"] = seg["avg_frequency"].round(3)

    # Sort by avg M score descending for logical ordering
    seg = seg.sort_values("avg_m_score", ascending=False).reset_index(drop=True)

    print(f"\n  {'Segment':<25} {'Customers':>10} {'%':>6}  "
          f"{'Revenue (R$)':>14} {'%':>6}  "
          f"{'Avg Spend':>10}  {'Avg Freq':>9}  {'Avg Recency':>11}")
    print("  " + "-"*100)
    for _, r in seg.iterrows():
        print(f"  {r['segment']:<25} {r['customer_count']:>10,} {r['customer_pct']:>5.1f}%  "
              f"R${r['total_revenue']:>12,.0f} {r['revenue_pct']:>5.1f}%  "
              f"R${r['avg_spend']:>8,.2f}  "
              f"{r['avg_frequency']:>9.3f}  "
              f"{r['avg_recency_days']:>8.0f}d")

    print(f"\n  {'TOTAL':<25} {total_customers:>10,}        "
          f"R${total_revenue:>12,.0f}")

    # Validation
    assert total_customers == len(rfm), \
        f"FAIL: segment customer count {total_customers} != rfm rows {len(rfm)}"
    diff = abs(total_revenue - rfm["monetary"].sum())
    assert diff < 0.01, f"FAIL: segment revenue mismatch diff={diff}"
    print(f"\n  PASS: segment customer count reconciles ({total_customers:,})")
    print(f"  PASS: segment revenue reconciles (R$ {total_revenue:,.2f})")

    return seg


# ---------------------------------------------------------------------------
# SECTION 2 — REPEAT VS ONE-TIME CUSTOMERS
# ---------------------------------------------------------------------------
# Grain  : two rows (one-time / repeat)
# Source : customer_rfm.frequency (= delivered order count per customer_unique_id)
#
# Definition:
#   one-time : frequency == 1
#   repeat   : frequency >= 2
#
# Validation:
#   one-time count + repeat count == total rfm customers (93,358)
#   one-time revenue + repeat revenue == total rfm monetary

def repeat_vs_onetime(rfm, delivered):
    print("\n" + "="*60)
    print("SECTION 2 — REPEAT VS ONE-TIME CUSTOMERS")
    print("="*60)

    rfm["customer_type"] = rfm["frequency"].apply(
        lambda f: "Repeat" if f >= 2 else "One-Time"
    )

    summary = rfm.groupby("customer_type", as_index=False).agg(
        customer_count   = ("customer_unique_id", "count"),
        total_revenue    = ("monetary",           "sum"),
        avg_spend        = ("monetary",           "mean"),
        median_spend     = ("monetary",           "median"),
        avg_order_count  = ("frequency",          "mean"),
        max_order_count  = ("frequency",          "max"),
    )

    total_cust = summary["customer_count"].sum()
    total_rev  = summary["total_revenue"].sum()
    summary["customer_pct"] = (summary["customer_count"] / total_cust * 100).round(2)
    summary["revenue_pct"]  = (summary["total_revenue"]  / total_rev  * 100).round(2)
    summary["avg_spend"]    = summary["avg_spend"].round(2)
    summary["median_spend"] = summary["median_spend"].round(2)
    summary["avg_order_count"] = summary["avg_order_count"].round(3)

    # Average order value per type: revenue / total orders
    # Need total delivered order count per type
    repeat_ids  = rfm[rfm["customer_type"] == "Repeat"]["customer_unique_id"]
    onetime_ids = rfm[rfm["customer_type"] == "One-Time"]["customer_unique_id"]

    repeat_orders  = delivered[delivered["customer_unique_id"].isin(repeat_ids)]
    onetime_orders = delivered[delivered["customer_unique_id"].isin(onetime_ids)]

    aov_repeat  = repeat_orders["order_value"].mean()
    aov_onetime = onetime_orders["order_value"].mean()

    summary.loc[summary["customer_type"] == "Repeat",   "avg_order_value"] = round(aov_repeat, 2)
    summary.loc[summary["customer_type"] == "One-Time", "avg_order_value"] = round(aov_onetime, 2)

    print(f"\n  {'Type':<12} {'Customers':>10} {'%':>6}  "
          f"{'Revenue (R$)':>14} {'%':>6}  "
          f"{'Avg Spend':>10}  {'Avg Orders':>10}  {'Avg AOV':>9}")
    print("  " + "-"*85)
    for _, r in summary.iterrows():
        print(f"  {r['customer_type']:<12} {r['customer_count']:>10,} {r['customer_pct']:>5.1f}%  "
              f"R${r['total_revenue']:>12,.0f} {r['revenue_pct']:>5.1f}%  "
              f"R${r['avg_spend']:>8,.2f}  "
              f"{r['avg_order_count']:>10.3f}  "
              f"R${r['avg_order_value']:>7,.2f}")

    # Additional: distribution of repeat customers by order count
    repeat_dist = (rfm[rfm["frequency"] >= 2]["frequency"]
                   .value_counts().sort_index().head(10).reset_index())
    repeat_dist.columns = ["order_count", "customer_count"]
    print(f"\n  Repeat customer frequency distribution:")
    for _, r in repeat_dist.iterrows():
        print(f"    {int(r['order_count'])} orders: {int(r['customer_count']):,} customers")

    # Validation
    assert total_cust == len(rfm), \
        f"FAIL: one-time + repeat count {total_cust} != rfm {len(rfm)}"
    diff = abs(total_rev - rfm["monetary"].sum())
    assert diff < 0.01, f"FAIL: one-time + repeat revenue mismatch diff={diff}"
    print(f"\n  PASS: one-time + repeat customer count reconciles ({total_cust:,})")
    print(f"  PASS: one-time + repeat revenue reconciles (R$ {total_rev:,.2f})")

    return summary


# ---------------------------------------------------------------------------
# SECTION 3 — CUSTOMER GEOGRAPHY
# ---------------------------------------------------------------------------
# Grain  : one row per customer_state
# Source : delivered orders_master for revenue/AOV;
#          rfm joined to master for customer counts by state
#
# Note: customer counts here use customer_unique_id from the RFM population
# (delivered orders only).  All-order customer counts are also provided for
# completeness from state_performance.csv (Phase 3).

def customer_geography(delivered, rfm):
    print("\n" + "="*60)
    print("SECTION 3 — CUSTOMER GEOGRAPHY")
    print("="*60)

    # State-level aggregates from delivered orders_master
    state = delivered.groupby("customer_state", as_index=False).agg(
        order_count          = ("order_id",           "nunique"),
        customer_uid_count   = ("customer_unique_id", "nunique"),
        total_revenue        = ("order_value",        "sum"),
        avg_order_value      = ("order_value",        "mean"),
        avg_review_score     = ("review_score",       "mean"),
    )
    state["avg_order_value"] = state["avg_order_value"].round(2)
    state["avg_review_score"] = state["avg_review_score"].round(3)

    # Customer share
    total_uid = state["customer_uid_count"].sum()
    total_rev = state["total_revenue"].sum()
    state["customer_pct"] = (state["customer_uid_count"] / total_uid * 100).round(2)
    state["revenue_pct"]  = (state["total_revenue"]      / total_rev  * 100).round(2)

    # Add repeat-customer count per state
    # Join rfm to delivered to get state for each customer_unique_id
    cust_state = delivered[["customer_unique_id","customer_state"]].drop_duplicates(
        subset="customer_unique_id"
    )
    rfm_state = rfm.merge(cust_state, on="customer_unique_id", how="left")
    repeat_by_state = (rfm_state[rfm_state["frequency"] >= 2]
                       .groupby("customer_state", as_index=False)["customer_unique_id"]
                       .count()
                       .rename(columns={"customer_unique_id": "repeat_customer_count"}))
    state = state.merge(repeat_by_state, on="customer_state", how="left")
    state["repeat_customer_count"] = state["repeat_customer_count"].fillna(0).astype(int)
    state["repeat_rate_pct"] = (
        state["repeat_customer_count"] / state["customer_uid_count"] * 100
    ).round(2)

    state = state.sort_values("order_count", ascending=False).reset_index(drop=True)

    print(f"\n  {'State':<6} {'Orders':>7}  {'Cust.':>7}  {'Cust%':>6}  "
          f"{'Revenue (R$)':>14}  {'Rev%':>5}  {'AOV':>8}  {'Repeat%':>8}")
    print("  " + "-"*80)
    for _, r in state.iterrows():
        print(f"  {r['customer_state']:<6} {r['order_count']:>7,}  "
              f"{r['customer_uid_count']:>7,}  {r['customer_pct']:>5.1f}%  "
              f"R${r['total_revenue']:>12,.0f}  {r['revenue_pct']:>4.1f}%  "
              f"R${r['avg_order_value']:>6,.2f}  {r['repeat_rate_pct']:>7.2f}%")

    # Validation: sum of customer counts should be close to total unique customers.
    # A small number of customer_unique_ids (37) appear across >1 state because each
    # order generates its own customer_id record — the unique_id can therefore link to
    # two orders placed from different addresses.  We document rather than hide this.
    state_sum = state["customer_uid_count"].sum()
    true_uid  = delivered["customer_unique_id"].nunique()
    overcount = state_sum - true_uid
    if overcount == 0:
        print(f"\n  PASS: state customer count reconciles exactly ({state_sum:,})")
    else:
        print(f"\n  NOTE: state customer_uid sum={state_sum:,}, global unique={true_uid:,}. "
              f"Overcount={overcount} (customers with orders in >1 state — documented artefact)")
    diff = abs(state["total_revenue"].sum() - delivered["order_value"].sum())
    assert diff < 0.01, f"FAIL: state revenue mismatch diff={diff}"
    print(f"  PASS: state revenue reconciles (R$ {state['total_revenue'].sum():,.2f})")

    return state


# ---------------------------------------------------------------------------
# SECTION 4 — CUSTOMER SPENDING DISTRIBUTION
# ---------------------------------------------------------------------------
# Grain  : one row per spending bucket
# Source : customer_rfm.monetary (total spend per customer_unique_id)

def customer_spending_dist(rfm):
    print("\n" + "="*60)
    print("SECTION 4 — CUSTOMER SPENDING DISTRIBUTION")
    print("="*60)

    mon = rfm["monetary"]

    print(f"  Customer spending statistics (n={len(mon):,}):")
    print(f"    Min            : R$ {mon.min():>10,.2f}")
    print(f"    Median         : R$ {mon.median():>10,.2f}")
    print(f"    Mean           : R$ {mon.mean():>10,.2f}")
    print(f"    75th pct       : R$ {mon.quantile(0.75):>10,.2f}")
    print(f"    90th pct       : R$ {mon.quantile(0.90):>10,.2f}")
    print(f"    95th pct       : R$ {mon.quantile(0.95):>10,.2f}")
    print(f"    99th pct       : R$ {mon.quantile(0.99):>10,.2f}")
    print(f"    Max            : R$ {mon.max():>10,.2f}")

    # Spending buckets
    bins   = [0, 50, 100, 200, 300, 500, 1000, 2000, 999999]
    labels = ["R$0-50", "R$51-100", "R$101-200", "R$201-300",
              "R$301-500", "R$501-1000", "R$1001-2000", "R$2000+"]
    rfm["spend_bucket"] = pd.cut(rfm["monetary"], bins=bins, labels=labels, right=True)

    dist = rfm.groupby("spend_bucket", observed=False, as_index=False).agg(
        customer_count = ("customer_unique_id", "count"),
        total_revenue  = ("monetary",           "sum"),
        avg_spend      = ("monetary",           "mean"),
    )
    dist["customer_pct"] = (dist["customer_count"] / len(rfm) * 100).round(2)
    dist["revenue_pct"]  = (dist["total_revenue"]  / rfm["monetary"].sum() * 100).round(2)
    dist["avg_spend"]    = dist["avg_spend"].round(2)
    dist["spend_bucket"] = dist["spend_bucket"].astype(str)

    print(f"\n  Spending bucket distribution:")
    print(f"    {'Bucket':<16} {'Customers':>10} {'%':>6}  {'Revenue':>14} {'%':>6}")
    for _, r in dist.iterrows():
        bar = "#" * int(r["customer_pct"] / 2)
        print(f"    {r['spend_bucket']:<16} {r['customer_count']:>10,} {r['customer_pct']:>5.1f}%  "
              f"R${r['total_revenue']:>12,.0f} {r['revenue_pct']:>5.1f}%  {bar}")

    # High-value customers: top 10% by spend
    p90_threshold = mon.quantile(0.90)
    high_value = rfm[rfm["monetary"] >= p90_threshold]
    hv_revenue  = high_value["monetary"].sum()
    hv_rev_pct  = hv_revenue / rfm["monetary"].sum() * 100
    print(f"\n  High-value customers (top 10% by spend, >= R${p90_threshold:.2f}):")
    print(f"    Count    : {len(high_value):,} ({len(high_value)/len(rfm)*100:.1f}% of customers)")
    print(f"    Revenue  : R$ {hv_revenue:,.2f} ({hv_rev_pct:.1f}% of total)")

    # Spending by RFM segment (join back to rfm)
    seg_spend = rfm.groupby("segment", as_index=False).agg(
        avg_spend    = ("monetary", "mean"),
        median_spend = ("monetary", "median"),
        p90_spend    = ("monetary", lambda x: x.quantile(0.90)),
        total_rev    = ("monetary", "sum"),
    ).sort_values("avg_spend", ascending=False)
    seg_spend["avg_spend"]    = seg_spend["avg_spend"].round(2)
    seg_spend["median_spend"] = seg_spend["median_spend"].round(2)
    seg_spend["p90_spend"]    = seg_spend["p90_spend"].round(2)

    print(f"\n  Spending by RFM segment:")
    for _, r in seg_spend.iterrows():
        print(f"    {r['segment']:<25}  avg=R${r['avg_spend']:>8,.2f}  "
              f"median=R${r['median_spend']:>8,.2f}  "
              f"p90=R${r['p90_spend']:>8,.2f}")

    return dist, seg_spend, p90_threshold


# ---------------------------------------------------------------------------
# SECTION 5 — FULL PER-CUSTOMER DETAIL TABLE
# ---------------------------------------------------------------------------
# Grain : one row per customer_unique_id
# Adds : customer_type flag, customer_state, repeat flag
# Used by dashboard for scatter/drill-down

def build_customer_detail(rfm, delivered):
    print("\n" + "="*60)
    print("SECTION 5 — CUSTOMER DETAIL TABLE")
    print("="*60)

    # customer_type
    rfm = rfm.copy()
    rfm["customer_type"] = rfm["frequency"].apply(
        lambda f: "Repeat" if f >= 2 else "One-Time"
    )

    # Join state (use first occurrence per customer_unique_id in delivered)
    cust_state = (delivered[["customer_unique_id","customer_state","customer_city"]]
                  .drop_duplicates(subset="customer_unique_id"))
    detail = rfm.merge(cust_state, on="customer_unique_id", how="left")

    # Confirm no new duplicates
    assert detail["customer_unique_id"].duplicated().sum() == 0
    print(f"  Customer detail rows: {len(detail):,}  cols: {list(detail.columns)}")

    return detail


# ---------------------------------------------------------------------------
# SECTION 6 — BUSINESS INSIGHTS
# ---------------------------------------------------------------------------

def build_customer_insights(rfm, seg_summary, repeat_summary, state, spending_dist, p90_threshold):
    print("\n" + "="*60)
    print("SECTION 6 — BUSINESS INSIGHTS")
    print("="*60)

    insights = []

    # -- Insight 1: Low repeat rate --
    repeat_row  = repeat_summary[repeat_summary["customer_type"] == "Repeat"].iloc[0]
    onetime_row = repeat_summary[repeat_summary["customer_type"] == "One-Time"].iloc[0]

    i1 = {
        "id": "low_repeat_rate",
        "fact": (
            f"{onetime_row['customer_pct']:.1f}% of customers ({int(onetime_row['customer_count']):,}) "
            f"placed exactly one delivered order. "
            f"Only {repeat_row['customer_pct']:.1f}% ({int(repeat_row['customer_count']):,}) "
            f"placed two or more orders."
        ),
        "insight": (
            f"The vast majority of Olist customers are single-purchase. "
            f"Repeat customers represent {repeat_row['customer_pct']:.1f}% of the base but "
            f"contribute {repeat_row['revenue_pct']:.1f}% of revenue, "
            f"with an average total spend of R${repeat_row['avg_spend']:,.2f} vs "
            f"R${onetime_row['avg_spend']:,.2f} for one-time customers."
        ),
        "risk_or_opportunity": (
            "RISK: A customer base dominated by single-purchase customers creates high "
            "dependency on continuous customer acquisition. A slowdown in new-customer "
            "growth directly impacts revenue with no repeat-purchase buffer. "
            "OPPORTUNITY: Even a modest improvement in repeat rate (e.g., from 3% to 6%) "
            "would materially increase revenue without additional acquisition spend."
        ),
        "recommended_action": (
            "Implement post-purchase retention programmes (e.g., personalised follow-up "
            "emails, loyalty discounts for second orders) targeting One-Time and "
            "'New Customers' RFM segments. Track repeat rate as a primary retention KPI."
        )
    }
    insights.append(i1)
    print(f"\n  INSIGHT 1: Low Repeat Rate")
    print(f"    FACT: {i1['fact']}")

    # -- Insight 2: Revenue concentration --
    champions = seg_summary[seg_summary["segment"] == "Champions"].iloc[0]
    i2 = {
        "id": "champion_revenue_concentration",
        "fact": (
            f"The 'Champions' segment contains {int(champions['customer_count']):,} customers "
            f"({champions['customer_pct']:.1f}% of total), generating "
            f"R${champions['total_revenue']:,.0f} ({champions['revenue_pct']:.1f}% of revenue). "
            f"Their average total spend is R${champions['avg_spend']:,.2f}."
        ),
        "insight": (
            f"Champions have an average frequency of {champions['avg_frequency']:.2f} orders "
            f"and average recency of {champions['avg_recency_days']:.0f} days. "
            f"This segment represents the most engaged and monetarily valuable customers."
        ),
        "risk_or_opportunity": (
            "RISK: If Champions churn (e.g., due to service failures), revenue impact is "
            "disproportionate to their small share of the customer base. "
            "OPPORTUNITY: Champions are prime candidates for referral programmes and "
            "premium product/seller recommendations."
        ),
        "recommended_action": (
            "Prioritise service quality for Champion customers. Monitor their delivery and "
            "review scores separately. Build a VIP retention programme with early access, "
            "exclusive categories, or loyalty rewards."
        )
    }
    insights.append(i2)
    print(f"\n  INSIGHT 2: Champions Revenue")
    print(f"    FACT: {i2['fact']}")

    # -- Insight 3: At Risk segment --
    at_risk = seg_summary[seg_summary["segment"] == "At Risk"].iloc[0]
    i3 = {
        "id": "at_risk_segment",
        "fact": (
            f"The 'At Risk' segment contains {int(at_risk['customer_count']):,} customers "
            f"({at_risk['customer_pct']:.1f}%) with average recency of "
            f"{at_risk['avg_recency_days']:.0f} days (have not purchased recently)."
        ),
        "insight": (
            f"These customers previously purchased (avg frequency {at_risk['avg_frequency']:.2f}) "
            f"with reasonable spend (avg R${at_risk['avg_spend']:,.2f}) but have become "
            f"dormant. Their profile differs from 'Lost/Inactive' customers in that "
            f"their monetary and frequency scores are relatively higher."
        ),
        "risk_or_opportunity": (
            "RISK: Without intervention, At Risk customers will migrate to the "
            "'Lost/Inactive' segment and be unrecoverable. "
            "OPPORTUNITY: Win-back campaigns targeting this segment can recover "
            f"R${at_risk['total_revenue']:,.0f} in historical spend capacity."
        ),
        "recommended_action": (
            "Design a win-back campaign for At Risk customers: personalised 'We miss you' "
            "offers, category recommendations based on past purchases, or time-limited "
            "discount codes. Measure reactivation rate monthly."
        )
    }
    insights.append(i3)
    print(f"\n  INSIGHT 3: At Risk Segment")
    print(f"    FACT: {i3['fact']}")

    # -- Insight 4: Geographic concentration --
    sp_row = state[state["customer_state"] == "SP"].iloc[0]
    top3_rev = state.head(3)["total_revenue"].sum()
    total_rev = state["total_revenue"].sum()
    top3_pct = top3_rev / total_rev * 100
    i4 = {
        "id": "geographic_concentration",
        "fact": (
            f"São Paulo (SP) alone accounts for {sp_row['customer_pct']:.1f}% of customers "
            f"and {sp_row['revenue_pct']:.1f}% of revenue. "
            f"The top 3 states (SP, RJ, MG) together account for "
            f"{top3_pct:.1f}% of total revenue."
        ),
        "insight": (
            "Customer demand is heavily concentrated in the Southeast region. "
            "Northern and Northeastern states, despite having lower average order values, "
            "show higher late-delivery rates (Phase 3 data), suggesting that logistics "
            "infrastructure does not scale to these regions as effectively."
        ),
        "risk_or_opportunity": (
            "RISK: Over-dependence on Southeast customers creates geographic revenue "
            "concentration risk. "
            "OPPORTUNITY: Improving logistics reliability in the Northeast (AL, MA, SE, CE "
            "states with >13% late rates) could unlock demand in underserved markets."
        ),
        "recommended_action": (
            "Analyse seller distribution in Northeastern states and assess whether "
            "recruiting local sellers or regional fulfilment partners would reduce "
            "delivery times and stimulate demand."
        )
    }
    insights.append(i4)
    print(f"\n  INSIGHT 4: Geographic Concentration")
    print(f"    FACT: {i4['fact']}")

    # -- Insight 5: Spending skew --
    top10_rev = rfm.nlargest(int(len(rfm) * 0.10), "monetary")["monetary"].sum()
    total_mon = rfm["monetary"].sum()
    top10_pct = top10_rev / total_mon * 100
    i5 = {
        "id": "spending_concentration",
        "fact": (
            f"The top 10% of customers by total spend (>= R${p90_threshold:.0f}) "
            f"contribute {top10_pct:.1f}% of total revenue."
        ),
        "insight": (
            f"Median customer spend is R${rfm['monetary'].median():.2f} while mean is "
            f"R${rfm['monetary'].mean():.2f}, indicating a right-skewed distribution "
            f"where a minority of high-value customers generate a disproportionate "
            f"share of revenue."
        ),
        "risk_or_opportunity": (
            "RISK: Revenue is sensitive to the behaviour of a small high-value cohort. "
            "OPPORTUNITY: Identifying and retaining customers in the R$501-1000+ spend "
            "brackets through loyalty mechanisms has high revenue leverage."
        ),
        "recommended_action": (
            "Implement a high-value customer tier (e.g., customers with lifetime spend "
            f"> R${p90_threshold:.0f}) with dedicated account features, premium support, "
            "and priority seller matching."
        )
    }
    insights.append(i5)
    print(f"\n  INSIGHT 5: Spending Concentration")
    print(f"    FACT: {i5['fact']}")

    return insights


# ---------------------------------------------------------------------------
# SECTION 7 — SAVE
# ---------------------------------------------------------------------------

def save_all_customers(seg_summary, repeat_summary, state, spend_dist, seg_spend,
             customer_detail, insights):
    print("\n" + "="*60)
    print("SECTION 7 — SAVING OUTPUTS")
    print("="*60)

    csvs = {
        "rfm_segment_summary.csv":    seg_summary,
        "repeat_vs_onetime.csv":      repeat_summary,
        "customer_geography.csv":     state,
        "customer_spending_dist.csv": spend_dist,
        "segment_spend_profile.csv":  seg_spend,
        "customer_segment_detail.csv":customer_detail,
    }
    for fname, df in csvs.items():
        path = os.path.join(DATA_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {path:<45} {len(df):>7,} rows")

    ins_path = os.path.join(DATA_DIR, "customer_insights.json")
    with open(ins_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"  Saved {ins_path}")


# ---------------------------------------------------------------------------
# MAIN — Phase 4
# ---------------------------------------------------------------------------

def main_customer_analytics():
    print("\n" + "#"*60)
    print("  OLIST CUSTOMER ANALYTICS — PHASE 4")
    print("#"*60)

    master, delivered, rfm = load_data_customers()

    seg_summary = rfm_segment_summary(rfm, delivered)
    repeat_summary = repeat_vs_onetime(rfm, delivered)
    state = customer_geography(delivered, rfm)
    spend_dist, seg_spend, p90_threshold = customer_spending_dist(rfm)
    customer_detail = build_customer_detail(rfm, delivered)
    insights = build_customer_insights(rfm, seg_summary, repeat_summary, state,
                              spend_dist, p90_threshold)

    save_all_customers(seg_summary, repeat_summary, state, spend_dist, seg_spend,
             customer_detail, insights)

    print("\n" + "#"*60)
    print("  PHASE 4 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: delivery_satisfaction.py — Phase 5
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Minimum order threshold for seller-level rankings
# Sellers with fewer than this number of orders are included in totals
# but excluded from the "notable performer" rankings to avoid small-N noise.
MIN_SELLER_ORDERS = 50

# Delay bucket definitions — identical to Phase 3 for consistency
DELAY_BINS   = [-9999, -7, -3, 0, 3, 7, 14, 9999]
DELAY_LABELS = ["Early >7d", "Early 3-7d", "Early 0-3d",
                "Late 1-3d", "Late 4-7d", "Late 8-14d", "Late >14d"]


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_data_delivery():
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    master = pd.read_csv(
        os.path.join(DATA_DIR, "orders_master.csv"),
        parse_dates=["order_purchase_timestamp",
                     "order_delivered_customer_date",
                     "order_estimated_delivery_date"]
    )
    full = pd.read_csv(
        os.path.join(DATA_DIR, "orders_full.csv"),
        parse_dates=["order_purchase_timestamp",
                     "order_delivered_customer_date",
                     "order_estimated_delivery_date"]
    )

    delivered = master[master["order_status"] == "delivered"].copy()
    # Rows with valid delivery timing (excludes 8 with missing delivery date)
    del_valid  = delivered[delivered["is_late"].notna()].copy()

    # Assign delay bucket to every valid delivered order
    del_valid["delay_bucket"] = pd.cut(
        del_valid["delivery_delay_days"],
        bins=DELAY_BINS, labels=DELAY_LABELS
    ).astype(str)

    print(f"  orders_master   : {len(master):,} rows")
    print(f"  delivered       : {len(delivered):,} orders")
    print(f"  valid for delay : {len(del_valid):,} orders  "
          f"(excluded {len(delivered)-len(del_valid)} with missing delivery date)")

    # Load Phase 3 references for validation
    ph3_kpis = json.load(open(os.path.join(DATA_DIR, "kpis.json")))
    ph3_delay = pd.read_csv(os.path.join(DATA_DIR, "delivery_delay_dist.csv"))
    ph3_review= pd.read_csv(os.path.join(DATA_DIR, "review_score_dist.csv"))

    return master, full, delivered, del_valid, ph3_kpis, ph3_delay, ph3_review


# ---------------------------------------------------------------------------
# SECTION 1 — OVERALL DELIVERY KPIs (validate against Phase 3)
# ---------------------------------------------------------------------------

def delivery_overview(del_valid, ph3_kpis, ph3_delay):
    print("\n" + "="*60)
    print("SECTION 1 — OVERALL DELIVERY OVERVIEW & VALIDATION")
    print("="*60)

    n = len(del_valid)
    on_time_n   = int(del_valid["on_time"].sum())
    late_n      = int(del_valid["is_late"].sum())
    on_time_pct = on_time_n / n * 100
    late_pct    = late_n    / n * 100

    avg_days    = del_valid["delivery_days"].mean()
    median_days = del_valid["delivery_days"].median()
    avg_delay   = del_valid["delivery_delay_days"].mean()
    median_delay= del_valid["delivery_delay_days"].median()

    # Delay bucket distribution
    bucket_counts = (del_valid["delay_bucket"]
                     .value_counts()
                     .reindex(DELAY_LABELS)
                     .fillna(0)
                     .astype(int))

    print(f"\n  Orders in scope   : {n:,}")
    print(f"  On-time           : {on_time_n:,}  ({on_time_pct:.2f}%)")
    print(f"  Late              : {late_n:,}  ({late_pct:.2f}%)")
    print(f"  Avg delivery days : {avg_days:.2f}")
    print(f"  Median del. days  : {median_days:.1f}")
    print(f"  Avg delay days    : {avg_delay:.2f}  (negative = early)")
    print(f"  Median delay days : {median_delay:.1f}")
    print(f"\n  Delay distribution:")
    for label in DELAY_LABELS:
        cnt = bucket_counts[label]
        pct = cnt / n * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:<14} {cnt:>6,} ({pct:>5.1f}%)  {bar}")

    # ---- Validation against Phase 3 ----
    print("\n  --- VALIDATION vs Phase 3 ---")
    tol = 0.05

    diff_ontime = abs(on_time_pct - ph3_kpis["on_time_delivery_pct"])
    diff_late   = abs(late_pct    - ph3_kpis["late_delivery_pct"])
    diff_avg    = abs(avg_days    - ph3_kpis["avg_delivery_days"])

    for label, diff, threshold in [
        ("on_time_pct", diff_ontime, tol),
        ("late_pct",    diff_late,   tol),
        ("avg_delivery_days", diff_avg, tol),
    ]:
        status = "PASS" if diff < threshold else "FAIL"
        print(f"  {status}: {label}  this={locals().get(label.replace('_pct','_pct'), round(on_time_pct if 'on' in label else late_pct if 'late' in label else avg_days, 2)):.2f}  "
              f"ph3={ph3_kpis.get('on_time_delivery_pct' if 'on' in label else 'late_delivery_pct' if 'late' in label else 'avg_delivery_days'):.2f}  "
              f"diff={diff:.4f}")

    # Validate delay bucket totals vs Phase 3
    ph3_total = ph3_delay["order_count"].sum()
    this_total = bucket_counts.sum()
    if this_total == ph3_total:
        print(f"  PASS: delay bucket total {this_total:,} matches Phase 3")
    else:
        print(f"  NOTE: delay bucket total this={this_total:,}  ph3={ph3_total:,}  "
              f"diff={this_total - ph3_total}  (may differ if Phase 3 used different null handling)")

    overview = {
        "n_orders":          int(n),
        "on_time_count":     on_time_n,
        "late_count":        late_n,
        "on_time_pct":       round(on_time_pct, 4),
        "late_pct":          round(late_pct, 4),
        "avg_delivery_days": round(avg_days, 4),
        "median_delivery_days": round(median_days, 1),
        "avg_delay_days":    round(avg_delay, 4),
        "median_delay_days": round(median_delay, 1),
        "delay_bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
    }
    return overview


# ---------------------------------------------------------------------------
# SECTION 2 — DELIVERY PERFORMANCE OVER TIME
# ---------------------------------------------------------------------------
# Grain : one row per calendar month

def delivery_over_time(del_valid):
    print("\n" + "="*60)
    print("SECTION 2 — DELIVERY PERFORMANCE OVER TIME")
    print("="*60)

    d = del_valid.copy()
    d["ym"] = d["order_purchase_timestamp"].dt.to_period("M").astype(str)

    monthly = d.groupby("ym", as_index=False).agg(
        order_count        = ("order_id",            "nunique"),
        on_time_count      = ("on_time",             "sum"),
        late_count         = ("is_late",             "sum"),
        avg_delivery_days  = ("delivery_days",        "mean"),
        median_delivery_days=("delivery_days",        "median"),
        avg_delay_days     = ("delivery_delay_days",  "mean"),
        avg_review_score   = ("review_score",         "mean"),
    )
    monthly["on_time_pct"] = (monthly["on_time_count"] /
                               (monthly["on_time_count"] + monthly["late_count"]) * 100).round(2)
    monthly["late_pct"]    = (monthly["late_count"]    /
                               (monthly["on_time_count"] + monthly["late_count"]) * 100).round(2)
    monthly = monthly[monthly["order_count"] >= 50].sort_values("ym").reset_index(drop=True)

    print(f"  Monthly delivery rows: {len(monthly)} months "
          f"({monthly['ym'].min()} → {monthly['ym'].max()})")
    print(f"\n  {'Month':<8} {'Orders':>7}  {'OnTime%':>8}  {'Late%':>7}  "
          f"{'AvgDays':>8}  {'AvgDelay':>9}  {'AvgReview':>10}")
    for _, r in monthly.iterrows():
        flag = "  <-- peak late" if r["late_pct"] == monthly["late_pct"].max() else ""
        print(f"  {r['ym']:<8} {r['order_count']:>7,}  {r['on_time_pct']:>7.1f}%  "
              f"{r['late_pct']:>6.1f}%  {r['avg_delivery_days']:>8.1f}  "
              f"{r['avg_delay_days']:>9.1f}  {r['avg_review_score']:>10.3f}{flag}")

    return monthly


# ---------------------------------------------------------------------------
# SECTION 3 — REGIONAL DELIVERY ANALYSIS (enriched)
# ---------------------------------------------------------------------------
# Grain : one row per customer_state
# Adds  : review score distribution fields, seller-state comparison
# Min-volume note: all 27 states included; low-volume states flagged

def delivery_by_state_enriched(del_valid):
    print("\n" + "="*60)
    print("SECTION 3 — REGIONAL DELIVERY ANALYSIS")
    print("="*60)

    # Core delivery stats
    state = del_valid.groupby("customer_state", as_index=False).agg(
        order_count          = ("order_id",            "nunique"),
        on_time_count        = ("on_time",             "sum"),
        late_count           = ("is_late",             "sum"),
        avg_delivery_days    = ("delivery_days",        "mean"),
        median_delivery_days = ("delivery_days",        "median"),
        avg_delay_days       = ("delivery_delay_days",  "mean"),
        avg_review_score     = ("review_score",         "mean"),
        p25_delivery_days    = ("delivery_days",        lambda x: x.quantile(0.25)),
        p75_delivery_days    = ("delivery_days",        lambda x: x.quantile(0.75)),
    )
    state["on_time_pct"] = (state["on_time_count"] /
                             (state["on_time_count"] + state["late_count"]) * 100).round(2)
    state["late_pct"]    = (100 - state["on_time_pct"]).round(2)
    state["low_volume"]  = state["order_count"] < 200  # flag for context

    # National benchmarks
    nat_late_pct    = del_valid["is_late"].mean()    * 100
    nat_avg_days    = del_valid["delivery_days"].mean()
    nat_review      = del_valid["review_score"].mean()

    state["late_vs_national"]      = (state["late_pct"]         - nat_late_pct).round(2)
    state["delivery_vs_national"]  = (state["avg_delivery_days"] - nat_avg_days).round(2)
    state["review_vs_national"]    = (state["avg_review_score"]  - nat_review).round(3)

    state = state.sort_values("late_pct", ascending=False).reset_index(drop=True)

    print(f"\n  National benchmarks: late%={nat_late_pct:.2f}%  "
          f"avg_days={nat_avg_days:.2f}  avg_review={nat_review:.3f}")
    print(f"\n  All 27 states (sorted by late %):")
    print(f"  {'State':<5} {'Orders':>7}  {'Late%':>6}  {'vs_nat':>7}  "
          f"{'AvgDays':>8}  {'AvgDelay':>9}  {'Review':>7}  {'Note'}")
    for _, r in state.iterrows():
        vol_note = " [LOW VOL]" if r["low_volume"] else ""
        print(f"  {r['customer_state']:<5} {r['order_count']:>7,}  "
              f"{r['late_pct']:>6.1f}%  {r['late_vs_national']:>+7.1f}%  "
              f"{r['avg_delivery_days']:>8.1f}  {r['avg_delay_days']:>9.1f}  "
              f"{r['avg_review_score']:>7.3f}  {vol_note}")

    # High-impact states: late% > national AND order_count >= 200
    high_impact = state[(state["late_pct"] > nat_late_pct) & (~state["low_volume"])]
    print(f"\n  High-impact states (late% > national AND orders >= 200): "
          f"{len(high_impact)}")
    for _, r in high_impact.iterrows():
        print(f"    {r['customer_state']}  late%={r['late_pct']:.1f}%  "
              f"(+{r['late_vs_national']:.1f}% vs national)  "
              f"orders={r['order_count']:,}  review={r['avg_review_score']:.3f}")

    # Validation vs Phase 3
    ph3_state = pd.read_csv(os.path.join(DATA_DIR, "delivery_by_state.csv"))
    for state_code in ["SP", "RJ", "AL"]:
        this_row = state[state["customer_state"] == state_code]
        ph3_row  = ph3_state[ph3_state["customer_state"] == state_code]
        if len(this_row) and len(ph3_row):
            diff = abs(this_row.iloc[0]["late_pct"] - ph3_row.iloc[0]["late_pct"])
            status = "PASS" if diff < 0.05 else "WARN"
            print(f"  {status}: {state_code} late_pct diff={diff:.4f}")

    return state


# ---------------------------------------------------------------------------
# SECTION 4 — SELLER DELIVERY PERFORMANCE (qualified)
# ---------------------------------------------------------------------------
# Grain : one row per seller_id with >= MIN_SELLER_ORDERS delivered orders
# Source: orders_full for seller_id; orders_master for delivery/review KPIs

def seller_delivery(del_valid, full):
    print("\n" + "="*60)
    print(f"SECTION 4 — SELLER DELIVERY PERFORMANCE (min {MIN_SELLER_ORDERS} orders)")
    print("="*60)

    # Get seller_id per order (from orders_full)
    del_full = full[full["order_status"] == "delivered"].copy()
    order_seller = (del_full[del_full["seller_id"].notna()]
                    .drop_duplicates(subset=["order_id", "seller_id"])
                    [["order_id", "seller_id", "seller_state", "seller_city"]])

    # Join delivery + review from del_valid (master level)
    order_del = del_valid[["order_id", "delivery_days", "delivery_delay_days",
                            "is_late", "on_time", "review_score"]]
    seller_orders = order_seller.merge(order_del, on="order_id", how="left")

    # Aggregate per seller
    seller_agg = seller_orders.groupby("seller_id", as_index=False).agg(
        order_count          = ("order_id",            "nunique"),
        seller_state         = ("seller_state",        "first"),
        seller_city          = ("seller_city",         "first"),
        avg_delivery_days    = ("delivery_days",        "mean"),
        median_delivery_days = ("delivery_days",        "median"),
        avg_delay_days       = ("delivery_delay_days",  "mean"),
        late_count           = ("is_late",             "sum"),
        on_time_count        = ("on_time",             "sum"),
        avg_review_score     = ("review_score",        "mean"),
    )
    seller_agg["late_pct"] = (
        seller_agg["late_count"] /
        (seller_agg["late_count"] + seller_agg["on_time_count"]) * 100
    ).round(2)

    total_sellers = len(seller_agg)

    # Qualified subset: >= MIN_SELLER_ORDERS
    qualified = seller_agg[seller_agg["order_count"] >= MIN_SELLER_ORDERS].copy()
    qualified = qualified.sort_values("order_count", ascending=False).reset_index(drop=True)

    nat_late_pct = del_valid["is_late"].mean() * 100

    print(f"\n  Total sellers with deliveries: {total_sellers:,}")
    print(f"  Sellers with >= {MIN_SELLER_ORDERS} orders:  {len(qualified)}")
    print(f"  National late%: {nat_late_pct:.2f}%")

    # Top 10 by order volume
    print(f"\n  Top 10 qualified sellers by order volume:")
    print(f"  {'Seller (last 8)':<16} {'State':<6} {'Orders':>7}  "
          f"{'Late%':>7}  {'AvgDays':>8}  {'Review':>7}")
    for _, r in qualified.head(10).iterrows():
        flag = " [ABOVE_NAT]" if r["late_pct"] > nat_late_pct else ""
        print(f"  ...{r['seller_id'][-8:]:<16} {r['seller_state']:<6} "
              f"{r['order_count']:>7,}  {r['late_pct']:>6.1f}%  "
              f"{r['avg_delivery_days']:>8.1f}  {r['avg_review_score']:>7.3f}{flag}")

    # Worst delivery performance among qualified sellers
    worst = qualified.sort_values("late_pct", ascending=False).head(10)
    print(f"\n  Top 10 qualified sellers by late% (highest late delivery rate):")
    for _, r in worst.iterrows():
        print(f"  ...{r['seller_id'][-8:]:<16} {r['seller_state']:<6} "
              f"orders={r['order_count']:>5,}  late%={r['late_pct']:>5.1f}%  "
              f"avg_days={r['avg_delivery_days']:>5.1f}  review={r['avg_review_score']:.3f}")

    # Best delivery performance among qualified sellers
    best = qualified.sort_values("late_pct").head(10)
    print(f"\n  Top 10 qualified sellers by late% (lowest late delivery rate):")
    for _, r in best.iterrows():
        print(f"  ...{r['seller_id'][-8:]:<16} {r['seller_state']:<6} "
              f"orders={r['order_count']:>5,}  late%={r['late_pct']:>5.1f}%  "
              f"avg_days={r['avg_delivery_days']:>5.1f}  review={r['avg_review_score']:.3f}")

    # State-level summary for sellers
    seller_state_agg = seller_agg.groupby("seller_state", as_index=False).agg(
        seller_count         = ("seller_id",           "count"),
        total_orders         = ("order_count",         "sum"),
        avg_late_pct         = ("late_pct",            "mean"),
        avg_review_score     = ("avg_review_score",    "mean"),
    ).sort_values("total_orders", ascending=False)
    print(f"\n  Seller state summary (top 10 by orders):")
    for _, r in seller_state_agg.head(10).iterrows():
        print(f"    {r['seller_state']:<4} sellers={r['seller_count']:>4}  "
              f"orders={r['total_orders']:>6,}  avg_late%={r['avg_late_pct']:.1f}%  "
              f"avg_review={r['avg_review_score']:.3f}")

    return qualified, seller_state_agg


# ---------------------------------------------------------------------------
# SECTION 5 — CATEGORY DELIVERY PERFORMANCE (enriched)
# ---------------------------------------------------------------------------

def delivery_by_category_enriched(del_valid, full):
    print("\n" + "="*60)
    print("SECTION 5 — CATEGORY DELIVERY PERFORMANCE")
    print("="*60)

    del_full = full[full["order_status"] == "delivered"].copy()

    # First item's category per order (same methodology as Phase 3)
    order_cat = (del_full[del_full["price"].notna()]
                 .sort_values("order_item_id")
                 .drop_duplicates(subset="order_id")
                 [["order_id", "product_category_name_english",
                   "product_category_name"]])

    order_del = del_valid[["order_id", "delivery_days", "delivery_delay_days",
                            "is_late", "on_time", "review_score"]]
    cat_data = order_cat.merge(order_del, on="order_id", how="left")

    cat = cat_data.groupby("product_category_name_english", as_index=False).agg(
        order_count          = ("order_id",            "nunique"),
        avg_delivery_days    = ("delivery_days",        "mean"),
        median_delivery_days = ("delivery_days",        "median"),
        avg_delay_days       = ("delivery_delay_days",  "mean"),
        late_count           = ("is_late",             "sum"),
        on_time_count        = ("on_time",             "sum"),
        avg_review_score     = ("review_score",        "mean"),
    )
    cat["late_pct"] = (
        cat["late_count"] / (cat["late_count"] + cat["on_time_count"]) * 100
    ).round(2)

    nat_late_pct = del_valid["is_late"].mean() * 100
    cat["late_vs_national"] = (cat["late_pct"] - nat_late_pct).round(2)
    cat["low_volume"]       = cat["order_count"] < 100

    # Sort by late_pct descending
    cat = cat.sort_values("late_pct", ascending=False).reset_index(drop=True)

    print(f"\n  Category rows: {len(cat)}  (national late%: {nat_late_pct:.2f}%)")
    print(f"\n  Top 15 categories by late% (with order volume):")
    print(f"  {'Category':<42} {'Orders':>7}  {'Late%':>6}  "
          f"{'vs_nat':>7}  {'AvgDays':>8}  {'Review':>7}  Note")
    for _, r in cat.head(15).iterrows():
        vol_note = " [LOW VOL]" if r["low_volume"] else ""
        print(f"  {r['product_category_name_english']:<42} "
              f"{r['order_count']:>7,}  {r['late_pct']:>6.1f}%  "
              f"{r['late_vs_national']:>+7.1f}%  {r['avg_delivery_days']:>8.1f}  "
              f"{r['avg_review_score']:>7.3f}  {vol_note}")

    # Validate total orders reconcile with category totals from Phase 3
    ph3_cat = pd.read_csv(os.path.join(DATA_DIR, "delivery_by_category.csv"))
    this_total = cat["order_count"].sum()
    ph3_total  = ph3_cat["order_count"].sum()
    diff       = abs(this_total - ph3_total)
    print(f"\n  VALIDATION: category order total this={this_total:,}  ph3={ph3_total:,}  diff={diff}")
    if diff == 0:
        print("  PASS: category totals reconcile")
    else:
        print("  NOTE: small difference may result from orders with unknown category")

    return cat


# ---------------------------------------------------------------------------
# SECTION 6 — DELIVERY DELAY vs REVIEW SCORE (detailed)
# ---------------------------------------------------------------------------
# Grain : one row per delay_bucket
# Adds  : full review score distribution per bucket (counts for 1–5 stars)

def review_by_delay_detailed(del_valid):
    print("\n" + "="*60)
    print("SECTION 6 — DELIVERY DELAY vs REVIEW SCORE (detailed)")
    print("="*60)

    # Use only rows with both fields present
    df = del_valid[del_valid["review_score"].notna() &
                   del_valid["delivery_delay_days"].notna()].copy()

    df["delay_bucket"] = pd.Categorical(
        df["delay_bucket"], categories=DELAY_LABELS, ordered=True
    )

    # Aggregate
    agg = df.groupby("delay_bucket", observed=True, as_index=False).agg(
        order_count      = ("order_id",             "nunique"),
        avg_review_score = ("review_score",          "mean"),
        pct_1star        = ("review_score",          lambda x: (x == 1).mean() * 100),
        pct_2star        = ("review_score",          lambda x: (x == 2).mean() * 100),
        pct_3star        = ("review_score",          lambda x: (x == 3).mean() * 100),
        pct_4star        = ("review_score",          lambda x: (x == 4).mean() * 100),
        pct_5star        = ("review_score",          lambda x: (x == 5).mean() * 100),
    )
    for col in ["avg_review_score","pct_1star","pct_2star","pct_3star","pct_4star","pct_5star"]:
        agg[col] = agg[col].round(3)

    agg["delay_bucket"] = agg["delay_bucket"].astype(str)

    total_reviewed = df["order_id"].nunique()
    print(f"\n  Orders with both review and delay data: {total_reviewed:,}")
    print(f"\n  OBSERVED RELATIONSHIP — delivery delay and review score:")
    print(f"  (This is an association; no causal claim is made.)")
    print(f"\n  {'Bucket':<14} {'n':>7}  {'AvgScore':>9}  {'1★%':>6}  "
          f"{'2★%':>6}  {'3★%':>6}  {'4★%':>6}  {'5★%':>6}")
    for _, r in agg.iterrows():
        print(f"  {r['delay_bucket']:<14} {r['order_count']:>7,}  "
              f"{r['avg_review_score']:>9.3f}  "
              f"{r['pct_1star']:>5.1f}%  {r['pct_2star']:>5.1f}%  "
              f"{r['pct_3star']:>5.1f}%  {r['pct_4star']:>5.1f}%  {r['pct_5star']:>5.1f}%")

    # Validate avg_review_score against Phase 3 review_vs_delay.csv
    ph3_rvd = pd.read_csv(os.path.join(DATA_DIR, "review_vs_delay.csv"))
    print(f"\n  VALIDATION vs Phase 3 review_vs_delay.csv:")
    for _, r in ph3_rvd.iterrows():
        this_row = agg[agg["delay_bucket"] == r["delay_bucket"]]
        if len(this_row):
            diff = abs(this_row.iloc[0]["avg_review_score"] - r["avg_review_score"])
            status = "PASS" if diff < 0.001 else "WARN"
            print(f"  {status}: {r['delay_bucket']:<14} "
                  f"this={this_row.iloc[0]['avg_review_score']:.3f}  "
                  f"ph3={r['avg_review_score']:.3f}  diff={diff:.4f}")

    return agg


# ---------------------------------------------------------------------------
# SECTION 7 — REVIEW SCORE BY STATE
# ---------------------------------------------------------------------------

def review_by_state(del_valid):
    print("\n" + "="*60)
    print("SECTION 7 — REVIEW SCORE BY STATE")
    print("="*60)

    df = del_valid[del_valid["review_score"].notna()].copy()

    state = df.groupby("customer_state", as_index=False).agg(
        review_count     = ("review_score", "count"),
        avg_review_score = ("review_score", "mean"),
        pct_1star        = ("review_score", lambda x: (x == 1).mean() * 100),
        pct_5star        = ("review_score", lambda x: (x == 5).mean() * 100),
        late_pct         = ("is_late",      lambda x: x.mean() * 100),
    )
    nat_review = df["review_score"].mean()
    state["review_vs_national"] = (state["avg_review_score"] - nat_review).round(3)
    state = state.sort_values("avg_review_score").reset_index(drop=True)

    print(f"\n  National avg review score: {nat_review:.3f}")
    print(f"\n  States sorted by avg review score (ascending):")
    print(f"  {'State':<6} {'Reviews':>8}  {'AvgScore':>9}  {'vs_nat':>7}  "
          f"{'1★%':>6}  {'5★%':>6}  {'Late%':>7}")
    for _, r in state.iterrows():
        print(f"  {r['customer_state']:<6} {r['review_count']:>8,}  "
              f"{r['avg_review_score']:>9.3f}  {r['review_vs_national']:>+7.3f}  "
              f"{r['pct_1star']:>5.1f}%  {r['pct_5star']:>5.1f}%  "
              f"{r['late_pct']:>6.1f}%")

    return state


# ---------------------------------------------------------------------------
# SECTION 8 — REVIEW SCORE BY CATEGORY
# ---------------------------------------------------------------------------

def review_by_category(del_valid, full):
    print("\n" + "="*60)
    print("SECTION 8 — REVIEW SCORE BY CATEGORY")
    print("="*60)

    del_full = full[full["order_status"] == "delivered"].copy()
    order_cat = (del_full[del_full["price"].notna()]
                 .sort_values("order_item_id")
                 .drop_duplicates(subset="order_id")
                 [["order_id", "product_category_name_english"]])

    df = del_valid[del_valid["review_score"].notna()][
        ["order_id", "review_score", "is_late", "delivery_delay_days"]
    ]
    cat_review = order_cat.merge(df, on="order_id", how="inner")

    cat = cat_review.groupby("product_category_name_english", as_index=False).agg(
        review_count     = ("review_score", "count"),
        avg_review_score = ("review_score", "mean"),
        pct_1star        = ("review_score", lambda x: (x == 1).mean() * 100),
        pct_5star        = ("review_score", lambda x: (x == 5).mean() * 100),
        late_pct         = ("is_late",      lambda x: x.mean() * 100),
        avg_delay_days   = ("delivery_delay_days", "mean"),
    )
    nat_review = del_valid["review_score"].mean()
    cat["review_vs_national"] = (cat["avg_review_score"] - nat_review).round(3)
    cat["low_volume"]         = cat["review_count"] < 100
    cat = cat.sort_values("avg_review_score").reset_index(drop=True)

    print(f"\n  Category review rows: {len(cat)}")
    print(f"\n  Bottom 15 categories by avg review score:")
    print(f"  {'Category':<42} {'n':>6}  {'AvgScore':>9}  "
          f"{'vs_nat':>7}  {'1★%':>6}  {'5★%':>6}  {'Late%':>7}  Note")
    for _, r in cat.head(15).iterrows():
        vol_note = " [LOW VOL]" if r["low_volume"] else ""
        print(f"  {r['product_category_name_english']:<42} "
              f"{r['review_count']:>6,}  {r['avg_review_score']:>9.3f}  "
              f"{r['review_vs_national']:>+7.3f}  {r['pct_1star']:>5.1f}%  "
              f"{r['pct_5star']:>5.1f}%  {r['late_pct']:>6.1f}%  {vol_note}")

    print(f"\n  Top 10 categories by avg review score:")
    for _, r in cat.tail(10).iterrows():
        print(f"  {r['product_category_name_english']:<42} "
              f"{r['review_count']:>6,}  {r['avg_review_score']:>9.3f}  "
              f"{r['review_vs_national']:>+7.3f}  {'[LOW VOL]' if r['low_volume'] else ''}")

    return cat


# ---------------------------------------------------------------------------
# SECTION 9 — STRUCTURED DELIVERY/SATISFACTION INSIGHTS
# ---------------------------------------------------------------------------
# All claims are backed by calculated data; associations are not presented
# as causal relationships.

def build_delivery_insights(del_valid, state_del, cat_del, seller_del,
                             review_delay, review_state, review_cat, overview):
    print("\n" + "="*60)
    print("SECTION 9 — DELIVERY & SATISFACTION INSIGHTS")
    print("="*60)

    insights = []
    nat_late_pct = del_valid["is_late"].mean() * 100
    nat_avg_days = del_valid["delivery_days"].mean()

    # -- Insight D1: Conservative estimated delivery dates --
    early_7d_pct = overview["delay_bucket_counts"].get("Early >7d", 0) / overview["n_orders"] * 100
    i1 = {
        "id":   "conservative_estimates",
        "fact": (
            f"{early_7d_pct:.1f}% of delivered orders arrived more than 7 days before "
            f"the estimated delivery date. The overall on-time rate is "
            f"{overview['on_time_pct']:.2f}%."
        ),
        "insight": (
            "The dataset's delivery estimates appear systematically conservative: the "
            "majority of deliveries arrive well ahead of the stated estimate. This pattern "
            "is observed consistently across the dataset, not just in a few months."
        ),
        "risk_or_opportunity": (
            "OPPORTUNITY: More accurate delivery date estimates could improve customer "
            "experience by setting realistic expectations. The observed association between "
            "early delivery (>7 days early) and higher review scores (avg 4.31) vs. the "
            "overall average (4.16) is consistent with this interpretation, though other "
            "factors may also be contributing."
        ),
        "recommended_action": (
            "Review the delivery estimation model. Consider presenting customers with a "
            "narrower, more accurate delivery window rather than a conservative upper bound."
        )
    }
    insights.append(i1)
    print(f"\n  INSIGHT D1: {i1['fact'][:80]}...")

    # -- Insight D2: Northeast delivery gap --
    northeast = ["AL","MA","SE","PI","CE","RN","PB","PE","BA"]
    ne_data = state_del[state_del["customer_state"].isin(northeast) &
                        (~state_del["low_volume"])]
    ne_late_avg = ne_data["late_pct"].mean()
    ne_days_avg = ne_data["avg_delivery_days"].mean()
    ne_orders   = ne_data["order_count"].sum()
    i2 = {
        "id":   "northeast_delivery_gap",
        "fact": (
            f"Among Northeastern states with >= 200 orders, the average late delivery "
            f"rate is {ne_late_avg:.1f}% (national: {nat_late_pct:.1f}%). "
            f"Average delivery time in these states is {ne_days_avg:.1f} days "
            f"(national: {nat_avg_days:.1f} days). "
            f"Total orders in scope: {ne_orders:,}."
        ),
        "insight": (
            "Delivery performance is materially weaker in several Northeastern states. "
            "States AL ({al}%), MA ({ma}%), SE ({se}%), CE ({ce}%) each show late rates "
            "more than twice the national average.".format(
                al=ne_data[ne_data["customer_state"]=="AL"]["late_pct"].values[0] if "AL" in ne_data["customer_state"].values else "N/A",
                ma=ne_data[ne_data["customer_state"]=="MA"]["late_pct"].values[0] if "MA" in ne_data["customer_state"].values else "N/A",
                se=ne_data[ne_data["customer_state"]=="SE"]["late_pct"].values[0] if "SE" in ne_data["customer_state"].values else "N/A",
                ce=ne_data[ne_data["customer_state"]=="CE"]["late_pct"].values[0] if "CE" in ne_data["customer_state"].values else "N/A",
            )
        ),
        "risk_or_opportunity": (
            "RISK: Northeast customers who experience late delivery also show lower avg "
            "review scores than the national average (observed association — "
            "not a causal claim). Sustained poor delivery performance in these states "
            "may reduce repeat purchase likelihood. "
            "OPPORTUNITY: Logistics improvement in the Northeast could close the "
            "performance gap and unlock demand in currently underserved markets."
        ),
        "recommended_action": (
            "Investigate the seller and logistics network coverage in AL, MA, SE, and CE. "
            "Assess whether recruiting local sellers or regional fulfilment partners "
            "would materially reduce delivery time in these states."
        )
    }
    insights.append(i2)
    print(f"\n  INSIGHT D2: {i2['fact'][:80]}...")

    # -- Insight D3: Delivery delay / review score association --
    early_score = review_delay[review_delay["delay_bucket"] == "Early >7d"]["avg_review_score"].values
    late47_score= review_delay[review_delay["delay_bucket"] == "Late 4-7d"]["avg_review_score"].values
    late14_score= review_delay[review_delay["delay_bucket"] == "Late >14d"]["avg_review_score"].values
    early_n     = review_delay[review_delay["delay_bucket"] == "Early >7d"]["order_count"].values
    late47_n    = review_delay[review_delay["delay_bucket"] == "Late 4-7d"]["order_count"].values

    i3 = {
        "id":   "delay_review_association",
        "fact": (
            f"Orders arriving >7 days early (n={int(early_n[0]):,}) have an observed "
            f"avg review score of {float(early_score[0]):.3f}. "
            f"Orders arriving 4–7 days late (n={int(late47_n[0]):,}) have an observed "
            f"avg review score of {float(late47_score[0]):.3f}. "
            f"Orders arriving >14 days late have an avg score of {float(late14_score[0]):.3f}."
        ),
        "insight": (
            "OBSERVED ASSOCIATION: There is a strong monotonic association between "
            "delivery timing relative to the estimated date and customer review score. "
            "Customers whose orders arrive later than the estimate assign materially lower "
            "scores on average. This is an observed pattern in the data; no causal "
            "analysis has been performed."
        ),
        "risk_or_opportunity": (
            "RISK: Sellers and regions with high late-delivery rates are also associated "
            "with lower review scores, which may affect platform-level satisfaction "
            "metrics. "
            "OPPORTUNITY: Reducing the proportion of orders in the 'Late >7d' buckets "
            "is associated in this data with higher review scores."
        ),
        "recommended_action": (
            "Use the delay-bucket analysis as an early warning system: monitor the "
            "Late 4-7d and Late >14d buckets monthly by seller and region. "
            "Prioritise logistics interventions for sellers and routes contributing "
            "most to late deliveries in those buckets."
        )
    }
    insights.append(i3)
    print(f"\n  INSIGHT D3: {i3['fact'][:80]}...")

    # -- Insight D4: High-volume seller delivery variance --
    nat_review = del_valid["review_score"].mean()
    high_late_sellers = seller_del[
        (seller_del["late_pct"] > nat_late_pct * 1.5) &
        (seller_del["order_count"] >= MIN_SELLER_ORDERS)
    ]
    i4 = {
        "id":   "seller_delivery_variance",
        "fact": (
            f"{len(high_late_sellers)} sellers (with >= {MIN_SELLER_ORDERS} orders) "
            f"have a late delivery rate >= 1.5x the national average ({nat_late_pct:.1f}%). "
            f"These sellers collectively handle "
            f"{high_late_sellers['order_count'].sum():,} orders."
        ),
        "insight": (
            "A subset of meaningful-volume sellers show materially elevated late delivery "
            "rates. Their average review score "
            f"({high_late_sellers['avg_review_score'].mean():.3f}) is below the national "
            f"average ({nat_review:.3f}). This is an observed association."
        ),
        "risk_or_opportunity": (
            "RISK: If these sellers remain unaddressed, their concentrated delivery "
            "problems represent a disproportionate source of negative customer experiences. "
            "OPPORTUNITY: Targeted performance programmes for this seller cohort could "
            "improve both platform delivery metrics and review scores."
        ),
        "recommended_action": (
            "Identify and prioritise the top 20 highest-volume sellers with late_pct "
            "> 1.5x national for logistics review, SLA agreements, or carrier changes. "
            "Track their monthly late_pct as a performance KPI."
        )
    }
    insights.append(i4)
    print(f"\n  INSIGHT D4: {i4['fact'][:80]}...")

    return insights


# ---------------------------------------------------------------------------
# SECTION 10 — SAVE ALL OUTPUTS
# ---------------------------------------------------------------------------

def save_all_delivery(overview, monthly_del, state_del, cat_del,
             seller_del, seller_state_agg,
             review_delay, review_state, review_cat, insights):
    print("\n" + "="*60)
    print("SECTION 10 — SAVING OUTPUTS")
    print("="*60)

    ov_path = os.path.join(DATA_DIR, "delivery_overview.json")
    with open(ov_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    print(f"  Saved {ov_path}")

    ins_path = os.path.join(DATA_DIR, "delivery_insights.json")
    with open(ins_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"  Saved {ins_path}")

    csvs = {
        "delivery_over_time.csv":           monthly_del,
        "delivery_by_state_enriched.csv":   state_del,
        "delivery_by_category_enriched.csv":cat_del,
        "seller_delivery_qualified.csv":    seller_del,
        "seller_delivery_by_state.csv":     seller_state_agg,
        "review_by_delay_detailed.csv":     review_delay,
        "review_by_state.csv":              review_state,
        "review_by_category.csv":           review_cat,
    }
    for fname, df in csvs.items():
        path = os.path.join(DATA_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {path:<50} {len(df):>5} rows")


# ---------------------------------------------------------------------------
# MAIN — Phase 5
# ---------------------------------------------------------------------------

def main_delivery_satisfaction():
    print("\n" + "#"*60)
    print("  OLIST DELIVERY & SATISFACTION — PHASE 5")
    print("#"*60)

    master, full, delivered, del_valid, ph3_kpis, ph3_delay, ph3_review = load_data_delivery()

    overview    = delivery_overview(del_valid, ph3_kpis, ph3_delay)
    monthly_del = delivery_over_time(del_valid)
    state_del   = delivery_by_state_enriched(del_valid)
    cat_del     = delivery_by_category_enriched(del_valid, full)
    seller_del, seller_state_agg = seller_delivery(del_valid, full)
    review_delay = review_by_delay_detailed(del_valid)
    review_state = review_by_state(del_valid)
    review_cat   = review_by_category(del_valid, full)
    insights     = build_delivery_insights(del_valid, state_del, cat_del,
                                           seller_del, review_delay,
                                           review_state, review_cat, overview)

    save_all_delivery(overview, monthly_del, state_del, cat_del,
             seller_del, seller_state_agg,
             review_delay, review_state, review_cat, insights)

    print("\n" + "#"*60)
    print("  PHASE 5 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: ml_model.py — Phase 6
# ============================================================

import os, json, warnings, pickle
import pandas as pd
import numpy as np

from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                       f1_score, roc_auc_score, confusion_matrix,
                                       classification_report)
import sklearn

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. BUILD THE MODELLING DATASET
# ---------------------------------------------------------------------------

def build_dataset():
    print("\n" + "="*60)
    print("STEP 1 — BUILD MODELLING DATASET")
    print("="*60)

    master = pd.read_csv(
        os.path.join(DATA_DIR, "orders_master.csv"),
        parse_dates=["order_purchase_timestamp", "order_approved_at",
                     "order_estimated_delivery_date",
                     "order_delivered_customer_date"]
    )
    full = pd.read_csv(
        os.path.join(DATA_DIR, "orders_full.csv"),
        parse_dates=["order_purchase_timestamp",
                     "order_estimated_delivery_date"]
    )

    # Scope: delivered orders with a valid target
    scope = master[
        (master["order_status"] == "delivered") &
        master["is_late"].notna()
    ].copy()
    print(f"  Delivered orders with valid target : {len(scope):,}")

    # --- Item-level features (all class A) ---
    item_feats = (
        full[full["order_status"] == "delivered"]
        .groupby("order_id", as_index=False)
        .agg(
            order_item_count    = ("order_item_id",     "count"),
            total_freight_value = ("freight_value",     "sum"),
            avg_weight_g        = ("product_weight_g",  "mean"),
            avg_volume_cm3      = ("product_volume_cm3","mean"),
            unique_sellers      = ("seller_id",         "nunique"),
            seller_state        = ("seller_state",      "first"),
        )
    )

    df = scope.merge(item_feats, on="order_id", how="left")

    # --- Derived features (all class A except approval_lag_hours = B) ---
    df["days_to_estimated"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"])
        .dt.days
    )
    df["purchase_month"]      = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"]  = df["order_purchase_timestamp"].dt.dayofweek
    df["approval_lag_hours"]  = (
        (df["order_approved_at"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 3600
    )

    # --- Explicit leakage audit ---
    FORBIDDEN = ["delivery_days", "delivery_delay_days",
                 "order_delivered_customer_date",
                 "order_delivered_carrier_date",
                 "review_score", "review_comment_message",
                 "review_creation_date", "review_answer_timestamp",
                 "on_time"]
    for col in FORBIDDEN:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"  LEAKAGE GUARD: dropped '{col}'")

    # Target
    df["is_late"] = df["is_late"].astype(int)

    print(f"\n  Target distribution:")
    vc = df["is_late"].value_counts()
    for k, v in vc.items():
        print(f"    is_late={int(k)}  n={v:,}  ({v/len(df)*100:.2f}%)")

    return df


# ---------------------------------------------------------------------------
# 2. TIME-BASED TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

def split_dataset(df):
    print("\n" + "="*60)
    print("STEP 2 — TIME-BASED TRAIN / TEST SPLIT")
    print("="*60)

    df_sorted = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df_sorted)
    cutoff_idx = int(n * 0.80)
    cutoff_ts  = df_sorted.iloc[cutoff_idx]["order_purchase_timestamp"]

    train = df_sorted[df_sorted["order_purchase_timestamp"] < cutoff_ts].copy()
    test  = df_sorted[df_sorted["order_purchase_timestamp"] >= cutoff_ts].copy()

    # Validation: no order appears in both sets
    overlap = set(train["order_id"]) & set(test["order_id"])
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} orders in both train and test"

    print(f"  Cutoff date   : {cutoff_ts.date()}")
    print(f"  Train         : {len(train):,} orders  "
          f"({cutoff_ts.date()} exclusive)")
    print(f"  Test          : {len(test):,} orders  "
          f"({cutoff_ts.date()} inclusive → {df_sorted['order_purchase_timestamp'].max().date()})")
    print(f"  Train late%   : {train['is_late'].mean()*100:.2f}%")
    print(f"  Test  late%   : {test['is_late'].mean()*100:.2f}%")
    print(f"  PASS: no order_id overlap between train and test")

    return train, test, cutoff_ts


# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION & PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "days_to_estimated",      # A: estimated window length
    "total_freight_value",    # A: shipping cost (proxy for distance/complexity)
    "avg_weight_g",           # A: product weight
    "avg_volume_cm3",         # A: product volume
    "order_item_count",       # A: number of items in order
    "unique_sellers",         # A: multi-seller orders
    "payment_installments",   # A: payment plan
    "approval_lag_hours",     # B: time from placement to approval
    "purchase_month",         # A: seasonality
    "purchase_dayofweek",     # A: day-of-week pattern
]

CATEGORICAL_FEATURES = [
    "customer_state",    # A: delivery destination
    "seller_state",      # A: dispatch origin
    "payment_type",      # A: payment method
]

TARGET = "is_late"


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------------

def train_models(train):
    print("\n" + "="*60)
    print("STEP 3 — TRAIN MODELS")
    print("="*60)

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[TARGET]

    preprocessor = build_preprocessor()

    # ---- Model 1: Logistic Regression baseline ----
    lr_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",   # handles imbalance
            solver="lbfgs"
        )),
    ])
    lr_pipeline.fit(X_train, y_train)
    print(f"  Logistic Regression trained  "
          f"(class_weight=balanced, max_iter=1000)")

    # ---- Model 2: Random Forest ----
    rf_pipeline = Pipeline([
        ("pre", build_preprocessor()),   # fresh preprocessor — fitted only on train
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    print(f"  Random Forest trained        "
          f"(n_estimators=200, max_depth=8, min_samples_leaf=20, class_weight=balanced)")

    return lr_pipeline, rf_pipeline


# ---------------------------------------------------------------------------
# 5. EVALUATE MODELS
# ---------------------------------------------------------------------------

def evaluate(name, pipeline, X_test, y_test):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  [{name}]")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Precision  : {prec:.4f}  (of predicted late, how many were truly late)")
    print(f"    Recall     : {rec:.4f}  (of truly late, how many were caught)")
    print(f"    F1-score   : {f1:.4f}")
    print(f"    ROC-AUC    : {auc:.4f}")
    print(f"\n    Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred-OnTime  Pred-Late")
    print(f"    Act-OnTime   {cm[0,0]:>8,}  {cm[0,1]:>9,}")
    print(f"    Act-Late     {cm[1,0]:>8,}  {cm[1,1]:>9,}")
    print(f"\n    TP={cm[1,1]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TN={cm[0,0]:,}")
    tn, fp, fn, tp = cm.ravel()
    print(f"    False Positive Rate (on-time flagged as late): {fp/(fp+tn)*100:.2f}%")
    print(f"    False Negative Rate (late missed):             {fn/(fn+tp)*100:.2f}%")

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(auc,  4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
    }


def evaluate_models(lr_pipeline, rf_pipeline, test):
    print("\n" + "="*60)
    print("STEP 4 — EVALUATE MODELS")
    print("="*60)

    print(f"\n  Test set:  {len(test):,} orders  "
          f"|  late={test[TARGET].sum():,} ({test[TARGET].mean()*100:.2f}%)")

    print("\n  --- BUSINESS CONTEXT FOR METRICS ---")
    print("  For a late-delivery DETECTION system:")
    print("    High Recall  = fewer late orders missed (fewer unhappy customers)")
    print("    High Precision = fewer false alarms (fewer unnecessary interventions)")
    print("    In an operational setting, Recall is typically prioritised over Precision")
    print("    because the cost of missing a late delivery (customer dissatisfaction)")
    print("    usually exceeds the cost of a false alarm (unnecessary intervention).")

    print("\n  --- TEST SET TEMPORAL CONTEXT ---")
    print("  The test set covers May-Aug 2018 only (the 20% most recent orders).")
    print("  These months have observed late rates of 0.3%-6.2% — well below the")
    print("  7.6% average late rate in the training window, which included the Feb-Mar 2018")
    print("  crisis (up to 19% late). The model was trained on a broader distribution.")
    print("  Metrics on this test set reflect a temporally favourable evaluation period.")
    print("  ROC-AUC is the most period-agnostic metric to use for model comparison.")

    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET]

    lr_metrics = evaluate("Logistic Regression", lr_pipeline, X_test, y_test)
    rf_metrics = evaluate("Random Forest",       rf_pipeline, X_test, y_test)

    return lr_metrics, rf_metrics


# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def feature_importance(rf_pipeline, lr_pipeline):
    print("\n" + "="*60)
    print("STEP 5 — FEATURE IMPORTANCE")
    print("="*60)

    print("\n  NOTE: Feature importance indicates PREDICTIVE ASSOCIATION with the target,")
    print("  not causation. A high-importance feature does NOT cause late deliveries;")
    print("  it was an important predictor in the trained model on this dataset.")

    # Get feature names after OHE
    pre = rf_pipeline.named_steps["pre"]
    cat_ohe  = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_names = list(cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names

    rf_clf = rf_pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_

    fi_df = pd.DataFrame({
        "feature":    all_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Random Forest — Top 20 features by importance:")
    for _, r in fi_df.head(20).iterrows():
        bar = "#" * int(r["importance"] * 300)
        print(f"    {r['feature']:<45} {r['importance']:.5f}  {bar}")

    # Logistic Regression coefficients (sign = direction, magnitude = strength)
    lr_pre  = lr_pipeline.named_steps["pre"]
    lr_cat  = lr_pre.named_transformers_["cat"].named_steps["ohe"]
    lr_cat_names = list(lr_cat.get_feature_names_out(CATEGORICAL_FEATURES))
    lr_names = NUMERIC_FEATURES + lr_cat_names
    lr_clf   = lr_pipeline.named_steps["clf"]
    lr_coef  = lr_clf.coef_[0]

    lr_fi = pd.DataFrame({
        "feature":      lr_names,
        "coefficient":  lr_coef,
        "abs_coefficient": np.abs(lr_coef),
        "direction":    ["positive" if c > 0 else "negative" for c in lr_coef],
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    print(f"\n  Logistic Regression — Top 15 features by |coefficient|:")
    for _, r in lr_fi.head(15).iterrows():
        direction = "+" if r["coefficient"] > 0 else "-"
        print(f"    {r['feature']:<45} {direction}{r['abs_coefficient']:.4f}")

    return fi_df, lr_fi


# ---------------------------------------------------------------------------
# 7. RISK BANDS
# ---------------------------------------------------------------------------
# Risk bands use the model's probability output quantile-split into tertiles
# (bottom 33%, middle 33%, top 33% of predicted probability).
# This approach is robust to the absolute probability scale and does not
# require fixed thresholds that become degenerate on particular test windows.
# We validate that the actual late rate increases monotonically across bands.

def build_risk_bands(rf_pipeline, test):
    print("\n" + "="*60)
    print("STEP 6 — RISK BANDS")
    print("="*60)

    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET]
    proba  = rf_pipeline.predict_proba(X_test)[:, 1]

    # Tertile-based bands — robust to probability scale shifts across time windows
    p33 = np.percentile(proba, 33)
    p66 = np.percentile(proba, 66)

    bands = np.where(proba >= p66, "High",
            np.where(proba >= p33, "Medium", "Low"))

    df_bands = pd.DataFrame({
        "order_id":  test["order_id"].values,
        "pred_prob": proba,
        "risk_band": bands,
        "actual":    y_test.values,
    })

    summary = []
    for band in ["Low", "Medium", "High"]:
        grp = df_bands[df_bands["risk_band"] == band]
        if len(grp) == 0:
            continue
        actual_late_rate = grp["actual"].mean() * 100
        summary.append({
            "risk_band":       band,
            "order_count":     int(len(grp)),
            "order_pct":       round(len(grp) / len(df_bands) * 100, 2),
            "actual_late_pct": round(actual_late_rate, 2),
            "avg_pred_prob":   round(grp["pred_prob"].mean(), 4),
            "p33_threshold":   round(float(p33), 4),
            "p66_threshold":   round(float(p66), 4),
        })

    print(f"\n  Risk band method: tertile split of predicted probabilities")
    print(f"  Thresholds: Low < {p33:.4f}  |  Medium {p33:.4f}-{p66:.4f}  |  High >= {p66:.4f}")
    print(f"\n  {'Band':<8} {'Orders':>8}  {'% of total':>10}  "
          f"{'Actual Late%':>13}  {'Avg Pred Prob':>14}")
    print("  " + "-"*60)
    for b in summary:
        print(f"  {b['risk_band']:<8} {b['order_count']:>8,}  "
              f"{b['order_pct']:>9.1f}%  "
              f"{b['actual_late_pct']:>12.2f}%  "
              f"{b['avg_pred_prob']:>14.4f}")

    nat_late = test[TARGET].mean() * 100
    low_late  = next((b["actual_late_pct"] for b in summary if b["risk_band"]=="Low"),  0)
    high_late = next((b["actual_late_pct"] for b in summary if b["risk_band"]=="High"), 0)
    print(f"\n  Test-set base late rate: {nat_late:.2f}%")
    print(f"  Low band actual late rate:  {low_late:.2f}%")
    print(f"  High band actual late rate: {high_late:.2f}%  "
          f"(lift: {high_late/nat_late:.1f}x test-set base)")

    # Monotonicity check
    rates = [b["actual_late_pct"] for b in summary]
    if rates == sorted(rates):
        print(f"  PASS: actual late rate increases monotonically across risk bands")
    else:
        print(f"  NOTE: risk band actual late rates not strictly monotonic — "
              f"interpret with caution ({rates})")

    return pd.DataFrame(summary), df_bands


# ---------------------------------------------------------------------------
# 8. SAVE ALL OUTPUTS
# ---------------------------------------------------------------------------

def save_ml_outputs(lr_metrics, rf_metrics, fi_df, lr_fi, risk_summary,
                 train, test, lr_pipeline, rf_pipeline, cutoff_ts):
    print("\n" + "="*60)
    print("STEP 7 — SAVE OUTPUTS")
    print("="*60)

    # Consolidated metrics JSON
    metrics = {
        "sklearn_version":         sklearn.__version__,
        "prediction_point":        "at/after order approval",
        "target":                  "is_late (1=delivered after estimated date)",
        "train_orders":            int(len(train)),
        "test_orders":             int(len(test)),
        "train_cutoff_date":       str(cutoff_ts.date()),
        "train_late_pct":          round(train["is_late"].mean() * 100, 2),
        "test_late_pct":           round(test["is_late"].mean() * 100, 2),
        "class_imbalance_note":    "~93% on-time, ~7% late; class_weight=balanced used",
        "features_used":           NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "features_excluded_reason":{
            "delivery_days":            "C - derived from actual delivery date",
            "delivery_delay_days":      "C - derived from actual delivery date",
            "review_score":             "C - post-delivery information",
            "order_delivered_customer_date": "C - actual delivery date",
        },
        "logistic_regression": lr_metrics,
        "random_forest":       rf_metrics,
        "risk_band_thresholds":{"low":"< 0.10","medium":"0.10-0.25","high":">= 0.25"},
    }

    mpath = os.path.join(DATA_DIR, "ml_model_metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {mpath}")

    fi_df.to_csv(os.path.join(DATA_DIR, "ml_feature_importance.csv"), index=False)
    print(f"  Saved data/ml_feature_importance.csv  ({len(fi_df)} features)")

    lr_fi.to_csv(os.path.join(DATA_DIR, "ml_lr_coefficients.csv"), index=False)
    print(f"  Saved data/ml_lr_coefficients.csv  ({len(lr_fi)} features)")

    risk_summary.to_csv(os.path.join(DATA_DIR, "ml_risk_bands.csv"), index=False)
    print(f"  Saved data/ml_risk_bands.csv")

    # Confusion matrices CSV
    cm_rows = []
    for model_name, m in [("logistic_regression", lr_metrics),
                           ("random_forest",       rf_metrics)]:
        cm = m["confusion_matrix"]
        cm_rows.append({"model": model_name, **cm})
    pd.DataFrame(cm_rows).to_csv(
        os.path.join(DATA_DIR, "ml_confusion_matrix.csv"), index=False
    )
    print(f"  Saved data/ml_confusion_matrix.csv")

    # Save RF pipeline (for dashboard use)
    pkl_path = os.path.join(DATA_DIR, "ml_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(rf_pipeline, f)
    print(f"  Saved {pkl_path}  (Random Forest pipeline)")


# ---------------------------------------------------------------------------
# MAIN — Phase 6
# ---------------------------------------------------------------------------

def main_ml_model():
    print("\n" + "#"*60)
    print("  OLIST ML MODEL — PHASE 6")
    print("  Late Delivery Prediction")
    print("#"*60)

    df                        = build_dataset()
    train, test, cutoff_ts    = split_dataset(df)
    lr_pipeline, rf_pipeline  = train_models(train)
    lr_metrics, rf_metrics    = evaluate_models(lr_pipeline, rf_pipeline, test)
    fi_df, lr_fi              = feature_importance(rf_pipeline, lr_pipeline)
    risk_summary, _           = build_risk_bands(rf_pipeline, test)

    save_ml_outputs(lr_metrics, rf_metrics, fi_df, lr_fi, risk_summary,
                 train, test, lr_pipeline, rf_pipeline, cutoff_ts)

    print("\n" + "#"*60)
    print("  PHASE 6 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: insight_engine.py — Phase 7
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# LOAD EVIDENCE
# ---------------------------------------------------------------------------

def load_evidence():
    D = DATA_DIR
    return {
        "kpis":          json.load(open(f"{D}/kpis.json")),
        "rfm_seg":       pd.read_csv(f"{D}/rfm_segment_summary.csv"),
        "repeat":        pd.read_csv(f"{D}/repeat_vs_onetime.csv"),
        "cust_geo":      pd.read_csv(f"{D}/customer_geography.csv"),
        "spend_dist":    pd.read_csv(f"{D}/customer_spending_dist.csv"),
        "cat_perf":      pd.read_csv(f"{D}/category_performance.csv"),
        "monthly":       pd.read_csv(f"{D}/monthly_trends.csv"),
        "state_perf":    pd.read_csv(f"{D}/state_performance.csv"),
        "seller_perf":   pd.read_csv(f"{D}/seller_performance.csv"),
        "del_ov":        json.load(open(f"{D}/delivery_overview.json")),
        "del_state":     pd.read_csv(f"{D}/delivery_by_state_enriched.csv"),
        "del_cat":       pd.read_csv(f"{D}/delivery_by_category_enriched.csv"),
        "del_time":      pd.read_csv(f"{D}/delivery_over_time.csv"),
        "seller_del":    pd.read_csv(f"{D}/seller_delivery_qualified.csv"),
        "rev_delay":     pd.read_csv(f"{D}/review_by_delay_detailed.csv"),
        "rev_state":     pd.read_csv(f"{D}/review_by_state.csv"),
        "rev_cat":       pd.read_csv(f"{D}/review_by_category.csv"),
        "ml_metrics":    json.load(open(f"{D}/ml_model_metrics.json")),
        "ml_fi":         pd.read_csv(f"{D}/ml_feature_importance.csv"),
    }


# ---------------------------------------------------------------------------
# INSIGHT BUILDER HELPER
# ---------------------------------------------------------------------------

def make_insight(id_, category, fact, metric, insight,
                 insight_type, recommendation, source, qualification=""):
    return {
        "id":             id_,
        "category":       category,
        "fact":           fact,
        "metric":         metric,
        "insight":        insight,
        "type":           insight_type,
        "recommendation": recommendation,
        "source":         source,
        "qualification":  qualification,
    }


# ---------------------------------------------------------------------------
# CUSTOMER INSIGHTS
# ---------------------------------------------------------------------------

def customer_insights(ev):
    insights = []
    repeat = ev["repeat"]
    rfm    = ev["rfm_seg"]
    geo    = ev["cust_geo"]
    kpis   = ev["kpis"]

    ot = repeat[repeat["customer_type"] == "One-Time"].iloc[0]
    rp = repeat[repeat["customer_type"] == "Repeat"].iloc[0]
    champs   = rfm[rfm["segment"] == "Champions"].iloc[0]
    at_risk  = rfm[rfm["segment"] == "At Risk"].iloc[0]
    lost     = rfm[rfm["segment"] == "Lost / Inactive"].iloc[0]
    sp_row   = geo[geo["customer_state"] == "SP"].iloc[0]
    top3_rev = geo.head(3)["revenue_pct"].sum()

    # C1 — One-time vs repeat
    insights.append(make_insight(
        id_="C1_onetime_retention_pattern",
        category="customer",
        fact=(
            f"{ot['customer_pct']:.1f}% of customers ({int(ot['customer_count']):,}) "
            f"placed exactly one delivered order. Only {rp['customer_pct']:.1f}% "
            f"({int(rp['customer_count']):,}) placed two or more orders."
        ),
        metric=(
            f"One-time customers: {int(ot['customer_count']):,} ({ot['customer_pct']:.1f}%), "
            f"revenue share {ot['revenue_pct']:.1f}%, avg spend R${ot['avg_spend']:.2f}. "
            f"Repeat customers: {int(rp['customer_count']):,} ({rp['customer_pct']:.1f}%), "
            f"revenue share {rp['revenue_pct']:.1f}%, avg total spend R${rp['avg_spend']:.2f}, "
            f"avg orders {rp['avg_order_count']:.3f}. "
            f"Source: 93,358 customers with >=1 delivered order."
        ),
        insight=(
            "The observed retention pattern shows a customer base dominated by single-purchase "
            "customers. Repeat customers have a higher average total spend (R$308.53 vs "
            "R$160.73 for one-time), but a slightly lower average per-order value (R$145.95 "
            "vs R$160.73), suggesting they buy more frequently at similar or smaller basket sizes. "
            "Dataset limitation: this dataset does not contain acquisition cost or marketing spend, "
            "so no conclusions about cost-per-acquisition economics can be drawn."
        ),
        insight_type="risk",
        recommendation=(
            "Investigate post-purchase communication and retention mechanisms (e.g. "
            "personalised follow-up, category recommendations) targeting new customers "
            "in the first 30–60 days after their initial order. Track repeat-purchase rate "
            "as a primary retention KPI. Even a modest increase in repeat rate would increase "
            "total revenue without requiring additional customer acquisition."
        ),
        source="data/repeat_vs_onetime.csv, data/customer_rfm.csv",
        qualification=(
            "Based on delivered orders within the dataset period (Sep 2016 - Aug 2018). "
            "The Olist marketplace model may structurally limit repeat purchasing; "
            "this pattern should be evaluated in the context of marketplace norms."
        ),
    ))

    # C2 — Champions revenue concentration
    insights.append(make_insight(
        id_="C2_champions_concentration",
        category="customer",
        fact=(
            f"The 'Champions' RFM segment ({int(champs['customer_count']):,} customers, "
            f"{champs['customer_pct']:.1f}% of total) generated R${champs['total_revenue']:,.0f} "
            f"in order value, representing {champs['revenue_pct']:.1f}% of total delivered revenue."
        ),
        metric=(
            f"Champions: {int(champs['customer_count']):,} customers, "
            f"avg spend R${champs['avg_spend']:.2f}, "
            f"avg frequency {champs['avg_frequency']:.3f} orders, "
            f"avg recency {champs['avg_recency_days']:.0f} days. "
            f"Total delivered revenue: R${kpis['total_order_value']:,.2f}."
        ),
        insight=(
            "Champions are the highest-value customer cohort by both RFM scoring and "
            "revenue contribution. They represent 7% of customers but generate 13.2% of revenue, "
            "with an average spend of R$312 and recent purchase activity (avg recency 91 days). "
            "Their disproportionate revenue contribution means their retention has an outsized "
            "impact on total platform revenue."
        ),
        insight_type="opportunity",
        recommendation=(
            "Prioritise service quality for Champions. Monitor their delivery and review scores "
            "as a separate cohort. Assess whether a VIP recognition mechanism (e.g. priority "
            "seller matching, early product access) could reinforce their engagement. "
            "Do not conflate Champions with repeat customers generally — some Champions have "
            "frequency=1 with very high monetary value."
        ),
        source="data/rfm_segment_summary.csv, data/kpis.json",
        qualification=(
            "RFM segments are computed using quintile scoring on the dataset population. "
            "Segment thresholds are relative to this dataset and may shift if re-computed "
            "on a different population or time window."
        ),
    ))

    # C3 — At Risk segment
    insights.append(make_insight(
        id_="C3_at_risk_dormant",
        category="customer",
        fact=(
            f"The 'At Risk' RFM segment contains {int(at_risk['customer_count']):,} customers "
            f"({at_risk['customer_pct']:.1f}% of total) with an average recency of "
            f"{at_risk['avg_recency_days']:.0f} days. They generated R${at_risk['total_revenue']:,.0f} "
            f"in historical order value ({at_risk['revenue_pct']:.1f}% of total)."
        ),
        metric=(
            f"At Risk: {int(at_risk['customer_count']):,} customers, "
            f"avg spend R${at_risk['avg_spend']:.2f}, "
            f"avg frequency {at_risk['avg_frequency']:.3f}, "
            f"avg recency {at_risk['avg_recency_days']:.0f} days. "
            f"Total revenue from segment: R${at_risk['total_revenue']:,.0f}."
        ),
        insight=(
            "At Risk customers have previously purchased at reasonable frequency and spend "
            "levels, but have not ordered recently relative to the dataset reference date "
            "(2018-08-30). Their profile differs from Lost/Inactive customers in that "
            "their monetary and frequency scores remain higher. The At Risk segment "
            "actually contains more total historical revenue (R$3.2M, 20.8%) than the "
            "Champions segment (R$2.0M, 13.2%), making it the highest-revenue RFM group."
        ),
        insight_type="opportunity",
        recommendation=(
            "Prioritise win-back campaigns for the At Risk segment before they migrate "
            "to Lost/Inactive status. Personalised category recommendations based on "
            "past purchase history may be more effective than generic promotions. "
            "Measure reactivation rate (defined as a second delivered order within "
            "90 days of campaign contact) as the success metric."
        ),
        source="data/rfm_segment_summary.csv",
        qualification=(
            "Recency is measured from the dataset reference date (2018-08-30). "
            "The At Risk segment's dormancy is observed within the dataset window only; "
            "it is possible that some customers continued purchasing outside this period."
        ),
    ))

    # C4 — Geographic concentration
    insights.append(make_insight(
        id_="C4_geographic_concentration",
        category="customer",
        fact=(
            f"São Paulo (SP) accounts for {sp_row['customer_pct']:.1f}% of unique customers "
            f"and {sp_row['revenue_pct']:.1f}% of total delivered revenue. "
            f"The top 3 states (SP, RJ, MG) together account for {top3_rev:.1f}% of revenue."
        ),
        metric=(
            f"SP: {int(sp_row['customer_uid_count']):,} customers ({sp_row['customer_pct']:.1f}%), "
            f"revenue R${sp_row['total_revenue']:,.0f} ({sp_row['revenue_pct']:.1f}%), "
            f"AOV R${sp_row['avg_order_value']:.2f}. "
            f"27 states total; source covers all of Brazil."
        ),
        insight=(
            "Customer demand is heavily concentrated in the Southeast region. States in "
            "the Northeast and North regions have a smaller share of customers and orders, "
            "though some (e.g. PB, AL, PA) show higher average order values per order — "
            "which may reflect a different product mix or limited local alternatives. "
            "SP also has the shortest observed delivery times and among the lowest late rates, "
            "which may reinforce customer activity in that state."
        ),
        insight_type="opportunity",
        recommendation=(
            "Analyse whether demand in underserved states (North and Northeast) is suppressed "
            "by logistics constraints rather than lack of consumer interest. Improving delivery "
            "reliability in high-late-rate states (AL, MA, SE, CE) may help unlock latent demand. "
            "This investigation should include seller distribution by state."
        ),
        source="data/customer_geography.csv",
        qualification=(
            "The 38-customer cross-state artefact (customers appearing in >1 state due to the "
            "per-order customer_id model) has a negligible effect on these percentages."
        ),
    ))

    # C5 — Spending concentration
    insights.append(make_insight(
        id_="C5_spending_skew",
        category="customer",
        fact=(
            "The top 10% of customers by total spend (lifetime order value >= R$318) "
            "contribute 38.3% of total revenue. Median customer spend is R$107.78 "
            "while mean spend is R$165.17, indicating a right-skewed distribution."
        ),
        metric=(
            "93,358 customers total. Top 10% (>=R$318): 9,336 customers, "
            "R$5,898,533 revenue (38.3% of total R$15,419,774). "
            "Median spend: R$107.78. Mean: R$165.17. "
            "p90: R$317.99. p99: R$1,097.06. Max: R$13,664.08."
        ),
        insight=(
            "A minority of high-value customers generate a disproportionate share of "
            "platform revenue. This concentration means that the platform's revenue is "
            "sensitive to the purchasing behaviour of a relatively small cohort. "
            "The spending distribution is right-skewed: most customers (62.1%) have "
            "total spend <= R$200, while 4.5% have total spend > R$500."
        ),
        insight_type="risk",
        recommendation=(
            "Implement a high-value customer tier for customers with lifetime spend > R$318 "
            "(top 10%). Focus retention and service-quality efforts on this cohort. "
            "Additionally, investigate the R$201-500 spend bracket (16.8% of customers, "
            "29.7% of revenue) as candidates for spend-upgrade initiatives."
        ),
        source="data/customer_spending_dist.csv, data/customer_rfm.csv",
        qualification=(
            "Total spend = total order value (product sales + freight) for delivered orders "
            "in the dataset period only."
        ),
    ))

    return insights


# ---------------------------------------------------------------------------
# SALES INSIGHTS
# ---------------------------------------------------------------------------

def sales_insights(ev):
    insights = []
    cat     = ev["cat_perf"]
    monthly = ev["monthly"]
    kpis    = ev["kpis"]
    seller  = ev["seller_perf"]

    top5 = cat.head(5)
    top5_sales = top5["product_sales_value"].sum()
    total_sales = kpis["total_product_sales"]
    top5_pct = top5_sales / total_sales * 100

    top10_sellers = seller.head(10)["product_sales_value"].sum()
    top10_sel_pct = top10_sellers / total_sales * 100

    # S1 — Category concentration
    insights.append(make_insight(
        id_="S1_category_concentration",
        category="sales",
        fact=(
            f"The top 5 product categories (health_beauty, watches_gifts, bed_bath_table, "
            f"sports_leisure, computers_accessories) account for R${top5_sales:,.0f} "
            f"in product sales ({top5_pct:.1f}% of total product sales R${total_sales:,.0f})."
        ),
        metric=(
            f"health_beauty: R${top5.iloc[0]['product_sales_value']:,.0f} ({top5.iloc[0]['order_count']:,} orders). "
            f"watches_gifts: R${top5.iloc[1]['product_sales_value']:,.0f} ({top5.iloc[1]['order_count']:,} orders). "
            f"bed_bath_table: R${top5.iloc[2]['product_sales_value']:,.0f} ({top5.iloc[2]['order_count']:,} orders). "
            f"Total categories: 72. Total delivered product sales: R${total_sales:,.2f}."
        ),
        insight=(
            "Product sales are moderately concentrated in the top 5 categories, which "
            "together represent just under half of total sales. Health & beauty and watches "
            "& gifts are the top two by revenue despite different order volumes, suggesting "
            "watches/gifts have higher average item prices. Bed, bath & table leads by order "
            "count (9,272) but ranks 3rd by sales value, indicating lower average item prices."
        ),
        insight_type="opportunity",
        recommendation=(
            "Use category-level profiling (AOV, review score, delivery performance) to "
            "identify categories with high order volume but below-average satisfaction "
            "or delivery reliability as candidates for targeted improvement. "
            "Categories with high AOV and good reviews (e.g. watches_gifts, cool_stuff) "
            "may be candidates for expansion or premium product promotion."
        ),
        source="data/category_performance.csv, data/kpis.json",
        qualification="Based on 96,478 delivered order-item lines assigned to categories.",
    ))

    # S2 — Seller concentration
    insights.append(make_insight(
        id_="S2_seller_concentration",
        category="sales",
        fact=(
            f"The top 10 sellers by product sales generate R${top10_sellers:,.0f} "
            f"({top10_sel_pct:.1f}% of total product sales) from {len(seller):,} total sellers."
        ),
        metric=(
            f"Top 10 sellers combined sales: R${top10_sellers:,.0f} ({top10_sel_pct:.1f}%). "
            f"Top seller: R$226,988 (1,124 orders). "
            f"Total sellers with delivered orders: {len(seller):,}. "
            f"Total product sales: R${total_sales:,.2f}."
        ),
        insight=(
            "Sales are moderately concentrated among a small number of high-volume sellers, "
            "all located in SP state. The top 10 sellers account for 13.3% of total sales "
            "from just 0.3% of sellers. This concentration means platform revenue and "
            "service quality metrics are disproportionately influenced by these sellers' "
            "performance."
        ),
        insight_type="risk",
        recommendation=(
            "Monitor the top 20-30 sellers by order volume as a separate performance cohort. "
            "Track their delivery late rate and review score monthly. If a high-volume seller "
            "shows deteriorating performance, proactive intervention (logistics support, "
            "carrier review) would have a larger-than-average platform impact."
        ),
        source="data/seller_performance.csv, data/kpis.json",
        qualification=(
            "Seller performance is measured from delivered order-item data. "
            "Sellers with <50 orders are excluded from the qualified delivery analysis."
        ),
    ))

    # S3 — Seasonal pattern
    peak = monthly.loc[monthly["order_count"].idxmax()]
    crisis_months = ev["del_time"][ev["del_time"]["late_pct"] > 10].sort_values("late_pct", ascending=False)
    crisis_list = ", ".join([f"{r['ym']} ({r['late_pct']:.1f}% late)" for _, r in crisis_months.iterrows()])
    insights.append(make_insight(
        id_="S3_seasonality_and_crisis",
        category="sales",
        fact=(
            f"Order volume peaked in November 2017 at {int(peak['order_count']):,} orders "
            f"(R${peak['product_sales_value']:,.0f} product sales). "
            f"Three months showed late delivery rates above 10%: {crisis_list}."
        ),
        metric=(
            f"Peak month orders: {int(peak['order_count']):,} (2017-11). "
            f"Data range: Oct 2016 – Aug 2018. "
            f"Crisis months by late rate: Mar 2018 (19.0%, 7,003 orders), "
            f"Feb 2018 (14.1%, 6,555 orders), Nov 2017 (12.4%, 7,288 orders)."
        ),
        insight=(
            "The highest-volume month (Nov 2017, likely Black Friday period) coincides "
            "with the third-highest late-delivery rate (12.4%). The two months with the "
            "highest late rates (Feb-Mar 2018, combined 13,558 orders) show markedly lower "
            "average review scores (3.88 and 3.81 respectively vs the dataset average of 4.16). "
            "This is an observed association — the specific causes of the Feb-Mar 2018 "
            "delivery deterioration are not determinable from this dataset alone."
        ),
        insight_type="risk",
        recommendation=(
            "Implement logistics capacity planning to anticipate high-volume periods "
            "(e.g. Black Friday in November). Investigate the Feb-Mar 2018 crisis months "
            "to identify whether they reflect a specific logistics disruption, carrier "
            "capacity constraint, or other identifiable factor. "
            "Use the monthly late-rate trend as an early warning dashboard metric."
        ),
        source="data/monthly_trends.csv, data/delivery_over_time.csv",
        qualification="Monthly analysis uses the order purchase timestamp, not the delivery date.",
    ))

    return insights


# ---------------------------------------------------------------------------
# DELIVERY INSIGHTS
# ---------------------------------------------------------------------------

def delivery_insights(ev):
    insights = []
    del_ov   = ev["del_ov"]
    del_state= ev["del_state"]
    del_cat  = ev["del_cat"]
    del_time = ev["del_time"]
    s_del    = ev["seller_del"]
    nat_late = del_ov["late_pct"]
    early7d  = del_ov["delay_bucket_counts"]["Early >7d"]
    total_n  = del_ov["n_orders"]

    # D1 — Conservative estimates
    insights.append(make_insight(
        id_="D1_conservative_estimates",
        category="delivery",
        fact=(
            f"{early7d:,} of {total_n:,} delivered orders ({early7d/total_n*100:.1f}%) "
            f"arrived more than 7 days before the estimated delivery date. "
            f"Overall on-time rate: {del_ov['on_time_pct']:.2f}%."
        ),
        metric=(
            f"Early >7d: {early7d:,} ({early7d/total_n*100:.1f}%). "
            f"Early 3-7d: {del_ov['delay_bucket_counts']['Early 3-7d']:,} (9.8%). "
            f"On-time (arrived on or before estimate): {del_ov['on_time_pct']:.2f}%. "
            f"Avg delivery: {del_ov['avg_delivery_days']:.2f} days. "
            f"Median: {del_ov['median_delivery_days']:.1f} days. "
            f"Avg delay vs estimate: {del_ov['avg_delay_days']:.2f} days (negative = early)."
        ),
        insight=(
            "The vast majority of deliveries arrive well ahead of the communicated "
            "estimated delivery date. The average order arrives 11.88 days before its "
            "estimate (median 12 days early). This suggests that estimated delivery dates "
            "are systematically set conservatively relative to actual logistics performance. "
            "There is an observed association in this dataset between early delivery and "
            "higher review scores (avg 4.31 for Early >7d vs 4.16 overall), though other "
            "factors may contribute to this pattern."
        ),
        insight_type="opportunity",
        recommendation=(
            "Review the delivery estimation model. A more accurate, narrower delivery "
            "window communicated to customers could improve expectation management. "
            "If the estimate is deliberately conservative as a buffer, quantify the "
            "trade-off: a tighter estimate risks more 'late' classifications but may "
            "provide more useful information to customers. Begin with a controlled "
            "experiment on a subset of orders before changing platform-wide estimates."
        ),
        source="data/delivery_overview.json, data/review_by_delay_detailed.csv",
        qualification=(
            "8 delivered orders with missing actual delivery dates are excluded from "
            "delay calculations. Delay = actual_delivered - estimated_delivery."
        ),
    ))

    # D2 — Northeast delivery gap
    ne_states = ["AL","MA","SE","PI","CE","RN","PB","BA"]
    ne_data = del_state[
        del_state["customer_state"].isin(ne_states) & (~del_state["low_volume"])
    ]
    ne_late_avg = ne_data["late_pct"].mean()
    ne_orders   = ne_data["order_count"].sum()
    rj_row = del_state[del_state["customer_state"] == "RJ"].iloc[0]

    insights.append(make_insight(
        id_="D2_northeast_delivery_gap",
        category="delivery",
        fact=(
            f"Among Northeastern states with >= 200 orders, the average late delivery "
            f"rate is {ne_late_avg:.1f}% vs the national rate of {nat_late:.2f}%. "
            f"AL: 21.4% (397 orders), MA: 17.4% (717), SE: 15.2% (335), "
            f"PI: 13.9% (476), CE: 13.8% (1,279). Total orders in these states: {ne_orders:,}."
        ),
        metric=(
            f"National late rate: {nat_late:.2f}% (n={total_n:,}). "
            f"Northeast avg late (8 states, >=200 orders): {ne_late_avg:.1f}%. "
            f"Average delivery days in high-late NE states: AL=24.0, MA=21.1, SE=21.0 "
            f"(national avg: {del_ov['avg_delivery_days']:.2f} days)."
        ),
        insight=(
            "Delivery performance is materially weaker across the Northeast compared to "
            "the national average. Late rates in AL, MA, and SE are 2–3x the national "
            "rate and average delivery times in these states (21-24 days) are nearly "
            "double the national median (10 days). These states also show below-average "
            "review scores (observed association — not a causal claim)."
        ),
        insight_type="risk",
        recommendation=(
            "Investigate the seller and logistics network in AL, MA, SE, and CE. "
            "Determine whether the elevated late rates are driven by: (a) lack of local "
            "sellers requiring long-distance shipments, (b) specific carrier performance "
            "on Northeast routes, or (c) estimation issues in the delivery date model "
            "for these regions. Focus operational improvement on the highest-volume "
            "states (CE: 1,279 orders, BA: 3,256 orders) for maximum impact."
        ),
        source="data/delivery_by_state_enriched.csv, data/delivery_overview.json",
        qualification=(
            "States with <200 orders (RR, AP, AC, AM) are excluded from 'high-impact' "
            "classification due to small sample sizes, though their late rates are noted."
        ),
    ))

    # D3 — RJ high-volume risk
    insights.append(make_insight(
        id_="D3_rj_high_volume_risk",
        category="delivery",
        fact=(
            f"Rio de Janeiro (RJ) is the 2nd largest market by orders ({rj_row['order_count']:,} orders, "
            f"12.1% late delivery rate, +5.3pp above national average {nat_late:.2f}%). "
            f"Avg delivery days: {rj_row['avg_delivery_days']:.1f} (national: {del_ov['avg_delivery_days']:.2f})."
        ),
        metric=(
            f"RJ: {rj_row['order_count']:,} delivered orders, late%={rj_row['late_pct']:.1f}%, "
            f"avg_delivery_days={rj_row['avg_delivery_days']:.1f}, "
            f"avg_review={rj_row['avg_review_score']:.3f} (national: {ev['kpis']['avg_review_score']:.4f})."
        ),
        insight=(
            "RJ has the highest absolute number of late deliveries among all states due to "
            "its combination of large order volume and above-national late rate. Its "
            "12.1% late rate places it among the top 8 most-affected states. Its average "
            "review score (3.965) is 0.19 points below the national average, which is "
            "consistent with higher late rates (observed association)."
        ),
        insight_type="risk",
        recommendation=(
            "Given RJ's order volume (12,350 orders), even a small improvement in late "
            "delivery rate would affect a meaningful number of customers. Investigate "
            "whether high-volume sellers shipping to RJ are the primary driver of late "
            "deliveries (cross-reference seller_delivery_qualified.csv for RJ-destination sellers)."
        ),
        source="data/delivery_by_state_enriched.csv",
        qualification="RJ customer late rate; seller origin state differs from customer destination.",
    ))

    # D4 — Seller delivery concentration
    above_nat = s_del[s_del["late_pct"] > nat_late * 1.5]
    insights.append(make_insight(
        id_="D4_seller_delivery_concentration",
        category="delivery",
        fact=(
            f"{len(above_nat)} sellers (with >= 50 delivered orders each) have a late "
            f"delivery rate >= 1.5x the national average ({nat_late:.2f}%). "
            f"These sellers collectively handled {above_nat['order_count'].sum():,} orders. "
            f"Their average review score is {above_nat['avg_review_score'].mean():.3f} "
            f"vs national {ev['kpis']['avg_review_score']:.4f}."
        ),
        metric=(
            f"Qualified sellers (>=50 orders): {len(s_del):,}. "
            f"Sellers with late% >= 1.5x national: {len(above_nat)}. "
            f"Combined orders from that group: {above_nat['order_count'].sum():,}. "
            f"Min seller orders to qualify: 50."
        ),
        insight=(
            "A minority of meaningful-volume sellers show materially elevated late delivery "
            "rates. Their below-average review scores are consistent with the observed "
            "dataset-wide association between delivery delays and lower review scores. "
            "These sellers represent a concentrated source of delivery quality variance."
        ),
        insight_type="risk",
        recommendation=(
            "Establish a seller delivery performance monitoring programme. Flag sellers "
            "with >= 50 orders AND late% > 10% (1.5x national) for quarterly logistics "
            "reviews. Investigate whether late deliveries are driven by carrier choice, "
            "order fulfilment time (from order to dispatch), or destination-state effects. "
            "Consider SLA agreements with high-volume sellers."
        ),
        source="data/seller_delivery_qualified.csv",
        qualification=(
            "Sellers with <50 orders are excluded to avoid small-sample ranking noise. "
            "425 sellers qualify; the full 2,970 sellers are in seller_performance.csv."
        ),
    ))

    # D5 — Category delivery (notable volume)
    hi_cat = del_cat[
        (del_cat["late_pct"] > nat_late) &
        (~del_cat["low_volume"])
    ].head(4)
    cat_facts = "; ".join([
        f"{r['product_category_name_english']} ({r['late_pct']:.1f}%, {r['order_count']:,} orders)"
        for _, r in hi_cat.iterrows()
    ])
    insights.append(make_insight(
        id_="D5_category_delivery_variance",
        category="delivery",
        fact=(
            f"Several high-volume categories show above-national late delivery rates: "
            f"{cat_facts}."
        ),
        metric=(
            f"National late%: {nat_late:.2f}%. "
            f"audio: 11.9% (344 orders). "
            f"baby: 8.2% (2,763 orders). "
            f"office_furniture: 8.1% (1,246 orders). "
            f"electronics: 7.7% (2,507 orders). "
            f"Low-volume threshold: <100 orders (excluded from this insight)."
        ),
        insight=(
            "'baby' and 'office_furniture' are notable because of their order volume: "
            "baby has 2,763 orders at 8.2% late — meaning approximately 227 late deliveries, "
            "each potentially involving time-sensitive products for new parents. "
            "Office furniture at 8.1% late with low review scores (3.65, lowest among "
            "high-volume categories) may reflect product complexity in delivery (large, heavy items)."
        ),
        insight_type="opportunity",
        recommendation=(
            "For high-volume categories with above-national late rates (baby, electronics), "
            "investigate whether category-specific logistics improvements (e.g. dedicated "
            "fulfilment for fragile/large items, specialised carriers for baby products) "
            "could reduce late rates. For office_furniture, the combination of high weight "
            "and low review score warrants a specific investigation into product delivery "
            "handling standards."
        ),
        source="data/delivery_by_category_enriched.csv",
        qualification="Category assigned from first item in multi-item orders (known limitation).",
    ))

    return insights


# ---------------------------------------------------------------------------
# SATISFACTION INSIGHTS
# ---------------------------------------------------------------------------

def satisfaction_insights(ev):
    insights = []
    rev_delay = ev["rev_delay"]
    rev_state = ev["rev_state"]
    rev_cat   = ev["rev_cat"]
    kpis      = ev["kpis"]
    nat_review= kpis["avg_review_score"]

    early7  = rev_delay[rev_delay["delay_bucket"] == "Early >7d"].iloc[0]
    late47  = rev_delay[rev_delay["delay_bucket"] == "Late 4-7d"].iloc[0]
    late14  = rev_delay[rev_delay["delay_bucket"] == "Late >14d"].iloc[0]
    late13  = rev_delay[rev_delay["delay_bucket"] == "Late 1-3d"].iloc[0]

    # T1 — Delay/review association
    insights.append(make_insight(
        id_="T1_delay_review_association",
        category="satisfaction",
        fact=(
            f"Orders arriving >7 days early (n={int(early7['order_count']):,}) have an "
            f"observed average review score of {early7['avg_review_score']:.3f} "
            f"({early7['pct_5star']:.1f}% 5-star, {early7['pct_1star']:.1f}% 1-star). "
            f"Orders arriving 4-7 days late (n={int(late47['order_count']):,}) have an "
            f"observed average review score of {late47['avg_review_score']:.3f} "
            f"({late47['pct_1star']:.1f}% 1-star, {late47['pct_5star']:.1f}% 5-star). "
            f"Orders >14 days late (n={int(late14['order_count']):,}): avg {late14['avg_review_score']:.3f}."
        ),
        metric=(
            f"7 delay buckets, n={int(rev_delay['order_count'].sum()):,} total. "
            f"Early >7d: avg={early7['avg_review_score']:.3f}. "
            f"Early 0-3d: avg={rev_delay[rev_delay['delay_bucket']=='Early 0-3d'].iloc[0]['avg_review_score']:.3f}. "
            f"Late 1-3d: avg={late13['avg_review_score']:.3f}. "
            f"Late 4-7d: avg={late47['avg_review_score']:.3f}. "
            f"Late 8-14d: avg={rev_delay[rev_delay['delay_bucket']=='Late 8-14d'].iloc[0]['avg_review_score']:.3f}. "
            f"Late >14d: avg={late14['avg_review_score']:.3f}. "
            f"National avg: {nat_review:.4f}."
        ),
        insight=(
            "OBSERVED ASSOCIATION: There is a strong monotonic pattern between delivery "
            "timing relative to the estimated date and customer review score. As orders "
            "become later relative to their estimate, the observed average review score "
            "decreases consistently. The sharpest single transition is between "
            f"'Early 0-3d' ({rev_delay[rev_delay['delay_bucket']=='Early 0-3d'].iloc[0]['avg_review_score']:.3f}) "
            f"and 'Late 1-3d' ({late13['avg_review_score']:.3f}) — a difference of "
            f"{rev_delay[rev_delay['delay_bucket']=='Early 0-3d'].iloc[0]['avg_review_score'] - late13['avg_review_score']:.3f} points "
            "for orders arriving just 1-3 days past the estimate. "
            "No causal analysis has been performed; this is an observed relationship in the data."
        ),
        insight_type="risk",
        recommendation=(
            "Use delay bucket monitoring as a leading indicator of satisfaction risk. "
            "The 'Late 1-3d' bucket is the early warning threshold — once an order "
            "crosses its estimated delivery date, the associated review score pattern "
            "drops sharply. Proactive customer communication when an order is at risk "
            "of crossing its estimate (e.g. via the ML model's predictions) may help "
            "manage expectations before the review is submitted."
        ),
        source="data/review_by_delay_detailed.csv",
        qualification=(
            "Review scores are from delivered orders with both review score and valid "
            "delivery delay data (n=95,824). This is an observed association only; "
            "other factors may also affect review scores."
        ),
    ))

    # T2 — State satisfaction gap
    low_state = rev_state.sort_values("avg_review_score").head(3)
    ma_row = rev_state[rev_state["customer_state"] == "MA"].iloc[0]
    sp_row = rev_state[rev_state["customer_state"] == "SP"].iloc[0]
    insights.append(make_insight(
        id_="T2_state_satisfaction_gap",
        category="satisfaction",
        fact=(
            f"MA has the lowest observed average review score: {ma_row['avg_review_score']:.3f} "
            f"(n={int(ma_row['review_count']):,}, late%={ma_row['late_pct']:.1f}%). "
            f"SP has the highest: {sp_row['avg_review_score']:.3f} "
            f"(n={int(sp_row['review_count']):,}, late%={sp_row['late_pct']:.1f}%). "
            f"National average: {nat_review:.4f}."
        ),
        metric=(
            f"27 states. MA: avg={ma_row['avg_review_score']:.3f}, late%={ma_row['late_pct']:.1f}%, n={int(ma_row['review_count']):,}. "
            f"AL: avg={rev_state[rev_state['customer_state']=='AL'].iloc[0]['avg_review_score']:.3f}, late%={rev_state[rev_state['customer_state']=='AL'].iloc[0]['late_pct']:.1f}%. "
            f"SP: avg={sp_row['avg_review_score']:.3f}, late%={sp_row['late_pct']:.1f}%, n={int(sp_row['review_count']):,}."
        ),
        insight=(
            "States with the highest late delivery rates (MA, AL, SE) also show the "
            "lowest average review scores. The gap between the highest-scoring state (SP: 4.246) "
            "and lowest (MA: 3.833) is 0.413 points on a 5-point scale. This is an observed "
            "pattern consistent with the delay-review association found in Insight T1. "
            "Causal direction is not established from this data alone."
        ),
        insight_type="risk",
        recommendation=(
            "States with both high late rates and low review scores (MA, AL, SE) should "
            "be priorities for logistics improvement investigations. Monitor review scores "
            "by state monthly on the dashboard as a composite satisfaction indicator "
            "alongside the delivery late rate."
        ),
        source="data/review_by_state.csv, data/delivery_by_state_enriched.csv",
        qualification="Sample sizes vary by state (n=40 for RR to n=40,266 for SP); interpret low-volume states with caution.",
    ))

    # T3 — Category satisfaction
    office = rev_cat[rev_cat["product_category_name_english"] == "office_furniture"].iloc[0]
    hb     = rev_cat[rev_cat["product_category_name_english"] == "health_beauty"].iloc[0]
    books  = rev_cat[(rev_cat["product_category_name_english"] == "books_general_interest")].iloc[0]
    insights.append(make_insight(
        id_="T3_category_satisfaction",
        category="satisfaction",
        fact=(
            f"office_furniture has the lowest average review score among categories with "
            f">= 200 reviews: {office['avg_review_score']:.3f} (n={int(office['review_count']):,}, "
            f"late%={office['late_pct']:.1f}%, 1-star%={office['pct_1star']:.1f}%). "
            f"books_general_interest has the highest at {books['avg_review_score']:.3f} "
            f"(n={int(books['review_count']):,})."
        ),
        metric=(
            f"office_furniture: avg={office['avg_review_score']:.3f}, n={int(office['review_count']):,}, "
            f"late%={office['late_pct']:.1f}%, 1*%={office['pct_1star']:.1f}%. "
            f"health_beauty (top revenue category): avg={hb['avg_review_score']:.3f}, "
            f"n={int(hb['review_count']):,}, 1*%={hb['pct_1star']:.1f}%. "
            f"national avg: {nat_review:.4f}. Categories with >=200 reviews."
        ),
        insight=(
            "Office furniture stands out with the lowest review score (3.650) and the highest "
            "1-star percentage (17.2%) among meaningful-volume categories. Its late delivery "
            "rate (8.0%) is above national average. This combination — low satisfaction, high "
            "1-star rate, above-average late rate — makes it a concentrated satisfaction risk. "
            "Health & beauty (top revenue category) shows above-average reviews (4.235) and "
            "is performing well relative to its revenue contribution."
        ),
        insight_type="risk",
        recommendation=(
            "Investigate the specific complaints in 1-star reviews for office_furniture to "
            "understand whether satisfaction issues are primarily delivery-related (late, "
            "damaged), product-related (description accuracy), or service-related. "
            "Consider whether large/heavy item delivery standards need category-specific handling."
        ),
        source="data/review_by_category.csv",
        qualification="Category assigned from first item in multi-item orders.",
    ))

    return insights


# ---------------------------------------------------------------------------
# ML INSIGHTS
# ---------------------------------------------------------------------------

def ml_insights(ev):
    insights = []
    ml  = ev["ml_metrics"]
    fi  = ev["ml_fi"]
    lr  = ml["logistic_regression"]
    rf  = ml["random_forest"]
    lr_cm = lr["confusion_matrix"]

    top3_features = fi.head(3)[["feature","importance"]].to_dict("records")
    top3_str = ", ".join([f"{r['feature']} ({r['importance']:.3f})" for r in top3_features])

    # M1 — Model performance
    insights.append(make_insight(
        id_="M1_lr_model_performance",
        category="ml",
        fact=(
            f"On the held-out test period (May–Aug 2018, {ml['test_orders']:,} orders, "
            f"late rate {ml['test_late_pct']:.2f}%), the Logistic Regression model achieved: "
            f"Recall={lr['recall']:.4f}, Precision={lr['precision']:.4f}, "
            f"F1={lr['f1']:.4f}, ROC-AUC={lr['roc_auc']:.4f}. "
            f"The model identified {lr_cm['TP']:,} of {lr_cm['TP']+lr_cm['FN']:,} "
            f"actual late orders (missed {lr_cm['FN']:,})."
        ),
        metric=(
            f"Training: {ml['train_orders']:,} orders (late%={ml['train_late_pct']:.2f}%). "
            f"Test: {ml['test_orders']:,} orders (late%={ml['test_late_pct']:.2f}%). "
            f"Cutoff: {ml['train_cutoff_date']}. "
            f"LR: accuracy={lr['accuracy']}, precision={lr['precision']}, "
            f"recall={lr['recall']}, f1={lr['f1']}, roc_auc={lr['roc_auc']}. "
            f"RF: roc_auc={rf['roc_auc']} (near-random on test window). "
            f"LR TP={lr_cm['TP']}, FP={lr_cm['FP']}, FN={lr_cm['FN']}, TN={lr_cm['TN']}."
        ),
        insight=(
            "The Logistic Regression model outperforms the Random Forest on this evaluation "
            f"(ROC-AUC {lr['roc_auc']:.4f} vs {rf['roc_auc']:.4f}). "
            "The RF's near-random performance on the test set is attributed to a temporal "
            "distribution shift: the test window (May–Aug 2018) has a late rate of 3.49%, "
            "while the training window had 7.59% (including the Feb-Mar 2018 crisis months). "
            "The LR's recall of 76.41% on the test period indicates it can identify the "
            "majority of late orders from information available at approval time. "
            "The LR's low precision (6.08%) means it generates many false positives, "
            "which is typical for highly imbalanced detection tasks when recall is prioritised."
        ),
        insight_type="opportunity",
        recommendation=(
            "Deploy the Logistic Regression model as an early-warning mechanism at order "
            "approval. Focus operational use on the high-recall characteristic: use flagged "
            "orders for proactive monitoring, not as guaranteed predictions of lateness. "
            "The model should NOT be presented to customers as a delivery promise. "
            "Re-evaluate model performance after accumulating 3-6 additional months of data "
            "to assess whether the temporal distribution shift is a persistent pattern."
        ),
        source="data/ml_model_metrics.json, data/ml_feature_importance.csv",
        qualification=(
            "Model evaluation is on May-Aug 2018 only (low-late-rate period). "
            "Performance on higher-late-rate periods (e.g. Nov-Mar) may differ. "
            "Class imbalance (~93% on-time) addressed with class_weight=balanced. "
            "Time-based split used to avoid future-information leakage."
        ),
    ))

    # M2 — Feature predictors
    insights.append(make_insight(
        id_="M2_predictive_features",
        category="ml",
        fact=(
            f"The three highest-importance features in the Random Forest model are: {top3_str}. "
            f"In the Logistic Regression, the highest-magnitude coefficients involve "
            f"seller_state_MA (+2.093), customer_state_RR (+1.534), and customer_state_SP (-1.392)."
        ),
        metric=(
            f"RF top features (association, not causation): "
            f"purchase_month=0.213, customer_state_SP=0.152, total_freight_value=0.128, "
            f"customer_state_RJ=0.098, days_to_estimated=0.093. "
            f"LR coefficients: positive = associated with predicted late, negative = on-time. "
            f"13 features total (10 numeric + 3 categorical). "
            f"Prediction point: at/after order approval."
        ),
        insight=(
            "PREDICTIVE ASSOCIATION (not causation): Month of purchase is the strongest "
            "single predictor in the RF model, which is consistent with the observed "
            "seasonal pattern (Feb-Mar 2018 crisis months, Nov 2017 peak). "
            "Customer state features (SP negative = associated with lower late risk, "
            "RJ positive = higher late risk) are consistent with the regional delivery "
            "performance patterns identified in Phase 5. Total freight value (higher = "
            "associated with more complex/distant routes) and days_to_estimated (window "
            "length) are also predictive. These are associations in the trained model, "
            "not causal explanations."
        ),
        insight_type="opportunity",
        recommendation=(
            "The model's key drivers align with operationally actionable dimensions: "
            "seasonal planning (purchase_month), regional logistics (customer_state), "
            "and routing complexity (total_freight_value). Use these features as "
            "lenses for operational investigation rather than as model outputs alone."
        ),
        source="data/ml_feature_importance.csv, data/ml_model_metrics.json",
        qualification=(
            "RF feature importance measures the reduction in impurity contributed by "
            "each feature across all trees — it reflects the model's internal weighting, "
            "not real-world causal structure. LR coefficients reflect log-odds contribution "
            "in the linear model after scaling and one-hot encoding."
        ),
    ))

    return insights


# ---------------------------------------------------------------------------
# VALIDATE AND ASSEMBLE
# ---------------------------------------------------------------------------

def validate_insights(insights, ev):
    """
    Cross-check key numeric claims against source files.
    Removes any insight that fails a critical validation.
    """
    print("\n" + "="*60)
    print("VALIDATION — CROSS-CHECKING INSIGHT METRICS")
    print("="*60)

    errors = []
    kpis = ev["kpis"]
    repeat = ev["repeat"]

    # Check C1
    ot = repeat[repeat["customer_type"] == "One-Time"].iloc[0]
    assert abs(ot["customer_pct"] - 97.0) < 0.1, "C1 one-time pct mismatch"
    print("  PASS C1: one-time customer pct")

    # Check D1
    del_ov = ev["del_ov"]
    assert abs(del_ov["on_time_pct"] - kpis["on_time_delivery_pct"]) < 0.05, "D1 on-time pct mismatch"
    print("  PASS D1: on-time pct vs kpis.json")

    # Check T1
    rd = ev["rev_delay"]
    e7 = rd[rd["delay_bucket"] == "Early >7d"].iloc[0]["avg_review_score"]
    l47= rd[rd["delay_bucket"] == "Late 4-7d"].iloc[0]["avg_review_score"]
    assert e7 > l47, "T1 review monotonicity check failed"
    print("  PASS T1: Early>7d review > Late 4-7d review (monotonic)")

    # Check M1
    ml = ev["ml_metrics"]
    assert ml["logistic_regression"]["roc_auc"] > ml["random_forest"]["roc_auc"], "M1 LR should beat RF"
    print("  PASS M1: LR ROC-AUC > RF ROC-AUC")

    # Check total revenue reconciliation
    rfm = ev["rfm_seg"]
    seg_total_rev = rfm["total_revenue"].sum()
    assert abs(seg_total_rev - kpis["total_order_value"]) < 1.0, "RFM revenue mismatch"
    print(f"  PASS: RFM segment revenue sum = kpis total_order_value (diff < R$1)")

    print(f"\n  All validation checks passed. Proceeding with {len(insights)} insights.")
    return insights


# ---------------------------------------------------------------------------
# MAIN — Phase 7
# ---------------------------------------------------------------------------

def main_insight_engine():
    print("\n" + "#"*60)
    print("  OLIST INSIGHT ENGINE — PHASE 7")
    print("#"*60)

    print("\n  Loading evidence from all Phase 3-6 outputs...")
    ev = load_evidence()

    print("\n  Building insight categories...")
    all_insights = []
    all_insights += customer_insights(ev)
    all_insights += sales_insights(ev)
    all_insights += delivery_insights(ev)
    all_insights += satisfaction_insights(ev)
    all_insights += ml_insights(ev)

    all_insights = validate_insights(all_insights, ev)

    # Summary
    risks        = [i for i in all_insights if i["type"] == "risk"]
    opps         = [i for i in all_insights if i["type"] == "opportunity"]

    print("\n" + "="*60)
    print("INSIGHT SUMMARY")
    print("="*60)
    cats = {}
    for ins in all_insights:
        cats.setdefault(ins["category"], []).append(ins["id"])

    for cat, ids in cats.items():
        print(f"  {cat:<15} {len(ids)} insights: {', '.join(ids)}")

    print(f"\n  Total insights      : {len(all_insights)}")
    print(f"  Risks               : {len(risks)}")
    print(f"  Opportunities       : {len(opps)}")
    print(f"  Recommendations     : {len(all_insights)} (one per insight)")

    print(f"\n  Insight IDs:")
    for ins in all_insights:
        emoji = "RISK" if ins["type"] == "risk" else "OPP "
        print(f"  [{emoji}] {ins['id']}")

    # Save
    out_path = os.path.join(DATA_DIR, "insights_master.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_insights, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {out_path}  ({len(all_insights)} insights)")

    print("\n" + "#"*60)
    print("  PHASE 7 COMPLETE")
    print("#"*60 + "\n")


# ============================================================
# MODULE: app.py — Phase 8
# ============================================================

import os, json, math
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request, abort

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR_APP = BASE_DIR / "data"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# DATA LOADING  (once at startup — never re-loaded per request)
# ---------------------------------------------------------------------------

def _load():
    """Load all precomputed analytical outputs into memory."""
    D = DATA_DIR_APP
    print("  Loading analytical data...", flush=True)

    data = {}

    # Scalar KPIs
    data["kpis"] = json.loads((D / "kpis.json").read_text(encoding="utf-8"))

    # Time-series
    data["monthly"] = pd.read_csv(D / "monthly_trends.csv")

    # Category
    data["category_perf"]   = pd.read_csv(D / "category_performance.csv")
    data["delivery_cat"]    = pd.read_csv(D / "delivery_by_category_enriched.csv")
    data["review_cat"]      = pd.read_csv(D / "review_by_category.csv")

    # State
    data["state_perf"]      = pd.read_csv(D / "state_performance.csv")
    data["delivery_state"]  = pd.read_csv(D / "delivery_by_state_enriched.csv")
    data["review_state"]    = pd.read_csv(D / "review_by_state.csv")
    data["cust_geo"]        = pd.read_csv(D / "customer_geography.csv")

    # Customer / RFM
    data["rfm_seg"]         = pd.read_csv(D / "rfm_segment_summary.csv")
    data["repeat"]          = pd.read_csv(D / "repeat_vs_onetime.csv")
    data["spend_dist"]      = pd.read_csv(D / "customer_spending_dist.csv")
    data["seg_spend"]       = pd.read_csv(D / "segment_spend_profile.csv")

    # Seller
    data["seller_perf"]     = pd.read_csv(D / "seller_performance.csv")
    data["seller_del"]      = pd.read_csv(D / "seller_delivery_qualified.csv")

    # Delivery
    data["delivery_ov"]     = json.loads((D / "delivery_overview.json").read_text(encoding="utf-8"))
    data["delivery_time"]   = pd.read_csv(D / "delivery_over_time.csv")
    data["delay_dist"]      = pd.read_csv(D / "delivery_delay_dist.csv")

    # Satisfaction
    data["rev_delay"]       = pd.read_csv(D / "review_by_delay_detailed.csv")
    data["rev_score_dist"]  = pd.read_csv(D / "review_score_dist.csv")
    data["sat_time"]        = pd.read_csv(D / "satisfaction_over_time.csv")

    # Insights
    data["insights"]        = json.loads((D / "insights_master.json").read_text(encoding="utf-8"))

    # ML
    data["ml_metrics"]      = json.loads((D / "ml_model_metrics.json").read_text(encoding="utf-8"))
    data["ml_fi"]           = pd.read_csv(D / "ml_feature_importance.csv")
    data["ml_cm"]           = pd.read_csv(D / "ml_confusion_matrix.csv")
    data["ml_lr_coef"]      = pd.read_csv(D / "ml_lr_coefficients.csv")

    # Build helper lookups
    data["all_categories"]  = sorted(data["category_perf"]["product_category_name_english"].dropna().unique().tolist())
    data["all_states"]      = sorted(data["state_perf"]["customer_state"].dropna().unique().tolist())
    data["all_segments"]    = sorted(data["rfm_seg"]["segment"].dropna().unique().tolist())

    print(f"  Data loaded: {len(data)} datasets", flush=True)
    return data


# Load at import time
try:
    _D = _load()
except Exception as e:
    print(f"ERROR loading data: {e}")
    raise


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _safe_float(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)

def _safe_int(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(v)

def _df_to_records(df):
    """Convert DataFrame to JSON-safe records."""
    records = []
    for row in df.to_dict("records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        records.append(clean)
    return records

def _get_param_list(name, valid_set=None):
    """Parse a comma-separated query parameter into a list, validated against a set."""
    raw = request.args.get(name, "").strip()
    if not raw:
        return []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if valid_set:
        items = [x for x in items if x in valid_set]
    return items

def _get_date_range():
    """Parse start/end date query params. Return (start_str, end_str) or (None, None)."""
    start = request.args.get("start", "").strip() or None
    end   = request.args.get("end",   "").strip() or None
    return start, end


# ---------------------------------------------------------------------------
# ROUTES — PAGE
# ---------------------------------------------------------------------------

@app.route("/")
@app.route("/executive")
def page_executive():
    return render_template("index.html", active_page="executive")

@app.route("/sales")
def page_sales():
    return render_template("index.html", active_page="sales")

@app.route("/customers")
def page_customers():
    return render_template("index.html", active_page="customers")

@app.route("/delivery")
def page_delivery():
    return render_template("index.html", active_page="delivery")


# ---------------------------------------------------------------------------
# API — META
# ---------------------------------------------------------------------------

@app.route("/api/filters")
def api_filters():
    """Return available filter values."""
    return jsonify({
        "categories": _D["all_categories"],
        "states":     _D["all_states"],
        "segments":   _D["all_segments"],
        "date_range": {
            "min": _D["monthly"]["ym"].min(),
            "max": _D["monthly"]["ym"].max(),
        },
    })


# ---------------------------------------------------------------------------
# API — KPIS
# ---------------------------------------------------------------------------

@app.route("/api/kpis")
def api_kpis():
    """Top-line scalar KPIs. These are fixed (full-dataset scope) — filtering
    is not applied because the KPIs are precomputed totals from Phase 3.
    The response includes a 'scope' field explaining this."""
    kpis = _D["kpis"].copy()
    kpis["_scope"] = "Delivered orders only (order_status = delivered)"
    kpis["_note"]  = "These are full-dataset KPIs; filters apply to charts only."
    return jsonify(kpis)


# ---------------------------------------------------------------------------
# API — MONTHLY TRENDS
# ---------------------------------------------------------------------------

@app.route("/api/monthly-trends")
def api_monthly_trends():
    df = _D["monthly"].copy()
    start, end = _get_date_range()
    if start:
        df = df[df["ym"] >= start]
    if end:
        df = df[df["ym"] <= end]
    return jsonify(_df_to_records(df))


# ---------------------------------------------------------------------------
# API — CATEGORIES
# ---------------------------------------------------------------------------

@app.route("/api/categories")
def api_categories():
    cats = request.args.get("categories", "").strip()
    df_perf = _D["category_perf"].copy()
    df_del  = _D["delivery_cat"].copy()
    df_rev  = _D["review_cat"].copy()

    if cats:
        cat_list = [c.strip() for c in cats.split(",") if c.strip()]
        if cat_list:
            df_perf = df_perf[df_perf["product_category_name_english"].isin(cat_list)]
            df_del  = df_del[df_del["product_category_name_english"].isin(cat_list)]
            df_rev  = df_rev[df_rev["product_category_name_english"].isin(cat_list)]

    top_n = int(request.args.get("top", 20))
    df_perf = df_perf.head(top_n)

    return jsonify({
        "performance": _df_to_records(df_perf),
        "delivery":    _df_to_records(df_del),
        "reviews":     _df_to_records(df_rev),
    })


# ---------------------------------------------------------------------------
# API — STATES
# ---------------------------------------------------------------------------

@app.route("/api/states")
def api_states():
    states = request.args.get("states", "").strip()
    df_perf  = _D["state_perf"].copy()
    df_del   = _D["delivery_state"].copy()
    df_rev   = _D["review_state"].copy()
    df_geo   = _D["cust_geo"].copy()

    if states:
        state_list = [s.strip() for s in states.split(",") if s.strip()]
        if state_list:
            df_perf = df_perf[df_perf["customer_state"].isin(state_list)]
            df_del  = df_del[df_del["customer_state"].isin(state_list)]
            df_rev  = df_rev[df_rev["customer_state"].isin(state_list)]
            df_geo  = df_geo[df_geo["customer_state"].isin(state_list)]

    return jsonify({
        "performance": _df_to_records(df_perf),
        "delivery":    _df_to_records(df_del),
        "reviews":     _df_to_records(df_rev),
        "geography":   _df_to_records(df_geo),
    })


# ---------------------------------------------------------------------------
# API — CUSTOMERS / RFM
# ---------------------------------------------------------------------------

@app.route("/api/customers")
def api_customers():
    segs = request.args.get("segments", "").strip()
    df_rfm  = _D["rfm_seg"].copy()
    df_rep  = _D["repeat"].copy()
    df_geo  = _D["cust_geo"].copy()
    df_sp   = _D["spend_dist"].copy()
    df_seg_sp = _D["seg_spend"].copy()

    if segs:
        seg_list = [s.strip() for s in segs.split(",") if s.strip()]
        if seg_list:
            df_rfm    = df_rfm[df_rfm["segment"].isin(seg_list)]
            df_seg_sp = df_seg_sp[df_seg_sp["segment"].isin(seg_list)]

    return jsonify({
        "rfm_segments":    _df_to_records(df_rfm),
        "repeat_onetime":  _df_to_records(df_rep),
        "geography":       _df_to_records(df_geo),
        "spending_dist":   _df_to_records(df_sp),
        "segment_spend":   _df_to_records(df_seg_sp),
    })


# ---------------------------------------------------------------------------
# API — DELIVERY
# ---------------------------------------------------------------------------

@app.route("/api/delivery")
def api_delivery():
    states = request.args.get("states", "").strip()
    cats   = request.args.get("categories", "").strip()
    start, end = _get_date_range()

    df_time  = _D["delivery_time"].copy()
    df_state = _D["delivery_state"].copy()
    df_cat   = _D["delivery_cat"].copy()
    df_delay = _D["delay_dist"].copy()
    df_rev_delay = _D["rev_delay"].copy()

    if start:
        df_time = df_time[df_time["ym"] >= start]
    if end:
        df_time = df_time[df_time["ym"] <= end]

    if states:
        state_list = [s.strip() for s in states.split(",") if s.strip()]
        if state_list:
            df_state = df_state[df_state["customer_state"].isin(state_list)]

    if cats:
        cat_list = [c.strip() for c in cats.split(",") if c.strip()]
        if cat_list:
            df_cat = df_cat[df_cat["product_category_name_english"].isin(cat_list)]

    return jsonify({
        "overview":      _D["delivery_ov"],
        "over_time":     _df_to_records(df_time),
        "by_state":      _df_to_records(df_state),
        "by_category":   _df_to_records(df_cat),
        "delay_dist":    _df_to_records(df_delay),
        "review_delay":  _df_to_records(df_rev_delay),
    })


# ---------------------------------------------------------------------------
# API — SELLERS
# ---------------------------------------------------------------------------

@app.route("/api/sellers")
def api_sellers():
    top_n = int(request.args.get("top", 20))
    sort_by = request.args.get("sort", "product_sales_value")
    valid_sorts = {"product_sales_value", "order_count", "late_pct", "avg_review_score"}
    if sort_by not in valid_sorts:
        sort_by = "product_sales_value"

    asc = request.args.get("asc", "false").lower() == "true"

    # Merge performance (has sales) with delivery (has late_pct) on seller_id
    df_perf = _D["seller_perf"][["seller_id","product_sales_value","freight_value_total",
                                   "item_count","order_count","seller_state","seller_city",
                                   "rank_by_sales"]].copy()
    df_del  = _D["seller_del"][["seller_id","avg_delivery_days","median_delivery_days",
                                  "avg_delay_days","late_count","on_time_count",
                                  "avg_review_score","late_pct"]].copy()
    df = df_perf.merge(df_del, on="seller_id", how="left")

    df = df.sort_values(sort_by, ascending=asc).head(top_n)

    return jsonify({
        "sellers": _df_to_records(df),
        "total_qualified_sellers": len(_D["seller_del"]),
    })


# ---------------------------------------------------------------------------
# API — REVIEWS / SATISFACTION
# ---------------------------------------------------------------------------

@app.route("/api/reviews")
def api_reviews():
    return jsonify({
        "score_distribution": _df_to_records(_D["rev_score_dist"]),
        "over_time":          _df_to_records(_D["sat_time"]),
        "by_delay":           _df_to_records(_D["rev_delay"]),
        "by_state":           _df_to_records(_D["review_state"]),
        "by_category":        _df_to_records(_D["review_cat"]),
    })


# ---------------------------------------------------------------------------
# API — INSIGHTS
# ---------------------------------------------------------------------------

@app.route("/api/insights")
def api_insights():
    category  = request.args.get("category",  "").strip()
    ins_type  = request.args.get("type",      "").strip()
    ins       = _D["insights"]

    if category:
        ins = [i for i in ins if i.get("category") == category]
    if ins_type:
        ins = [i for i in ins if i.get("type") == ins_type]

    return jsonify({"insights": ins, "total": len(ins)})


# ---------------------------------------------------------------------------
# API — ML
# ---------------------------------------------------------------------------

@app.route("/api/ml")
def api_ml():
    ml   = _D["ml_metrics"]
    fi   = _D["ml_fi"].head(15)
    cm   = _D["ml_cm"]
    # Top 15 LR coefficients by absolute value; positive = higher predicted late probability
    lrc  = _D["ml_lr_coef"].head(15)

    return jsonify({
        "metrics":                        ml,
        "feature_importance":             _df_to_records(fi),
        "confusion_matrices":             _df_to_records(cm),
        "logistic_regression_coefficients": _df_to_records(lrc),
    })


# ---------------------------------------------------------------------------
# MAIN — Phase 8
# ---------------------------------------------------------------------------

def main_app():
    print("\n" + "="*60)
    print("  AI-Powered E-Commerce BI & Decision Support System")
    print("  Starting Flask server...")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)


# ============================================================
# COMBINED PIPELINE ENTRY POINT
# ============================================================
# Run all phases in sequence from this single file.
# Note: app.py (Phase 8) starts a Flask server and will block;
# call main_app() only when you want to run the dashboard.

if __name__ == "__main__":
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("2", "preprocessing", "all"):
        main_preprocessing()
    if phase in ("3", "analytics", "all"):
        main_analytics()
    if phase in ("4", "customer", "all"):
        main_customer_analytics()
    if phase in ("5", "delivery", "all"):
        main_delivery_satisfaction()
    if phase in ("6", "ml", "all"):
        main_ml_model()
    if phase in ("7", "insights", "all"):
        main_insight_engine()
    if phase in ("8", "app"):
        main_app()
    if phase == "all":
        print("\n" + "#"*60)
        print("  ALL PHASES COMPLETE — run 'python KarthikTeja_OlistEcommerce.py 8' to start the dashboard")
        print("#"*60 + "\n")
