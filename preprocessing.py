"""
preprocessing.py
================
Phase 2 — Data Loading, Validation, Preprocessing, Table Joins & Feature Engineering

PURPOSE
-------
Load all nine Olist CSV files, validate them, apply transformations, join tables into
analytical datasets, and persist the results to data/.

ORIGINAL DATA IS NEVER MODIFIED — every output is written to data/ as a separate file.

Run:
    D:\python\python.exe preprocessing.py

Outputs (data/ directory):
    orders_master.csv        — order-level analytical table (one row per order)
    orders_full.csv          — order+item-level table (one row per order-item line)
    customer_rfm.csv         — RFM scores and segments per customer_unique_id
"""

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
# MAIN
# ---------------------------------------------------------------------------

def main():
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


if __name__ == "__main__":
    main()
