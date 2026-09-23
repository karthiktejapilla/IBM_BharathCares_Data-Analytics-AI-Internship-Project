"""
analytics.py
============
Phase 3 — KPI Calculation and Analytics Layer

PURPOSE
-------
Load the Phase-2 processed data and compute all business KPIs and analytical
aggregations required by the dashboard.  Every metric is documented with:
  - grain (what one row represents)
  - source table and column(s)
  - formula
  - validation check

ANTI-DOUBLE-COUNT RULES
-----------------------
1. Order-level KPIs (revenue, order count, AOV) use orders_master with DISTINCT
   order_id.  Revenue is NEVER summed from orders_full (which has one row per item).
2. Product/category revenue is calculated from orders_full at item level (price per item
   row) — which IS the correct grain for category splits.
3. Customer counts always use customer_unique_id, NOT customer_id.
4. Delivery KPIs are restricted to order_status == 'delivered'.

Run:
    D:\python\python.exe analytics.py

Outputs (data/ directory):
    kpis.json                   — top-line scalar KPIs
    monthly_trends.csv          — monthly time-series aggregates
    category_performance.csv    — per-category aggregates
    state_performance.csv       — per-customer-state aggregates
    seller_performance.csv      — per-seller aggregates
    delivery_analysis.csv       — delivery statistics (same grain as state/category)
    review_delivery.csv         — review score vs delivery delay buckets
    satisfaction_over_time.csv  — monthly avg review score
"""

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = "data"
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

def save_all(kpis, monthly, category, state, sellers,
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
# MAIN
# ---------------------------------------------------------------------------

def main():
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

    save_all(kpis, monthly, category, state, sellers,
             delay_dist, state_del, cat_del, score_dist, sat_time, rev_delay)

    print("\n" + "#"*60)
    print("  PHASE 3 COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
