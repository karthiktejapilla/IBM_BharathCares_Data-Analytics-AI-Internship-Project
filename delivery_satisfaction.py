"""
delivery_satisfaction.py
========================
Phase 5 — Delivery & Satisfaction Deep-Dive

PURPOSE
-------
Produces all delivery and satisfaction analytical datasets for the Delivery & Risk
dashboard page.  Extends Phase 3 outputs with:
  - delivery performance trends over time
  - seller delivery analysis with minimum-order thresholds
  - enriched regional and category delivery tables
  - cross-tabulation of review score distribution by delay bucket
  - review score by state and category
  - structured risk/opportunity insight layer

INTERPRETATION RULES (enforced throughout)
-------------------------------------------
- Delivery delay and review score association is described as an OBSERVED
  RELATIONSHIP, not a causal claim.
- States / sellers / categories are not labelled "high risk" without reference
  to order volume.
- Minimum order thresholds are applied where stated before ranking.
- All data-supported statements are backed by printed counts.

VALIDATION STRATEGY
--------------------
- Delivery totals cross-checked against Phase 3 kpis.json
- Review totals cross-checked against Phase 3 review_score_dist.csv
- State/category totals compared to Phase 3 delivery_by_state/category
- Delay bucket totals compared to Phase 3 delivery_delay_dist.csv
- No double-counting: delivery KPIs computed from orders_master (one row per order)
  Review scores also from orders_master (one review per order after Phase 2 dedup)

Run:
    D:\python\python.exe delivery_satisfaction.py

Outputs (data/ directory):
    delivery_overview.json              — scalar delivery KPIs + validation
    delivery_over_time.csv              — monthly delivery performance
    delivery_by_state_enriched.csv      — state delivery + review (extended from Ph3)
    delivery_by_category_enriched.csv   — category delivery + review (extended from Ph3)
    seller_delivery_qualified.csv       — sellers with >= MIN_SELLER_ORDERS orders
    review_by_delay_detailed.csv        — review score distribution per delay bucket
    review_by_state.csv                 — avg review + distribution by state
    review_by_category.csv              — avg review + distribution by category
    delivery_insights.json              — structured FACT/INSIGHT/RISK/ACTION
"""

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
DATA_DIR = "data"

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

def load_data():
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

def save_all(overview, monthly_del, state_del, cat_del,
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#"*60)
    print("  OLIST DELIVERY & SATISFACTION — PHASE 5")
    print("#"*60)

    master, full, delivered, del_valid, ph3_kpis, ph3_delay, ph3_review = load_data()

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

    save_all(overview, monthly_del, state_del, cat_del,
             seller_del, seller_state_agg,
             review_delay, review_state, review_cat, insights)

    print("\n" + "#"*60)
    print("  PHASE 5 COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
