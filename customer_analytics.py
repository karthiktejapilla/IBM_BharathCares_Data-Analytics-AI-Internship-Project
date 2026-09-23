"""
customer_analytics.py
=====================
Phase 4 — Customer Analytics

PURPOSE
-------
Produces all customer-level analytical datasets required by the dashboard:
  - RFM segment summary
  - Repeat vs one-time customer analysis
  - Customer geographic analysis
  - Customer spending distributions
  - Business insights

DEFINITIONS (explicitly documented)
-------------------------------------
customer_id
    Olist's per-ORDER customer identifier.  A single real customer who places
    two orders will have two distinct customer_id values.  NOT used for
    customer-level KPIs.

customer_unique_id
    Olist's true per-PERSON identifier.  Used for all customer counts,
    segments, and spending aggregates throughout this phase.

one-time customer
    A customer_unique_id with exactly ONE delivered order in the dataset.

repeat customer
    A customer_unique_id with TWO OR MORE delivered orders.

RFM reference date
    The day AFTER the latest order_purchase_timestamp in the delivered dataset
    (2018-08-30), inherited from Phase 2.  NOT wall-clock time, so the dataset
    is self-consistent regardless of when it is analysed.

revenue / monetary in customer analysis
    order_value = product_sales_value + freight_value_total per order.
    This is the total amount the customer paid, consistent with Phase 3.
    Labelled "order_value" or "total_spend" to avoid ambiguity.

Scope
    All customer analyses are restricted to DELIVERED orders (order_status ==
    'delivered') unless otherwise noted, for the same reason as Phase 3:
    cancelled/unavailable orders did not generate real revenue.

Run:
    D:\python\python.exe customer_analytics.py

Outputs (data/ directory):
    rfm_segment_summary.csv         — one row per segment
    repeat_vs_onetime.csv           — two rows: one-time / repeat
    customer_geography.csv          — one row per customer_state
    customer_spending_dist.csv      — spending buckets (histogram data)
    customer_segment_detail.csv     — full per-customer RFM + repeat flag
    customer_insights.json          — structured business insights
"""

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_data():
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

def build_insights(rfm, seg_summary, repeat_summary, state, spending_dist, p90_threshold):
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

def save_all(seg_summary, repeat_summary, state, spend_dist, seg_spend,
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "#"*60)
    print("  OLIST CUSTOMER ANALYTICS — PHASE 4")
    print("#"*60)

    master, delivered, rfm = load_data()

    seg_summary = rfm_segment_summary(rfm, delivered)
    repeat_summary = repeat_vs_onetime(rfm, delivered)
    state = customer_geography(delivered, rfm)
    spend_dist, seg_spend, p90_threshold = customer_spending_dist(rfm)
    customer_detail = build_customer_detail(rfm, delivered)
    insights = build_insights(rfm, seg_summary, repeat_summary, state,
                              spend_dist, p90_threshold)

    save_all(seg_summary, repeat_summary, state, spend_dist, seg_spend,
             customer_detail, insights)

    print("\n" + "#"*60)
    print("  PHASE 4 COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
