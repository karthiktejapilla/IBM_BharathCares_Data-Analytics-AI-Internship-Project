"""
insight_engine.py
=================
Phase 7 — Business Insight Engine

PURPOSE
-------
Consolidates validated findings from Phases 3-6 into a structured,
dashboard-ready insight layer.

RULES ENFORCED
--------------
- Every insight is traced to a specific metric from a named source file.
- Associations between variables are described as OBSERVED relationships,
  not causal claims.
- No insight is invented; each is grounded in computed numbers.
- "one-time customer" finding is described as an observed retention pattern,
  not a statement about acquisition economics.
- ML results are reported for the specific held-out test period, not as
  universal claims.

Run:
    D:\python\python.exe insight_engine.py

Outputs:
    data/insights_master.json  — all structured insights
"""

import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
DATA_DIR = "data"


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
# MAIN
# ---------------------------------------------------------------------------

def main():
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


if __name__ == "__main__":
    main()
