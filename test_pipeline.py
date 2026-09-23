"""
test_pipeline.py
================
Phase 9 — End-to-End Project Test Suite

Tests every analytical pipeline module (preprocessing → analytics → customer →
delivery → ML → insights) and all dashboard API endpoints.

Run:
    D:\python\python.exe test_pipeline.py
"""

import os, sys, json, math, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
DATA_DIR = "data"

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =========================================================
# 1. DATA FILE EXISTENCE
# =========================================================
section("1. DATA FILE EXISTENCE")

required_files = [
    # Phase 2
    "data/orders_master.csv",
    "data/orders_full.csv",
    "data/customer_rfm.csv",
    # Phase 3
    "data/kpis.json",
    "data/monthly_trends.csv",
    "data/category_performance.csv",
    "data/state_performance.csv",
    "data/seller_performance.csv",
    "data/delivery_delay_dist.csv",
    "data/delivery_by_state.csv",
    "data/delivery_by_category.csv",
    "data/review_score_dist.csv",
    "data/review_vs_delay.csv",
    "data/satisfaction_over_time.csv",
    # Phase 4
    "data/rfm_segment_summary.csv",
    "data/repeat_vs_onetime.csv",
    "data/customer_geography.csv",
    "data/customer_spending_dist.csv",
    "data/segment_spend_profile.csv",
    "data/customer_segment_detail.csv",
    "data/customer_insights.json",
    # Phase 5
    "data/delivery_overview.json",
    "data/delivery_over_time.csv",
    "data/delivery_by_state_enriched.csv",
    "data/delivery_by_category_enriched.csv",
    "data/seller_delivery_qualified.csv",
    "data/review_by_delay_detailed.csv",
    "data/review_by_state.csv",
    "data/review_by_category.csv",
    "data/delivery_insights.json",
    # Phase 6
    "data/ml_model_metrics.json",
    "data/ml_feature_importance.csv",
    "data/ml_confusion_matrix.csv",
    "data/ml_risk_bands.csv",
    "data/ml_model.pkl",
    # Phase 7
    "data/insights_master.json",
    # App files
    "app.py",
    "templates/index.html",
    "static/style.css",
    "static/script.js",
    "preprocessing.py",
    "analytics.py",
    "customer_analytics.py",
    "delivery_satisfaction.py",
    "ml_model.py",
    "insight_engine.py",
]

for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))

# =========================================================
# 2. RAW DATA NOT MODIFIED
# =========================================================
section("2. RAW OLIST FILES UNTOUCHED")

raw_files = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]
expected_rows = {
    "olist_customers_dataset.csv":       99441,
    "olist_orders_dataset.csv":          99441,
    "olist_order_items_dataset.csv":     112650,
    "olist_order_payments_dataset.csv":  103886,
    "olist_order_reviews_dataset.csv":   99224,
    "olist_products_dataset.csv":        32951,
    "olist_sellers_dataset.csv":         3095,
}
for f in raw_files:
    if os.path.exists(f):
        if f in expected_rows:
            # Use pandas to count rows properly (handles multi-line quoted fields)
            try:
                rows = len(pd.read_csv(f, encoding="utf-8", low_memory=False))
            except Exception:
                rows = -1
            check(f"Row count unchanged: {f}", rows == expected_rows[f],
                  f"expected {expected_rows[f]}, got {rows}")
        else:
            check(f"File exists: {f}", True)

# =========================================================
# 3. ORDERS MASTER VALIDATION
# =========================================================
section("3. ORDERS MASTER — GRAIN & COUNTS")

master = pd.read_csv(f"{DATA_DIR}/orders_master.csv",
                     parse_dates=["order_purchase_timestamp"])
delivered = master[master["order_status"] == "delivered"]

check("orders_master: no duplicate order_ids",
      master["order_id"].duplicated().sum() == 0)
check("orders_master: row count = 99,441",
      len(master) == 99441, f"got {len(master)}")
check("orders_master: delivered count = 96,478",
      len(delivered) == 96478, f"got {len(delivered)}")
check("orders_master: order_value = product_sales + freight",
      (abs(master["order_value"] - (master["product_sales_value"] + master["freight_value_total"])).dropna() < 0.01).all())
check("orders_master: is_late only for delivered",
      master[master["order_status"] != "delivered"]["is_late"].isna().all())

# =========================================================
# 4. KPI VALIDATION
# =========================================================
section("4. KPI CROSS-VALIDATION")

kpis = json.load(open(f"{DATA_DIR}/kpis.json"))

check("KPI total_product_sales = 13,221,498.11",
      abs(kpis["total_product_sales"] - 13221498.11) < 0.02)
check("KPI total_order_value = 15,419,773.75",
      abs(kpis["total_order_value"] - 15419773.75) < 0.02)
check("KPI total_orders = 99,441",
      kpis["total_orders"] == 99441)
check("KPI total_delivered_orders = 96,478",
      kpis["total_delivered_orders"] == 96478)
check("KPI total_customers = 96,096",
      kpis["total_customers"] == 96096)
check("KPI avg_order_value rounds to 159.83",
      abs(kpis["avg_order_value"] - 159.83) < 0.01)
check("KPI on_time_delivery_pct = 93.23",
      abs(kpis["on_time_delivery_pct"] - 93.23) < 0.01)
check("KPI late + on_time = 100",
      abs(kpis["on_time_delivery_pct"] + kpis["late_delivery_pct"] - 100) < 0.01)

# Cross-check: product sales from orders_master
master_sales = delivered["product_sales_value"].sum()
check("KPI product_sales matches orders_master delivered sum",
      abs(master_sales - kpis["total_product_sales"]) < 0.02)

# =========================================================
# 5. CATEGORY PERFORMANCE — NO DOUBLE COUNT
# =========================================================
section("5. CATEGORY PERFORMANCE — REVENUE RECONCILIATION")

cat = pd.read_csv(f"{DATA_DIR}/category_performance.csv")
check("Category rows = 72", len(cat) == 72, f"got {len(cat)}")
check("Category sales sum matches KPI total",
      abs(cat["product_sales_value"].sum() - kpis["total_product_sales"]) < 0.02)
check("Category rank_by_sales is 1-based sequential",
      cat["rank_by_sales"].min() == 1)

# =========================================================
# 6. RFM VALIDATION
# =========================================================
section("6. RFM SEGMENT VALIDATION")

rfm = pd.read_csv(f"{DATA_DIR}/customer_rfm.csv")
rfm_seg = pd.read_csv(f"{DATA_DIR}/rfm_segment_summary.csv")

check("RFM: no duplicate customer_unique_id",
      rfm["customer_unique_id"].duplicated().sum() == 0)
check("RFM: 93,358 customers",
      len(rfm) == 93358, f"got {len(rfm)}")
check("RFM segment total customers = 93,358",
      rfm_seg["customer_count"].sum() == 93358)
check("RFM segment revenue matches KPI total_order_value",
      abs(rfm_seg["total_revenue"].sum() - kpis["total_order_value"]) < 1.0)
check("RFM: 7 segments",
      rfm["segment"].nunique() == 7, f"got {rfm['segment'].nunique()}")
check("RFM: monetary = rfm_seg total_revenue (reconciled)",
      abs(rfm["monetary"].sum() - rfm_seg["total_revenue"].sum()) < 1.0)

# =========================================================
# 7. REPEAT VS ONE-TIME
# =========================================================
section("7. REPEAT VS ONE-TIME CUSTOMERS")

repeat = pd.read_csv(f"{DATA_DIR}/repeat_vs_onetime.csv")
check("Repeat rows = 2", len(repeat) == 2)
total_cust = repeat["customer_count"].sum()
check("One-time + Repeat = 93,358",
      total_cust == 93358, f"got {total_cust}")
total_rev = repeat["total_revenue"].sum()
check("Repeat revenue sum matches KPI",
      abs(total_rev - kpis["total_order_value"]) < 1.0)
ot = repeat[repeat["customer_type"] == "One-Time"].iloc[0]
check("One-time customer_pct ≈ 97%",
      abs(ot["customer_pct"] - 97.0) < 0.2)

# =========================================================
# 8. DELIVERY VALIDATION
# =========================================================
section("8. DELIVERY ANALYSIS VALIDATION")

del_ov = json.load(open(f"{DATA_DIR}/delivery_overview.json"))
check("Delivery on_time_pct matches KPI",
      abs(del_ov["on_time_pct"] - kpis["on_time_delivery_pct"]) < 0.05)
check("Delivery late_pct matches KPI",
      abs(del_ov["late_pct"] - kpis["late_delivery_pct"]) < 0.05)
check("Delivery n_orders = 96,470",
      del_ov["n_orders"] == 96470, f"got {del_ov['n_orders']}")

delay_dist = pd.read_csv(f"{DATA_DIR}/delivery_delay_dist.csv")
check("Delay dist total = 96,470",
      delay_dist["order_count"].sum() == 96470)
check("Delay dist has 7 buckets",
      len(delay_dist) == 7)

del_state = pd.read_csv(f"{DATA_DIR}/delivery_by_state_enriched.csv")
check("State delivery rows = 27",
      len(del_state) == 27, f"got {len(del_state)}")

# Review vs delay: monotonicity check
rev_delay = pd.read_csv(f"{DATA_DIR}/review_by_delay_detailed.csv")
scores = rev_delay["avg_review_score"].tolist()
check("Review scores decrease from Early>7d to Late 4-7d",
      scores[0] > scores[3] > scores[4])

# =========================================================
# 9. ML MODEL VALIDATION
# =========================================================
section("9. ML MODEL VALIDATION")

ml = json.load(open(f"{DATA_DIR}/ml_model_metrics.json"))
lr = ml["logistic_regression"]
rf = ml["random_forest"]

check("LR ROC-AUC = 0.7066",
      abs(lr["roc_auc"] - 0.7066) < 0.001)
check("LR Recall = 0.7641",
      abs(lr["recall"] - 0.7641) < 0.001)
check("LR Precision = 0.0608",
      abs(lr["precision"] - 0.0608) < 0.001)
check("LR F1 = 0.1126",
      abs(lr["f1"] - 0.1126) < 0.001)
check("RF ROC-AUC = 0.5184",
      abs(rf["roc_auc"] - 0.5184) < 0.001)
check("LR outperforms RF on ROC-AUC",
      lr["roc_auc"] > rf["roc_auc"])
check("ML train_orders = 77,176",
      ml["train_orders"] == 77176)
check("ML test_orders = 19,294",
      ml["test_orders"] == 19294)

ml_fi = pd.read_csv(f"{DATA_DIR}/ml_feature_importance.csv")
check("Feature importance sums to ≈ 1.0",
      abs(ml_fi["importance"].sum() - 1.0) < 0.01)
check("Top feature = purchase_month",
      ml_fi.iloc[0]["feature"] == "purchase_month")

# No target leakage check
forbidden = ["delivery_days","delivery_delay_days","review_score",
             "on_time","order_delivered_customer_date"]
for f in forbidden:
    check(f"Leakage guard: '{f}' not in features",
          f not in ml["features_used"])

# =========================================================
# 10. INSIGHT ENGINE VALIDATION
# =========================================================
section("10. INSIGHT ENGINE VALIDATION")

insights = json.load(open(f"{DATA_DIR}/insights_master.json"))
check("Total insights = 18",
      len(insights) == 18, f"got {len(insights)}")

risks = [i for i in insights if i["type"] == "risk"]
opps  = [i for i in insights if i["type"] == "opportunity"]
check("Risks = 10",        len(risks) == 10)
check("Opportunities = 8", len(opps)  == 8)

cats = set(i["category"] for i in insights)
check("All 5 categories present",
      {"customer","sales","delivery","satisfaction","ml"} == cats)

required_fields = {"id","category","fact","metric","insight","type","recommendation","source"}
for ins in insights:
    missing = required_fields - set(ins.keys())
    check(f"Insight {ins['id']} has all required fields",
          len(missing) == 0, f"missing: {missing}")

# =========================================================
# 11. DASHBOARD API TESTS
# =========================================================
section("11. DASHBOARD API TESTS")

import app as flask_app
with flask_app.app.test_client() as c:

    # All endpoints return 200
    endpoints = [
        "/api/filters", "/api/kpis", "/api/monthly-trends",
        "/api/categories", "/api/states", "/api/customers",
        "/api/delivery", "/api/sellers", "/api/reviews",
        "/api/insights", "/api/ml",
    ]
    for ep in endpoints:
        r = c.get(ep)
        check(f"API {ep} → 200", r.status_code == 200)

    # All page routes return 200
    for route in ["/", "/executive", "/sales", "/customers", "/delivery"]:
        r = c.get(route)
        check(f"Page {route} → 200", r.status_code == 200)

    # Filter endpoints work
    r = c.get("/api/delivery?states=SP,RJ")
    d = json.loads(r.data)
    check("Filter delivery by SP,RJ: 2 states in response",
          len(d["by_state"]) == 2)

    r = c.get("/api/categories?top=5")
    d = json.loads(r.data)
    check("Filter categories top=5: <=5 results",
          len(d["performance"]) <= 5)

    r = c.get("/api/insights?category=delivery")
    d = json.loads(r.data)
    check("Filter insights by category=delivery: all delivery",
          all(i["category"] == "delivery" for i in d["insights"]))

    r = c.get("/api/insights?type=risk")
    d = json.loads(r.data)
    check("Filter insights by type=risk: all risks",
          all(i["type"] == "risk" for i in d["insights"]))

    # KPI values match Phase 3
    kpi_r = json.loads(c.get("/api/kpis").data)
    check("Dashboard KPI total_product_sales matches",
          abs(kpi_r["total_product_sales"] - 13221498.11) < 0.02)
    check("Dashboard KPI on_time_delivery_pct matches",
          abs(kpi_r["on_time_delivery_pct"] - 93.23) < 0.01)

    # ML endpoint has both models
    ml_r = json.loads(c.get("/api/ml").data)
    check("ML endpoint has logistic_regression",
          "logistic_regression" in ml_r["metrics"])
    check("ML endpoint has random_forest",
          "random_forest" in ml_r["metrics"])
    check("ML endpoint has feature_importance",
          len(ml_r["feature_importance"]) >= 10)

    # Invalid sort parameter handled safely
    r = c.get("/api/sellers?sort=INVALID_COLUMN")
    check("Invalid sort param handled safely (200)",
          r.status_code == 200)

# =========================================================
# 12. NO HARD-CODED PATHS
# =========================================================
section("12. NO HARD-CODED PERSONAL PATHS")

for fname in ["app.py", "preprocessing.py", "analytics.py",
              "customer_analytics.py", "delivery_satisfaction.py",
              "ml_model.py", "insight_engine.py"]:
    if os.path.exists(fname):
        content = open(fname, encoding="utf-8", errors="replace").read()
        has_hardcode = ("C:\\Users\\" in content or
                        "D:\\D\\Interships" in content or
                        "/home/" in content)
        check(f"No hard-coded personal paths in {fname}", not has_hardcode)

# =========================================================
# SUMMARY
# =========================================================
print(f"\n{'='*60}")
print(f"  TEST SUMMARY")
print(f"{'='*60}")
print(f"  PASSED : {PASS}")
print(f"  FAILED : {FAIL}")
print(f"  TOTAL  : {PASS + FAIL}")
if FAIL == 0:
    print(f"\n  ALL TESTS PASSED")
else:
    print(f"\n  {FAIL} TEST(S) FAILED — see details above")
sys.exit(0 if FAIL == 0 else 1)
