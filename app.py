"""
app.py
======
AI-Powered E-Commerce Business Intelligence & Decision Support System
Flask Application — Phase 8

ARCHITECTURE
------------
- All analytical data is loaded once at startup into module-level DataFrames/dicts.
- Raw Olist CSVs and the 1M-row geolocation file are NEVER loaded by this app.
- All /api/* endpoints return JSON consumed by the frontend Plotly/JS layer.
- Filters are applied server-side before serialisation.
- No analytical recalculations are performed on request — only filtering/slicing
  of the precomputed Phase 2-7 outputs.

Run:
    D:\\python\\python.exe app.py
Then open: http://localhost:5000
"""

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
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# DATA LOADING  (once at startup — never re-loaded per request)
# ---------------------------------------------------------------------------

def _load():
    """Load all precomputed analytical outputs into memory."""
    D = DATA_DIR
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
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI-Powered E-Commerce BI & Decision Support System")
    print("  Starting Flask server...")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
