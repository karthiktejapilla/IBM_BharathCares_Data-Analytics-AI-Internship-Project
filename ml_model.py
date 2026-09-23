"""
ml_model.py
===========
Phase 6 — Late-Delivery Prediction Model

==================================================
PREDICTION SCENARIO
==================================================
Prediction point: immediately AFTER order approval.

At this moment, the system knows:
  - The order has been placed and approved
  - The estimated delivery date has been communicated to the customer
  - The items, their weights and volumes, freight cost, and the seller are known
  - Payment type and installments are confirmed
  - The customer's delivery state is known

The question answered by the model is:
  "Given what we know at the moment of approval, is this order likely to
   arrive after its promised estimated delivery date?"

This prediction can be used operationally to flag high-risk orders for
proactive logistics attention BEFORE delivery problems materialise.

==================================================
TARGET DEFINITION
==================================================
  is_late = 1  if order_delivered_customer_date > order_estimated_delivery_date
  is_late = 0  otherwise

Scope: delivered orders only (order_status == 'delivered') with non-null
delivery dates (excludes 8 records with missing delivery date).

==================================================
DATA SPLITTING
==================================================
TIME-BASED SPLIT (not random) to respect the temporal structure of the data
and to prevent future information from contaminating the training signal.

  Training set : orders placed BEFORE the 80th percentile timestamp (~2018-05-26)
  Test set     : orders placed ON OR AFTER that date

Rationale for 80/20:
  The 80th-percentile cutoff (~2018-05-26) gives ~77,000 training records and
  ~19,000 test records — large enough in both sets to produce reliable metrics.

Note on late-rate shift:
  Train late rate: ~7.59%  |  Test late rate: ~3.49%
  The two worst crisis months (Feb–Mar 2018, up to 19% late) fall inside the
  training window, so the test set is from a relatively better-performing period.
  This is the honest representation of the temporal data; it is NOT a data
  quality problem, and results are interpreted with this context.

==================================================
FEATURE AVAILABILITY CLASSIFICATION
==================================================
Every candidate feature is classified as:
  A = Available at order placement / approval
  B = Available after order placement but before delivery
  C = Available only after or during delivery (EXCLUDED)

  Feature                      Class  Reason
  ---------------------------  -----  ------------------------------------------
  days_to_estimated_delivery     A    Derived from purchase_ts & estimated_date,
                                      both known at approval
  total_freight_value            A    Confirmed at order placement
  avg_product_weight_g           A    Product attribute, known at placement
  avg_product_volume_cm3         A    Product attribute, known at placement
  seller_state                   A    Seller location, known at placement
  customer_state                 A    Customer address, known at placement
  payment_installments           A    Confirmed at payment
  payment_type                   A    Confirmed at payment
  order_item_count               A    Known at placement
  purchase_month                 A    Derived from purchase timestamp
  purchase_dayofweek             A    Derived from purchase timestamp
  unique_sellers_in_order        A    Known at placement (multi-seller orders)
  approval_lag_hours             B    Known only after approval; some orders
                                      approved instantly, some hours later;
                                      included since prediction point is POST-
                                      approval and lag is a proxy for processing
                                      complexity/bottlenecks
  ---------------------------  -----  ------------------------------------------
  delivery_days                  C    EXCLUDED — derived from actual delivery date
  delivery_delay_days            C    EXCLUDED — derived from actual delivery date
  actual_delivery_date           C    EXCLUDED — target leakage
  review_score                   C    EXCLUDED — post-delivery information
  review_comment                 C    EXCLUDED — post-delivery information
  order_delivered_*_date         C    EXCLUDED — post-delivery timestamps

Run:
    D:\python\python.exe ml_model.py

Outputs (data/ directory):
    ml_model_metrics.json       — all model evaluation metrics
    ml_feature_importance.csv   — Random Forest feature importances
    ml_confusion_matrix.csv     — confusion matrix values for both models
    ml_risk_bands.csv           — test-set risk band summary
    ml_model.pkl                — saved Random Forest pipeline (for dashboard)
"""

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
DATA_DIR  = "data"
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

def save_outputs(lr_metrics, rf_metrics, fi_df, lr_fi, risk_summary,
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
# MAIN
# ---------------------------------------------------------------------------

def main():
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

    save_outputs(lr_metrics, rf_metrics, fi_df, lr_fi, risk_summary,
                 train, test, lr_pipeline, rf_pipeline, cutoff_ts)

    print("\n" + "#"*60)
    print("  PHASE 6 COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
