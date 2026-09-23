# AI-Powered E-Commerce Business Intelligence & Decision Support System

> **Internship Project** — IBM Data Analytics  
> Dataset: Brazilian E-Commerce Public Dataset by Olist (Kaggle)

---

## Problem Statement

Raw e-commerce transaction data contains valuable signals that are invisible without systematic analysis. This project transforms the Olist dataset — covering 100,000 orders, 3,000 sellers, and 96,000 customers across Brazil — into actionable business intelligence through data preprocessing, feature engineering, statistical analysis, machine learning, and an interactive BI dashboard.

---

## Objectives

1. Clean and validate all raw Olist data with documented transformations
2. Calculate meaningful business KPIs (revenue, orders, customers, delivery, satisfaction)
3. Perform customer segmentation using RFM analysis
4. Analyse delivery performance by region, seller, and product category
5. Investigate the observed relationship between delivery timing and customer review scores
6. Build a late-delivery prediction model using only pre-delivery information
7. Generate a structured business insight layer (FACT → INSIGHT → RISK/OPPORTUNITY → ACTION)
8. Deliver an interactive 4-page web dashboard for business users

---

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| File | Rows | Description |
|---|---|---|
| `olist_customers_dataset.csv` | 99,441 | Customer master (unique ID, zip, state) |
| `olist_orders_dataset.csv` | 99,441 | Order lifecycle (status, 4 timestamps) |
| `olist_order_items_dataset.csv` | 112,650 | Line items (product, seller, price, freight) |
| `olist_order_payments_dataset.csv` | 103,886 | Payment type, installments, value |
| `olist_order_reviews_dataset.csv` | 99,224 | Review score (1–5), optional comment |
| `olist_products_dataset.csv` | 32,951 | Product attributes, category, dimensions |
| `olist_sellers_dataset.csv` | 3,095 | Seller location |
| `olist_geolocation_dataset.csv` | 1,000,163 | Zip-code → lat/lng mapping |
| `product_category_name_translation.csv` | 71 | Portuguese → English category names |

**Important definitions used throughout the project:**

| Term | Definition |
|---|---|
| `customer_unique_id` | Person-level identifier (used for all customer KPIs) |
| `customer_id` | Order-level identifier — **not** used for customer counts |
| Product Sales | `SUM(price)` |
| Freight Value | `SUM(freight_value)` |
| Order Value | `Product Sales + Freight Value` |
| `is_late` | `actual_delivery_date > estimated_delivery_date` |
| Delivered scope | All analyses restricted to `order_status = 'delivered'` unless stated |

---

## Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualisation | Plotly |
| Web framework | Flask |
| Frontend | HTML, CSS, JavaScript |

---

## Project Architecture

```
archive/
├── app.py                          ← Flask application (API + page routes)
├── preprocessing.py                ← Phase 2: data loading, cleaning, joins, RFM
├── analytics.py                    ← Phase 3: KPIs, category/state/seller aggregates
├── customer_analytics.py           ← Phase 4: RFM segments, geography, spending
├── delivery_satisfaction.py        ← Phase 5: delivery deep-dive, review analysis
├── ml_model.py                     ← Phase 6: late-delivery prediction model
├── insight_engine.py               ← Phase 7: business insight consolidation
├── test_pipeline.py                ← Phase 9: 159-test end-to-end test suite
├── test_app.py                     ← Phase 8: dashboard API validation
├── requirements.txt
├── README.md
├── templates/
│   └── index.html                  ← Single-page dashboard shell
├── static/
│   ├── style.css
│   └── script.js
└── data/                           ← Generated outputs (never overwrites raw CSVs)
    ├── orders_master.csv
    ├── orders_full.csv
    ├── customer_rfm.csv
    ├── kpis.json
    ├── insights_master.json
    ├── ml_model.pkl
    └── ... (28 total analytical files)
```

**The original Olist CSV files are never modified.** All outputs are written to `data/`.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or extract the project into a directory
# 2. Place all Olist CSV files in the project root (same directory as app.py)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete preprocessing pipeline (generates data/ outputs)
python preprocessing.py
python analytics.py
python customer_analytics.py
python delivery_satisfaction.py
python ml_model.py
python insight_engine.py

# 5. (Optional) Run the test suite to verify everything
python test_pipeline.py

# 6. Start the dashboard
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## How to Run the Application

```bash
python app.py
```

The server starts on port 5000. Open `http://localhost:5000`.

- All analytical data is loaded from `data/` once at startup.
- The 1M-row geolocation file is **never** loaded by the dashboard.
- No internet connection required after initial `pip install`.

---

## Dashboard Pages

### Page 1 — Executive Overview (`/`)
KPI cards: Total Product Sales, Total Order Value, Total Orders, Delivered Orders, Unique Customers, Average Order Value, Average Review Score, On-Time Delivery %. Charts: monthly sales trend, monthly orders, top-15 categories, sales by state. Full business insight panel (18 insights filterable by risk/opportunity).

### Page 2 — Sales & Product Analysis (`/sales`)
Charts: product sales by category, orders by category, AOV by category, review score by category, monthly sales trend, sales by state, top-20 sellers with delivery performance overlay. Filters: state, category.

### Page 3 — Customer Analytics (`/customers`)
RFM segment pie, revenue by segment, avg/median spend by segment, one-time vs repeat bar chart, customer count and revenue by state, customer spending distribution, interactive RFM detail table.

### Page 4 — Delivery & Risk (`/delivery`)
Delivery KPI cards, performance over time (on-time%/late%/review dual-axis), delay distribution, late% by state (with national benchmark line), late% by category, review score by delay bucket (labelled "Observed Association"), seller delivery performance chart. ML section: model metrics, RF feature importance, LR confusion matrix heatmap. Delivery and satisfaction insight cards.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/kpis` | Top-line scalar KPIs |
| `GET /api/filters` | Available filter values (states, categories, segments) |
| `GET /api/monthly-trends?start=&end=` | Monthly time-series |
| `GET /api/categories?top=&categories=` | Category performance, delivery, reviews |
| `GET /api/states?states=` | State performance, delivery, reviews, geography |
| `GET /api/customers?segments=` | RFM segments, repeat/one-time, geography, spending |
| `GET /api/delivery?states=&categories=&start=&end=` | Full delivery analysis |
| `GET /api/sellers?top=&sort=&asc=` | Seller performance (merged sales + delivery) |
| `GET /api/reviews` | Review distributions and satisfaction analysis |
| `GET /api/insights?category=&type=` | Filtered business insights |
| `GET /api/ml` | ML model metrics, feature importance, confusion matrix |

---

## Business Insights (Phase 7 Summary)

18 structured insights across 5 categories, each containing FACT → INSIGHT → RISK/OPPORTUNITY → RECOMMENDED ACTION, all backed by calculated data.

**Key findings:**
- **97%** of customers placed exactly one delivered order (observed retention pattern)
- **At Risk RFM segment** contains the highest revenue of any segment: R$3.2M (20.8%)
- **Top 3 states (SP, RJ, MG)** = 62.5% of total revenue
- **78.9%** of deliveries arrive >7 days before the estimated date (conservative estimates)
- **AL 21.4%, MA 17.4%, SE 15.2%** late delivery rates — 2–3× the national average
- **Observed association**: orders arriving 4–7 days late average **2.11★** vs 4.31★ for early arrivals
- **office_furniture**: lowest review score among high-volume categories (3.650★, 17.2% 1-star)

---

## ML Methodology (Phase 6)

**Task:** Binary classification — predict whether an order will arrive after its estimated delivery date.

**Prediction point:** At/after order approval, using only information available before any delivery event.

**Target:** `is_late = 1` if `order_delivered_customer_date > order_estimated_delivery_date`

**Split:** Time-based (80/20 by purchase timestamp). Cutoff: 2018-05-26.

**Features (13 total, all class A/B — no post-delivery leakage):**
`days_to_estimated`, `total_freight_value`, `avg_weight_g`, `avg_volume_cm3`, `order_item_count`, `unique_sellers`, `payment_installments`, `approval_lag_hours`, `purchase_month`, `purchase_dayofweek`, `customer_state`, `seller_state`, `payment_type`

**Results on May–August 2018 held-out test period:**

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | **0.7066** | **76.41%** | 6.08% | 0.1126 |
| Random Forest | 0.5184 | 20.92% | 4.77% | 0.0777 |

The LR model is recommended. High recall (76%) means most late orders are identified at the cost of many false positives — appropriate for a proactive monitoring system.

**Important limitation:** The test window (May–Aug 2018) had a 3.49% late rate vs 7.59% in training. These metrics reflect that specific period, not guaranteed future performance.

---

## Testing

Run `python test_pipeline.py` to execute 159 automated tests covering:
- All 47 required files exist
- Raw Olist CSV row counts are unchanged
- KPI numerical accuracy (diff < 0.02)
- Revenue reconciliation (zero double-counting)
- RFM segment totals
- Delivery KPI cross-validation
- ML metric accuracy
- Insight structure and completeness
- All 11 API endpoints (HTTP 200)
- All 5 page routes (HTTP 200)
- Filter functionality
- No hard-coded personal paths

**Result: 159/159 tests pass.**

---

## License

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
