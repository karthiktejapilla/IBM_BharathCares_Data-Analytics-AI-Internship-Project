# Project Report
## AI-Powered E-Commerce Business Intelligence & Decision Support System

**Internship:** IBM Data Analytics  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist  
**Dataset Link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
**Analysis Period:** September 2016 – August 2018  

---

## 1. Project Overview

This project converts raw Olist e-commerce transaction data into a production-quality Business Intelligence system. The pipeline runs from raw CSV ingestion through data cleaning, feature engineering, statistical analysis, machine learning, and a fully interactive web dashboard.

The project answers seven business-critical questions:
1. What is happening? (KPIs, trends)
2. Why is it happening? (driver analysis)
3. What are the important trends? (time-series)
4. What are the major drivers? (category, region, seller)
5. What risks exist? (delivery failures, retention, concentration)
6. What opportunities exist? (retention, regional expansion, logistics)
7. What actions should the business take? (structured recommendations)

---

## 2. Dataset Structure

Nine source CSV files covering the complete order lifecycle:

| Table | Rows | Key Columns |
|---|---|---|
| Customers | 99,441 | customer_id, customer_unique_id, state |
| Orders | 99,441 | order_id, customer_id, status, 4 timestamps |
| Order Items | 112,650 | order_id, product_id, seller_id, price, freight |
| Payments | 103,886 | order_id, payment_type, installments, value |
| Reviews | 99,224 | order_id, review_score (1–5), comment |
| Products | 32,951 | product_id, category (PT), dimensions, weight |
| Sellers | 3,095 | seller_id, zip, city, state |
| Geolocation | 1,000,163 | zip, lat, lng, city, state |
| Category Translation | 71 | PT category → EN category |

**Key relationships:** Orders join to Customers (1:1), to Items (1:N), to Payments (1:N), to Reviews (1:1 after dedup). Items join to Products and Sellers.

---

## 3. Data Preprocessing (Phase 2)

### Review Data Investigation
The reviews table contained 99,224 rows for 98,673 unique order_ids. Investigation found:
- **789 review_ids** appeared across multiple order_ids (system pipeline artefact)
- **547 orders** had 2+ review events — genuine re-reviews over time
- **202 orders** showed score changes between events; the final score averaged 3.16 vs the initial 3.62 (customers who re-reviewed tended to revise downward)

**Aggregation decision:** Keep the LAST review per order (most recent settled sentiment). Justified and documented.

### Key Cleaning Steps
- Date type conversion for all 4 order timestamps
- Zip codes zero-padded to 5 characters
- 1,000,163 geolocation rows deduplicated to 19,015 unique zips (median lat/lng)
- Payments aggregated to one row per order (sum value, dominant type)
- English category translation joined; 3 known typos corrected in EN column only; original PT names preserved
- 8 delivered orders with missing actual delivery dates excluded from timing KPIs

### Referential Integrity
Zero orphan keys across all 6 foreign key relationships.

### Output Tables

| Table | Rows | Grain |
|---|---|---|
| `orders_master.csv` | 99,441 | One row per order |
| `orders_full.csv` | 113,425 | One row per order-item (+ 775 orders with no items) |
| `customer_rfm.csv` | 93,358 | One row per customer_unique_id |

---

## 4. Key Business KPIs (Phase 3)

All KPIs are computed from delivered orders only.

| KPI | Value | Notes |
|---|---|---|
| Total Product Sales | R$13,221,498 | SUM(price), delivered orders |
| Total Freight Value | R$2,198,276 | SUM(freight_value), delivered orders |
| Total Order Value | R$15,419,774 | Sales + Freight |
| Total Orders | 99,441 | All statuses |
| Total Delivered Orders | 96,478 | 97.0% of all orders |
| Unique Customers | 96,096 | By customer_unique_id |
| Average Order Value | R$159.83 | Order value / delivered orders |
| Average Review Score | 4.156 / 5 | 95,832 scored orders |
| On-Time Delivery | 93.23% | Actual ≤ estimated delivery date |
| Late Delivery | 6.77% | Actual > estimated delivery date |
| Average Delivery Days | 12.09 | From purchase to delivery |
| Repeat Customer Rate | 3.00% | ≥2 delivered orders |

**Cross-validation:** Product sales sum independently verified at item level (orders_full) — diff = R$0.00.

---

## 5. Customer Analysis (Phase 4)

### RFM Segmentation

Recency, Frequency, Monetary scored in quintiles. Reference date: 2018-08-30.

| Segment | Customers | % | Revenue (R$) | % | Avg Spend |
|---|---|---|---|---|---|
| Needs Attention | 23,607 | 25.3% | 3,648,478 | 23.7% | R$155 |
| At Risk | 13,162 | 14.1% | **3,205,152** | **20.8%** | R$244 |
| Loyal Customers | 14,210 | 15.2% | 2,942,647 | 19.1% | R$207 |
| New Customers | 14,984 | 16.1% | 2,448,694 | 15.9% | R$163 |
| Champions | 6,497 | 7.0% | 2,027,030 | 13.2% | R$312 |
| Potential Loyalists | 14,588 | 15.6% | 795,736 | 5.2% | R$55 |
| Lost / Inactive | 6,310 | 6.8% | 352,035 | 2.3% | R$56 |

Key finding: **At Risk** contains the most revenue of any segment (R$3.2M), making it the highest-value win-back target.

### Repeat vs One-Time

| Type | Customers | % | Revenue | % | Avg Total Spend |
|---|---|---|---|---|---|
| One-Time | 90,557 | 97.0% | R$14,555,586 | 94.4% | R$160.73 |
| Repeat | 2,801 | 3.0% | R$864,187 | 5.6% | R$308.53 |

97% of customers placed exactly one order. This is an observed retention pattern; the dataset does not contain acquisition cost data.

### Geographic Distribution
- SP: 41.9% of customers, 37.4% of revenue
- Top 3 states (SP, RJ, MG): 62.5% of revenue

### Spending Concentration
Top 10% of customers (spend ≥ R$318) contribute 38.3% of revenue.

---

## 6. Delivery & Satisfaction Analysis (Phase 5)

### Overall Performance
- **78.9%** of deliveries arrive >7 days before the estimated date
- Average order arrives 11.88 days before estimate (median 12 days early)
- Estimated delivery dates are systematically conservative

### Regional Performance (High-Impact States, ≥200 orders)

| State | Late % | vs National | Avg Days | Avg Review |
|---|---|---|---|---|
| AL | 21.4% | +14.6pp | 24.0 | 3.848 |
| MA | 17.4% | +10.7pp | 21.1 | 3.833 |
| SE | 15.2% | +8.4pp | 21.0 | 3.907 |
| PI | 13.9% | +7.1pp | 19.0 | 3.994 |
| CE | 13.8% | +7.0pp | 20.8 | 3.944 |
| RJ | 12.1% | +5.3pp | 14.8 | 3.965 |

### Crisis Months
Three months with >10% late rates were identified:
- **March 2018**: 19.0% late (7,003 orders), avg review 3.813
- **February 2018**: 14.1% late (6,555 orders), avg review 3.881
- **November 2017**: 12.4% late (7,288 orders), avg review 3.988

### Delay vs Review Score (OBSERVED ASSOCIATION)

| Delay Bucket | Avg Review | 1★ % | 5★ % | n |
|---|---|---|---|---|
| Early >7d | 4.313 | 6.5% | 63.4% | 75,744 |
| Early 3-7d | 4.191 | 7.0% | 57.2% | 9,429 |
| Early 0-3d | 4.108 | 8.1% | 54.0% | 4,270 |
| Late 1-3d | 3.291 | 25.1% | 33.2% | 1,852 |
| Late 4-7d | 2.105 | 58.6% | 14.1% | 1,748 |
| Late 8-14d | 1.671 | 70.6% | 6.8% | 1,446 |
| Late >14d | 1.723 | 69.1% | 7.2% | 1,335 |

This is an **observed association** in the data. No causal analysis was performed.

### Seller Analysis
76 of 425 qualified sellers (≥50 orders) have late rates ≥1.5× the national average, handling 10,357 combined orders.

---

## 7. Machine Learning (Phase 6)

### Prediction Scenario
**Predict at order approval** whether the order will arrive after its estimated delivery date, using only information available before any delivery event.

### Features Used (13, all pre-delivery)
`days_to_estimated`, `total_freight_value`, `avg_weight_g`, `avg_volume_cm3`, `order_item_count`, `unique_sellers`, `payment_installments`, `approval_lag_hours`, `purchase_month`, `purchase_dayofweek`, `customer_state`, `seller_state`, `payment_type`

**Explicitly excluded (post-delivery):** `delivery_days`, `delivery_delay_days`, `review_score`, `order_delivered_customer_date`, `on_time`

### Time-Based Split
- Training: 77,176 orders (Sep 2016 – May 2018), late rate 7.59%
- Test: 19,294 orders (May–Aug 2018), late rate 3.49%

### Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.5793 | 0.0608 | **0.7641** | 0.1126 | **0.7066** |
| Random Forest | 0.8264 | 0.0477 | 0.2092 | 0.0777 | 0.5184 |

**Logistic Regression is the recommended model** (ROC-AUC 0.7066 vs 0.5184 for RF).

The RF's near-random AUC reflects a temporal distribution shift: the test window (May–Aug 2018) had lower late rates than the training period including the Feb–Mar 2018 crisis. Accuracy alone is not a meaningful metric for this imbalanced task.

### Top Predictive Features (RF — association, not causation)
1. `purchase_month` (0.213) — strongest seasonality signal
2. `customer_state_SP` (0.152) — destination associated with lower risk
3. `total_freight_value` (0.128) — routing complexity proxy
4. `customer_state_RJ` (0.098) — destination associated with higher risk
5. `days_to_estimated` (0.093) — window length

### Business Use
The LR model serves as an **early-warning mechanism** at order approval. Its 76.41% recall means most at-risk orders can be identified before delivery. It cannot prevent delays, only flag elevated predicted risk.

---

## 8. Business Insight Engine (Phase 7)

18 structured insights across 5 categories, each following:
`FACT → INSIGHT → RISK/OPPORTUNITY → RECOMMENDED ACTION`

| Category | Count | Risks | Opportunities |
|---|---|---|---|
| Customer | 5 | 2 | 3 |
| Sales | 3 | 2 | 1 |
| Delivery | 5 | 4 | 1 |
| Satisfaction | 3 | 3 | 0 |
| ML | 2 | 0 | 2 |
| **Total** | **18** | **10** | **8** |

All insights are traceable to calculated data. No causal claims are made without supporting analysis.

---

## 9. Dashboard Architecture

**Technology:** Flask (Python), Plotly (JS), HTML/CSS/JavaScript

**Pattern:** API-first — all data served via `/api/*` JSON endpoints, rendered client-side by Plotly.

**Data flow:** Precomputed CSVs/JSONs loaded once at startup → filtered on request → serialised to JSON → rendered by Plotly in the browser.

**The 1M-row geolocation file is never loaded by the app.**

### API Endpoints (11 total)
`/api/kpis`, `/api/filters`, `/api/monthly-trends`, `/api/categories`, `/api/states`, `/api/customers`, `/api/delivery`, `/api/sellers`, `/api/reviews`, `/api/insights`, `/api/ml`

### Filters
- State (multi-select) — updates delivery, sales, state charts
- Category (single select) — updates category charts
- Date range — updates time-series
- Insight type (risk/opportunity) — filters insight cards

---

## 10. Testing Results (Phase 9)

**159 automated tests, 159 passed, 0 failed.**

Test categories:
- File existence (47 files)
- Raw data integrity (row counts unchanged)
- KPI numerical accuracy
- Revenue reconciliation (zero double-count)
- RFM segment totals
- Delivery KPI cross-validation
- ML metric accuracy
- Insight structure completeness
- API endpoint health (11 endpoints)
- Page route health (5 routes)
- Filter functionality
- No hard-coded personal paths

---

## 11. Limitations

1. **Acquisition cost**: not present in the dataset — no CAC/LTV analysis possible
2. **Causality**: all relationships between delivery timing and review scores are observed associations; no causal analysis was performed
3. **ML test period**: the test set covers only May–Aug 2018 (lower-than-average late rates); model performance on high-late-rate periods may differ
4. **Category assignment**: for multi-item orders, delivery and review KPIs are assigned to the first item's category — a known approximation
5. **Cross-state customers**: 38 customers appear in >1 state due to the per-order `customer_id` model; documented, not hidden
6. **Time horizon**: the dataset ends August 2018; trends and patterns may have changed subsequently

---

## 12. Files Reference

| File | Purpose |
|---|---|
| `preprocessing.py` | Phase 2: data cleaning, joins, feature engineering |
| `analytics.py` | Phase 3: KPIs, aggregations, time-series |
| `customer_analytics.py` | Phase 4: RFM, repeat/one-time, geography |
| `delivery_satisfaction.py` | Phase 5: delivery deep-dive, review analysis |
| `ml_model.py` | Phase 6: late-delivery prediction model |
| `insight_engine.py` | Phase 7: structured business insights |
| `app.py` | Phase 8: Flask application and API |
| `test_pipeline.py` | Phase 9: 159-test end-to-end suite |
| `requirements.txt` | Python dependencies |
| `README.md` | Installation and usage guide |
| `data/insights_master.json` | All 18 structured business insights |
| `data/ml_model.pkl` | Serialised Logistic Regression pipeline |
| `data/kpis.json` | Top-line scalar KPIs |
