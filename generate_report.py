"""
generate_report.py
Generate KarthikTeja_ProjectReport.docx from PROJECT_REPORT.md content.
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_table(doc, headers, rows, col_widths=None):
    n_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = 'Table Grid'
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(10)
        set_cell_bg(cell, 'E8EDF4')
    # Data rows
    for ri, row_data in enumerate(rows):
        r = table.rows[ri + 1]
        for ci, val in enumerate(row_data):
            r.cells[ci].text = str(val)
            for run in r.cells[ci].paragraphs[0].runs:
                run.font.size = Pt(10)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    if level == 1:
        for run in p.runs:
            run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def add_body(doc, text, italic=False, bold=False, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.italic = italic
    run.bold = bold
    if color:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(6)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(text, style='List Bullet')
    for run in p.runs:
        run.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    return p

def build_report():
    doc = Document()

    # ── Page margins ──────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.1)
        section.right_margin  = Inches(1.1)

    # ── Title block ───────────────────────────────────────────
    t = doc.add_heading('AI-Powered E-Commerce Business Intelligence & Decision Support System', 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run('Project Report — IBM Data Analytics Internship')
    r.font.size = Pt(13); r.bold = True
    sub.paragraph_format.space_after = Pt(4)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta.add_run('Author: KarthikTeja   |   Dataset: Brazilian E-Commerce Public Dataset by Olist   |   Period: Sep 2016 – Aug 2018')
    mr.font.size = Pt(10)
    mr.font.color.rgb = RGBColor(0x55, 0x60, 0x6A)
    meta.paragraph_format.space_after = Pt(16)

    doc.add_page_break()

    # ── 1. Project Overview ───────────────────────────────────
    add_heading(doc, '1. Project Overview')
    add_body(doc, ('This project converts raw Olist e-commerce transaction data into a production-quality '
                   'Business Intelligence system. The pipeline runs from raw CSV ingestion through data '
                   'cleaning, feature engineering, statistical analysis, machine learning, and a fully '
                   'interactive web dashboard.'))
    add_body(doc, ('The system answers seven business-critical questions: (1) What is happening? '
                   '(2) Why is it happening? (3) What are the important trends? (4) What are the major '
                   'drivers? (5) What risks exist? (6) What opportunities exist? '
                   '(7) What actions should the business take?'))
    add_body(doc, 'Dataset Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce', bold=True)

    # ── 2. Dataset Structure ──────────────────────────────────
    add_heading(doc, '2. Dataset Structure')
    add_body(doc, ('Nine source CSV files covering the complete order lifecycle across 100,000 orders, '
                   '96,000 customers, and 3,095 sellers in Brazil.'))
    add_table(doc,
        headers=['Table', 'Rows', 'Key Columns'],
        rows=[
            ['Customers',            '99,441',    'customer_id, customer_unique_id, state'],
            ['Orders',               '99,441',    'order_id, customer_id, status, 4 timestamps'],
            ['Order Items',          '112,650',   'order_id, product_id, seller_id, price, freight'],
            ['Payments',             '103,886',   'order_id, payment_type, installments, value'],
            ['Reviews',              '99,224',    'order_id, review_score (1-5), comment'],
            ['Products',             '32,951',    'product_id, category (PT/EN), dimensions, weight'],
            ['Sellers',              '3,095',     'seller_id, zip, city, state'],
            ['Geolocation',          '1,000,163', 'zip, lat, lng, city, state'],
            ['Category Translation', '71',        'Portuguese category -> English'],
        ],
        col_widths=[1.8, 1.0, 3.8]
    )
    doc.add_paragraph()
    add_body(doc, ('Key relationships: Orders join to Customers (1:1), to Items (1:N), to Payments (1:N), '
                   'to Reviews (1:1 after dedup). Items join to Products and Sellers. '
                   'Zero orphan keys across all six foreign key relationships.'))

    # ── 3. Data Preprocessing ─────────────────────────────────
    add_heading(doc, '3. Data Preprocessing (Phase 2)')
    add_heading(doc, 'Key Cleaning Steps', level=2)
    for b in [
        'Date type conversion for all 4 order timestamps',
        'Zip codes zero-padded to 5 characters',
        '1,000,163 geolocation rows deduplicated to 19,015 unique zips (median lat/lng per zip)',
        'Payments aggregated to one row per order (sum value, dominant payment type)',
        'English category translation joined; 3 known typos corrected in English column only',
        '8 delivered orders with missing actual delivery dates excluded from timing KPIs',
    ]:
        add_bullet(doc, b)

    add_heading(doc, 'Review Aggregation Strategy', level=2)
    add_body(doc, ('547 orders had 2+ review events. 202 showed score changes between events '
                   '(first score mean = 3.62, last score mean = 3.16 — customers who re-reviewed '
                   'tended to revise downward). Decision: keep the LAST review per order '
                   '(most recent settled sentiment). Documented and justified.'))

    add_heading(doc, 'Output Tables', level=2)
    add_table(doc,
        headers=['Table', 'Rows', 'Grain'],
        rows=[
            ['orders_master.csv', '99,441',  'One row per order'],
            ['orders_full.csv',   '113,425', 'One row per order-item (+ orders with no items)'],
            ['customer_rfm.csv',  '93,358',  'One row per customer_unique_id'],
        ],
        col_widths=[2.1, 1.0, 3.5]
    )

    # ── 4. Business KPIs ──────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, '4. Business KPIs (Phase 3)')
    add_body(doc, ("All KPIs computed from delivered orders only. Revenue is never double-counted: "
                   "order-level KPIs use orders_master; category-level revenue uses item-level orders_full."))
    add_table(doc,
        headers=['KPI', 'Value', 'Notes'],
        rows=[
            ['Total Product Sales',     'R$ 13,221,498', 'SUM(price), delivered orders'],
            ['Total Freight Value',      'R$ 2,198,276',  'SUM(freight_value), delivered orders'],
            ['Total Order Value',        'R$ 15,419,774', 'Sales + Freight'],
            ['Total Orders',             '99,441',        'All statuses'],
            ['Total Delivered Orders',   '96,478',        '97.0% of all orders'],
            ['Unique Customers',         '96,096',        'By customer_unique_id'],
            ['Average Order Value',      'R$ 159.83',     'Order value / delivered orders'],
            ['Average Review Score',     '4.156 / 5',     '95,832 scored orders'],
            ['On-Time Delivery',         '93.23%',        'Actual <= estimated delivery date'],
            ['Late Delivery',            '6.77%',         'Actual > estimated delivery date'],
            ['Average Delivery Days',    '12.09 days',    'From purchase to delivery'],
            ['Repeat Customer Rate',     '3.00%',         'Customers with >= 2 delivered orders'],
        ],
        col_widths=[2.2, 1.5, 2.9]
    )
    doc.add_paragraph()
    add_body(doc, 'Cross-validation: Product sales sum independently verified at item level — diff = R$0.00.', italic=True)

    # ── 5. Customer Analysis — RFM ────────────────────────────
    add_heading(doc, '5. Customer Analysis — RFM Segmentation (Phase 4)')
    add_body(doc, ('RFM scored in quintiles (1-5). Reference date: 2018-08-30. '
                   'Recency = days since last purchase; Frequency = distinct delivered orders; '
                   'Monetary = total order_value.'))
    add_table(doc,
        headers=['Segment', 'Customers', '% Total', 'Revenue (R$)', '% Rev', 'Avg Spend'],
        rows=[
            ['Needs Attention',    '23,607', '25.3%', '3,648,478',  '23.7%', 'R$ 155'],
            ['At Risk',            '13,162', '14.1%', '3,205,152',  '20.8%', 'R$ 244'],
            ['Loyal Customers',    '14,210', '15.2%', '2,942,647',  '19.1%', 'R$ 207'],
            ['New Customers',      '14,984', '16.1%', '2,448,694',  '15.9%', 'R$ 163'],
            ['Champions',           '6,497',  '7.0%', '2,027,030',  '13.2%', 'R$ 312'],
            ['Potential Loyalists', '14,588', '15.6%',  '795,736',   '5.2%', 'R$ 55'],
            ['Lost / Inactive',     '6,310',  '6.8%',  '352,035',   '2.3%', 'R$ 56'],
            ['TOTAL',              '93,358', '100%',  '15,419,772', '100%',  '—'],
        ],
        col_widths=[1.6, 1.0, 0.8, 1.4, 0.7, 1.0]
    )
    doc.add_paragraph()
    add_body(doc, ('Key finding: The At Risk segment contains the highest revenue (R$ 3.2M, 20.8%) — '
                   'the highest-value win-back target. 97% of customers placed exactly one delivered order.'))

    # ── 6. Delivery & Satisfaction ────────────────────────────
    add_heading(doc, '6. Delivery & Satisfaction Analysis (Phase 5)')
    add_heading(doc, 'Overall Performance', level=2)
    add_body(doc, ('78.9% of deliveries arrive more than 7 days before the estimated date. '
                   'Average order arrives 11.88 days before the estimate (median 12 days early). '
                   'Estimated delivery dates are systematically conservative.'))

    add_heading(doc, 'Regional Performance — High-Impact States (>= 200 orders)', level=2)
    add_table(doc,
        headers=['State', 'Late %', 'vs National', 'Avg Days', 'Avg Review'],
        rows=[
            ['AL', '21.4%', '+14.6pp', '24.0', '3.848'],
            ['MA', '17.4%', '+10.7pp', '21.1', '3.833'],
            ['SE', '15.2%', '+8.4pp',  '21.0', '3.907'],
            ['PI', '13.9%', '+7.1pp',  '19.0', '3.994'],
            ['CE', '13.8%', '+7.0pp',  '20.8', '3.944'],
            ['RJ', '12.1%', '+5.3pp',  '14.8', '3.965'],
        ],
        col_widths=[0.7, 0.9, 1.2, 1.1, 1.2]
    )

    doc.add_paragraph()
    add_heading(doc, 'Delay vs Review Score — Observed Association (not causal)', level=2)
    add_table(doc,
        headers=['Delay Bucket', 'Avg Review', '1-star %', '5-star %', 'Orders'],
        rows=[
            ['Early >7d',  '4.313', '6.5%',  '63.4%', '75,744'],
            ['Early 3-7d', '4.191', '7.0%',  '57.2%',  '9,429'],
            ['Early 0-3d', '4.108', '8.1%',  '54.0%',  '4,270'],
            ['Late 1-3d',  '3.291', '25.1%', '33.2%',  '1,852'],
            ['Late 4-7d',  '2.105', '58.6%', '14.1%',  '1,748'],
            ['Late 8-14d', '1.671', '70.6%', '6.8%',   '1,446'],
            ['Late >14d',  '1.723', '69.1%', '7.2%',   '1,335'],
        ],
        col_widths=[1.3, 1.2, 1.0, 1.0, 1.0]
    )

    # ── 7. Machine Learning ───────────────────────────────────
    add_heading(doc, '7. Machine Learning — Late Delivery Prediction (Phase 6)')
    add_body(doc, ('Prediction scenario: Predict at order approval whether the order will arrive after '
                   'its estimated delivery date, using only information available before any delivery event.'))
    add_body(doc, ('Features (13, all pre-delivery): days_to_estimated, total_freight_value, avg_weight_g, '
                   'avg_volume_cm3, order_item_count, unique_sellers, payment_installments, '
                   'approval_lag_hours, purchase_month, purchase_dayofweek, customer_state, '
                   'seller_state, payment_type'))
    add_body(doc, ('Time-based split: Training 77,176 orders (Sep 2016 – May 2018, late rate 7.59%), '
                   'Test 19,294 orders (May–Aug 2018, late rate 3.49%).'))
    add_table(doc,
        headers=['Model', 'ROC-AUC', 'Recall', 'Precision', 'F1', 'Accuracy'],
        rows=[
            ['Logistic Regression', '0.7066 (recommended)', '76.41%', '6.08%', '0.1126', '57.93%'],
            ['Random Forest',       '0.5184',               '20.92%', '4.77%', '0.0777', '82.64%'],
        ],
        col_widths=[1.6, 1.8, 0.8, 1.0, 0.8, 0.9]
    )
    doc.add_paragraph()
    add_body(doc, ('Logistic Regression recommended (AUC 0.7066 vs 0.5184 for RF). High recall (76.41%) '
                   'identifies most at-risk orders early. The model serves as an early-warning mechanism, '
                   'not a delivery guarantee.'))
    add_body(doc, ('Limitation: Test window (May-Aug 2018) had 3.49% late rate vs 7.59% in training '
                   '(temporal distribution shift). Metrics reflect that specific period only.'), italic=True)

    # ── 8. Business Insight Engine ────────────────────────────
    add_heading(doc, '8. Business Insight Engine (Phase 7)')
    add_body(doc, ('18 structured insights across 5 categories, each following: '
                   'FACT -> INSIGHT -> RISK/OPPORTUNITY -> RECOMMENDED ACTION. '
                   'All insights are traceable to calculated data.'))
    add_table(doc,
        headers=['Category', 'Total Insights', 'Risks', 'Opportunities'],
        rows=[
            ['Customer',     '5', '2', '3'],
            ['Sales',        '3', '2', '1'],
            ['Delivery',     '5', '4', '1'],
            ['Satisfaction', '3', '3', '0'],
            ['ML',           '2', '0', '2'],
            ['TOTAL',        '18', '10', '8'],
        ],
        col_widths=[1.8, 1.5, 1.2, 1.5]
    )
    doc.add_paragraph()
    for b in [
        'At Risk RFM segment: R$3.2M revenue — highest of any segment (win-back priority)',
        'SP + RJ + MG = 62.5% of total revenue (geographic concentration risk)',
        '78.9% of deliveries arrive >7 days before estimated date (conservative estimates)',
        'AL 21.4% / MA 17.4% late rates — 2-3x the national average of 6.77%',
        'Late 4-7d orders: avg 2.11 stars vs 4.31 for Early >7d (observed association)',
        '76 qualified sellers (>=50 orders) have late rates >= 1.5x national average',
    ]:
        add_bullet(doc, b)

    # ── 9. Dashboard Architecture ─────────────────────────────
    add_heading(doc, '9. Dashboard Architecture (Phase 8)')
    add_body(doc, ('Technology: Flask (Python 3.11), Plotly (JS), HTML5/CSS3/JavaScript. '
                   'API-first pattern: all data served via /api/* JSON endpoints, rendered '
                   'client-side by Plotly. Precomputed analytical files loaded once at startup.'))
    add_heading(doc, 'Dashboard Pages', level=2)
    add_table(doc,
        headers=['Page', 'URL', 'Key Content'],
        rows=[
            ['Executive Overview',      '/',          '8 KPI cards, monthly trends, category/state sales, 18 insight cards'],
            ['Sales & Product Analysis','/sales',     'Revenue/orders by category, AOV, review scores, monthly trend, sellers'],
            ['Customer Analytics',      '/customers', 'RFM pie/bars, spending distribution, repeat vs one-time, geography'],
            ['Delivery & Risk',         '/delivery',  'Delivery KPIs, trends, delay dist., state/category/seller analysis, ML section'],
        ],
        col_widths=[1.7, 1.1, 3.8]
    )

    # ── 10. Testing ───────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, '10. Testing Results (Phase 9)')
    add_body(doc, '159 automated tests, 159 passed, 0 failed.  Run with: python test_pipeline.py', bold=True)
    for b in [
        'File existence (47 files checked)',
        'Raw data integrity — row counts of original CSVs unchanged',
        'KPI numerical accuracy (diff < 0.02)',
        'Revenue reconciliation — zero double-count at item and order level',
        'RFM segment customer and revenue totals reconcile',
        'Delivery KPI cross-validation against Phase 3',
        'ML metric accuracy (AUC, recall, precision, F1)',
        'Insight structure completeness (18 insights, required fields)',
        'All 11 API endpoints return HTTP 200',
        'All 5 page routes return HTTP 200',
        'Filter functionality (state, category, insight type)',
        'No hard-coded personal file paths in any source file',
    ]:
        add_bullet(doc, b)

    # ── 11. Technologies ──────────────────────────────────────
    add_heading(doc, '11. Technologies Used')
    add_table(doc,
        headers=['Layer', 'Technology'],
        rows=[
            ['Language',        'Python 3.11'],
            ['Data Processing', 'Pandas 2.1.2, NumPy 1.26.1'],
            ['Machine Learning','Scikit-learn 1.3.2'],
            ['Visualisation',   'Plotly 7.1.0'],
            ['Web Framework',   'Flask 3.0.0'],
            ['Frontend',        'HTML5, CSS3, JavaScript ES6'],
        ],
        col_widths=[2.0, 4.0]
    )

    # ── 12. Limitations ───────────────────────────────────────
    add_heading(doc, '12. Limitations')
    for b in [
        'Acquisition cost not in dataset — no CAC/LTV analysis possible',
        'All delivery-review relationships are observed associations; no causal analysis performed',
        'ML test window (May-Aug 2018) had lower late rates than training; results period-specific',
        'Multi-item orders: delivery/review KPIs assigned to first item\'s category (documented approximation)',
        'Dataset ends August 2018; trends may have changed since',
    ]:
        add_bullet(doc, b)

    # ── 13. How to Run ────────────────────────────────────────
    add_heading(doc, '13. How to Run')
    for b in [
        'pip install -r requirements.txt',
        'python preprocessing.py',
        'python analytics.py',
        'python customer_analytics.py',
        'python delivery_satisfaction.py',
        'python ml_model.py',
        'python insight_engine.py',
        'python test_pipeline.py   (optional — expect 159/159 passed)',
        'python app.py             then open http://localhost:5000',
    ]:
        add_bullet(doc, b)
    doc.add_paragraph()
    add_body(doc, 'The original Olist CSV files are NEVER modified. All outputs are written to data/.', italic=True)

    # ── Footer note ───────────────────────────────────────────
    doc.add_paragraph()
    add_body(doc, 'Dataset: Brazilian E-Commerce Public Dataset by Olist', bold=True)
    add_body(doc, 'URL: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce')
    add_body(doc, 'License: CC BY-NC-SA 4.0 — https://creativecommons.org/licenses/by-nc-sa/4.0/')

    out = os.path.join(os.path.dirname(__file__), 'KarthikTeja_ProjectReport.docx')
    doc.save(out)
    print(f'Saved: {out}')


if __name__ == '__main__':
    build_report()
