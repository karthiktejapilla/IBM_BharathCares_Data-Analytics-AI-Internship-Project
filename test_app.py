"""test_app.py — Phase 8 API validation"""
import json, sys
sys.path.insert(0, ".")
import app

def run():
    with app.app.test_client() as c:
        endpoints = [
            "/api/filters", "/api/kpis", "/api/monthly-trends",
            "/api/categories", "/api/states", "/api/customers",
            "/api/delivery", "/api/sellers", "/api/reviews",
            "/api/insights", "/api/ml",
        ]
        print("=== API ENDPOINT TESTS ===")
        all_ok = True
        for ep in endpoints:
            r = c.get(ep)
            ok = r.status_code == 200
            if not ok:
                all_ok = False
            data = json.loads(r.data)
            size = len(str(data))
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {ep:<35} status={r.status_code}  payload={size:,} chars")

        print()
        print("=== FILTER TESTS ===")
        filter_tests = [
            ("/api/delivery",  "states=SP,RJ"),
            ("/api/categories","top=5"),
            ("/api/sellers",   "top=10&sort=late_pct&asc=false"),
            ("/api/insights",  "category=delivery"),
            ("/api/insights",  "type=risk"),
        ]
        for ep, qs in filter_tests:
            r = c.get(f"{ep}?{qs}")
            ok = r.status_code == 200
            if not ok:
                all_ok = False
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {ep}?{qs}")

        print()
        print("=== KPI VALIDATION ===")
        kpis = json.loads(c.get("/api/kpis").data)
        checks = [
            ("total_product_sales",    13221498.11, kpis["total_product_sales"]),
            ("total_order_value",      15419773.75, kpis["total_order_value"]),
            ("total_orders",           99441,       kpis["total_orders"]),
            ("total_delivered_orders", 96478,       kpis["total_delivered_orders"]),
            ("total_customers",        96096,       kpis["total_customers"]),
            ("avg_order_value",        159.83,      kpis["avg_order_value"]),
            ("avg_review_score",       4.1559,      round(kpis["avg_review_score"],4)),
            ("on_time_delivery_pct",   93.23,       kpis["on_time_delivery_pct"]),
        ]
        for name, expected, actual in checks:
            diff = abs(float(actual) - float(expected))
            ok = diff < 0.02
            if not ok:
                all_ok = False
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {name:<30} expected={expected}  actual={actual}  diff={diff:.4f}")

        print()
        print("=== ML VALIDATION ===")
        ml = json.loads(c.get("/api/ml").data)
        lr = ml["metrics"]["logistic_regression"]
        ml_checks = [
            ("LR ROC-AUC",  0.7066, lr["roc_auc"]),
            ("LR Recall",   0.7641, lr["recall"]),
            ("LR Precision",0.0608, lr["precision"]),
            ("LR F1",       0.1126, lr["f1"]),
        ]
        for name, expected, actual in ml_checks:
            diff = abs(float(actual) - float(expected))
            ok = diff < 0.001
            if not ok:
                all_ok = False
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {name:<20} expected={expected}  actual={actual}")

        print()
        print("=== INSIGHT VALIDATION ===")
        ins = json.loads(c.get("/api/insights").data)
        risks = [i for i in ins["insights"] if i["type"] == "risk"]
        opps  = [i for i in ins["insights"] if i["type"] == "opportunity"]
        checks_i = [
            ("Total insights", 18, ins["total"]),
            ("Risks",          10, len(risks)),
            ("Opportunities",   8, len(opps)),
        ]
        for name, expected, actual in checks_i:
            ok = actual == expected
            if not ok:
                all_ok = False
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {name:<20} expected={expected}  actual={actual}")

        print()
        print("=== PAGE ROUTES ===")
        for route in ["/", "/executive", "/sales", "/customers", "/delivery"]:
            r = c.get(route)
            ok = r.status_code == 200 and b"Olist BI Dashboard" in r.data
            if not ok:
                all_ok = False
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {route}")

        print()
        overall = "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"
        print(f"=== {overall} ===")
        return all_ok

if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
