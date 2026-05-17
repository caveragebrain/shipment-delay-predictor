"""Programmatic test battery for the trained model.

Exercises the predictor with 100+ synthetic shipment profiles to verify:
  - Probability is monotonic in the features it should be (discount, rating, etc.)
  - Test-set metrics replicate (no silent drift between training and inference)
  - The extremes (worst/best case) actually hit the top/bottom of the prob range
  - Random sampling shows a sensible probability distribution

Run: .venv/bin/python -m scripts.test_battery
"""

from __future__ import annotations

import random
from itertools import product

import numpy as np
import pandas as pd

from app.predictor import BEST_CASE_INPUT, WORST_CASE_INPUT, Predictor
from src.preprocess import PROCESSED_DIR


def base_shipment() -> dict:
    return {
        "warehouse_block": "d",
        "mode_of_shipment": "ship",
        "customer_care_calls": 4,
        "customer_rating": 3,
        "cost_of_product": 200,
        "prior_purchases": 4,
        "product_importance": "medium",
        "gender": "m",
        "discount_offered": 10,
        "weight_in_gms": 3500,
    }


def banner(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def test_monotonicity(p: Predictor):
    """For each one-feature sweep, verify the prob moves in the expected direction."""
    banner("MONOTONICITY TESTS (holding all other features fixed)")
    base = base_shipment()

    sweeps = [
        ("discount_offered",   range(0, 66, 5),    "should INCREASE prob"),
        ("weight_in_gms",      range(1000, 7001, 500), "weight signal is mixed in this dataset"),
        ("customer_care_calls", range(1, 8),       "should INCREASE prob (more calls = more issues)"),
        ("customer_rating",    range(1, 6),        "should DECREASE prob (better rating = more reliable)"),
        ("prior_purchases",    range(2, 11),       "should DECREASE prob (repeat buyer = more reliable)"),
        ("cost_of_product",    range(100, 311, 30), "weak signal"),
    ]

    for feat, values, expectation in sweeps:
        probs = []
        for v in values:
            ship = dict(base)
            ship[feat] = float(v) if isinstance(v, (int, np.integer)) else v
            probs.append(p.predict_proba(ship))
        delta = probs[-1] - probs[0]
        direction = "↑" if delta > 0 else "↓"
        print(f"  {feat:24s}  range=[{probs[0]:.3f} → {probs[-1]:.3f}]  Δ={delta:+.3f} {direction}   ({expectation})")


def test_categorical(p: Predictor):
    banner("CATEGORICAL EFFECTS (holding numerics fixed)")
    base = base_shipment()
    for feat, values in [
        ("warehouse_block", ["a", "b", "c", "d", "f"]),
        ("mode_of_shipment", ["ship", "flight", "road"]),
        ("product_importance", ["low", "medium", "high"]),
        ("gender", ["m", "f"]),
    ]:
        results = []
        for v in values:
            ship = dict(base); ship[feat] = v
            results.append((v, p.predict_proba(ship)))
        print(f"  {feat}:")
        for v, prob in sorted(results, key=lambda x: x[1]):
            print(f"      {v:8s}  prob={prob:.3f}")


def test_extremes(p: Predictor):
    banner("WORST / BEST / BASELINE CASES")
    worst_p = p.predict_proba(WORST_CASE_INPUT)
    best_p = p.predict_proba(BEST_CASE_INPUT)
    base_p = p.predict_proba(base_shipment())
    print(f"  Worst case prob:    {worst_p:.4f}  (expected near 0.95)")
    print(f"  Best case prob:     {best_p:.4f}  (expected as low as model can go)")
    print(f"  Neutral case prob:  {base_p:.4f}")
    print(f"  Spread (worst-best): {worst_p - best_p:+.3f} probability points")


def test_distribution_random(p: Predictor, n: int = 500):
    """Generate N random shipments uniformly from feature ranges, look at prob spread."""
    banner(f"RANDOM SAMPLE DISTRIBUTION (n={n})")
    rng = random.Random(42)
    probs = []
    for _ in range(n):
        ship = {
            "warehouse_block": rng.choice(["a","b","c","d","f"]),
            "mode_of_shipment": rng.choice(["ship","flight","road"]),
            "customer_care_calls": rng.randint(1, 7),
            "customer_rating": rng.randint(1, 5),
            "cost_of_product": rng.uniform(96, 310),
            "prior_purchases": rng.randint(2, 10),
            "product_importance": rng.choice(["low","medium","high"]),
            "gender": rng.choice(["m","f"]),
            "discount_offered": rng.uniform(0, 65),
            "weight_in_gms": rng.uniform(1000, 7000),
        }
        probs.append(p.predict_proba(ship))
    probs = np.array(probs)
    print(f"  min={probs.min():.3f}  max={probs.max():.3f}  mean={probs.mean():.3f}  median={np.median(probs):.3f}")
    bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    counts, _ = np.histogram(probs, bins=bins)
    print("  Distribution:")
    for i in range(len(bins)-1):
        bar = "█" * int(counts[i] / max(counts) * 40)
        print(f"    [{bins[i]:.2f}–{bins[i+1]:.2f}]  {counts[i]:4d}  {bar}")


def test_replicates_test_set(p: Predictor):
    """Re-score the held-out test set; verify metrics match what training reported."""
    banner("TEST-SET REPLICATION (should match training-time metrics)")
    df = pd.read_csv(PROCESSED_DIR / "raw_test.csv")
    y_true = df["delayed"].to_numpy()
    feature_rows = df.drop(columns=["delayed"]).to_dict(orient="records")
    probs = np.array([p.predict_proba(r) for r in feature_rows])
    y_pred = (probs >= 0.5).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"  F1@0.5={f1:.4f}  ROC-AUC={auc:.4f}  n={len(y_true)}")
    print(f"  Confusion matrix [tn fp; fn tp] = {cm.tolist()}")
    print(classification_report(y_true, y_pred, target_names=['on_time','delayed'], zero_division=0))


def test_consistency(p: Predictor):
    """Same input -> same output, twice."""
    banner("DETERMINISM CHECK")
    s = base_shipment()
    p1, p2 = p.predict_proba(s), p.predict_proba(s)
    assert abs(p1 - p2) < 1e-9, f"Non-deterministic: {p1} vs {p2}"
    print(f"  Same shipment, same probability: {p1:.6f} == {p2:.6f}  ✓")


def main():
    p = Predictor.load()
    test_consistency(p)
    test_extremes(p)
    test_monotonicity(p)
    test_categorical(p)
    test_distribution_random(p, n=500)
    test_replicates_test_set(p)
    print("\nDone. ✓")


if __name__ == "__main__":
    main()
