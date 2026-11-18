"""Tests unitaires : reconstruction des dates, absence de fuite temporelle, économie des décisions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import END_DATE, MarketingConfig, ModelConfig  # noqa: E402
from data.instacart_loader import reconstruct_dates  # noqa: E402
from marketing import economics as eco  # noqa: E402
from marketing.decision import decide  # noqa: E402
from prediction.features import make_snapshots, snapshot_features  # noqa: E402
from prediction.models import baseline_predict  # noqa: E402


# --------------------------------------------------------------------------- données
def test_reconstruct_dates_preserves_intervals():
    orders = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "order_number": [1, 2, 3, 1, 2],
                           "days_since_prior_order": [np.nan, 7, 10, np.nan, 3]})
    o = reconstruct_dates(orders)
    gaps = o.groupby("user_id")["order_date"].diff().dt.days.dropna().tolist()
    assert gaps == [7, 10, 3]
    assert (o["order_date"] <= pd.Timestamp(END_DATE)).all()


# --------------------------------------------------------------------------- features
def _toy_events():
    rows = []
    for k, d in enumerate(pd.date_range("2026-01-01", periods=20, freq="10D")):
        rows.append((1, 100, d, 1))
        if k % 2 == 0:
            rows.append((1, 200, d, 2))
    ev = pd.DataFrame(rows, columns=["customer_id", "product_id", "date", "quantity"])
    ev = ev.sort_values(["customer_id", "product_id", "date"]).reset_index(drop=True)
    ev["interval"] = ev.groupby(["customer_id", "product_id"])["date"].diff().dt.days
    orders = ev[["customer_id", "date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    orders["gap"] = orders["date"].diff().dt.days
    customers = pd.DataFrame({"customer_id": [1], "household_size": [2]})
    products = pd.DataFrame({"product_id": [100, 200], "unit_price": [2.0, 3.0], "category_code": [0, 1]})
    return ev, orders, customers, products


def test_no_future_information_in_features():
    ev, orders, cu, pr = _toy_events()
    cfg = ModelConfig(min_purchases=3)
    s = pd.Timestamp("2026-03-15")
    f = snapshot_features(s, ev, orders, cu, pr, cfg).set_index("product_id")
    assert (f["last_purchase"] <= s).all()
    assert f.loc[100, "n_purchases"] == (ev.query("product_id == 100")["date"] <= s).sum()
    assert f.loc[100, "mean_purchase_interval"] == pytest.approx(10)
    # cible = premier achat strictement postérieur au snapshot
    assert f.loc[100, "days_until_next_purchase"] == (pd.Timestamp("2026-03-22") - s).days


def test_temporal_split_targets_before_test():
    ev, *_ = _toy_events()
    cfg = ModelConfig(horizon_days=30, snapshot_step_days=10, min_history_days=20, n_test_snapshots=2)
    plan = make_snapshots(ev, cfg)
    horizon = pd.Timedelta(days=cfg.horizon_days)
    assert max(plan.train) + horizon <= min(plan.test)
    assert max(plan.test) + horizon <= ev["date"].max()


def test_baseline_is_median_minus_elapsed():
    df = pd.DataFrame({"median_purchase_interval": [10.0, 5.0], "days_since_last_purchase": [4, 9]})
    assert baseline_predict(df, "median").tolist() == [6.0, 1.0]


# --------------------------------------------------------------------------- marketing
def test_discount_is_costly_for_loyal_buyers():
    cfg = MarketingConfig()
    # client quasi certain d'acheter : la réduction est un pur effet d'aubaine
    assert eco.gross_profit_discount(0.95, 2.0, 6.0, 0.10, cfg) < 0
    assert eco.gross_profit_reminder(0.95, 2.0, cfg) >= 0


def test_decision_engine_three_scenarios():
    cfg = MarketingConfig()
    as_of = pd.Timestamp("2026-10-31")
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "product_id": [10, 10, 10],
        "as_of_date": [as_of] * 3,
        "predicted_date": [as_of + pd.Timedelta(days=3)] * 3,
        "purchase_proba": [0.60, 0.05, 0.99],
        "margin": [4.0, 6.0, 0.05],
        "revenue": [10.0, 12.0, 1.0],
        "preferred_hour": [18, 12, 9],
    })
    out = decide(df, cfg).set_index("customer_id")
    assert out.loc[1, "decision"] == "A_relance"
    assert out.loc[2, "decision"] == "B_promotion" and out.loc[2, "discount_rate"] > 0
    assert out.loc[3, "decision"] == "C_aucune" and out.loc[3, "expected_profit"] == 0
    assert out.loc[1, "campaign_date"] == "2026-11-02 17:00"
