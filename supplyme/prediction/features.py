"""
Construction des variables explicatives SANS FUITE TEMPORELLE.

Principe (« snapshots ») : on se place à une date de prédiction s.
- Les variables sont calculées uniquement avec les achats datés <= s.
- La cible est le délai entre s et le premier achat du même produit > s.
- Si aucun réachat n'a lieu dans `horizon_days`, l'observation est
  censurée : exclue de la régression, mais conservée (y_cls = 0) pour le
  classifieur « achat dans les 7 jours ».

Le découpage train/test est temporel : toutes les cibles d'entraînement
sont observées AVANT la première date de test (cf. `make_snapshots`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DB_PATH, ModelConfig
from database.db import query

log = logging.getLogger(__name__)

FEATURES = [
    "days_since_last_purchase",
    "mean_purchase_interval",
    "median_purchase_interval",
    "std_purchase_interval",
    "last_purchase_interval",
    "overdue_ratio",
    "n_purchases",
    "purchase_frequency",
    "average_quantity",
    "customer_n_orders",
    "customer_mean_order_gap",
    "customer_days_since_last_order",
    "household_size",
    "product_median_interval",
    "product_n_buyers",
    "unit_price",
    "category_code",
]


# ---------------------------------------------------------------------------
# Chargement depuis SQLite
# ---------------------------------------------------------------------------
def load_events(db_path=DB_PATH) -> pd.DataFrame:
    """Un événement = un achat d'un produit par un client un jour donné."""
    ev = query("""
        SELECT o.customer_id, oi.product_id, o.order_date AS date, SUM(oi.quantity) AS quantity
        FROM order_items oi JOIN orders o ON o.order_id = oi.order_id
        GROUP BY o.customer_id, oi.product_id, o.order_date
    """, db_path=db_path, parse_dates=["date"])
    ev = ev.sort_values(["customer_id", "product_id", "date"]).reset_index(drop=True)
    ev["interval"] = ev.groupby(["customer_id", "product_id"])["date"].diff().dt.days
    return ev


def load_orders(db_path=DB_PATH) -> pd.DataFrame:
    o = query("SELECT customer_id, order_date AS date FROM orders", db_path=db_path, parse_dates=["date"])
    o = o.drop_duplicates().sort_values(["customer_id", "date"]).reset_index(drop=True)
    o["gap"] = o.groupby("customer_id")["date"].diff().dt.days
    return o


def load_static(db_path=DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = query("SELECT customer_id, household_size FROM customers", db_path=db_path)
    products = query("SELECT product_id, unit_price, category FROM products", db_path=db_path)
    products["category_code"] = products["category"].astype("category").cat.codes
    return customers, products


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
@dataclass
class SnapshotPlan:
    train: list[pd.Timestamp]
    test: list[pd.Timestamp]
    inference: pd.Timestamp


def make_snapshots(events: pd.DataFrame, cfg: ModelConfig) -> SnapshotPlan:
    start, end = events["date"].min(), events["date"].max()
    step = pd.Timedelta(days=cfg.snapshot_step_days)
    horizon = pd.Timedelta(days=cfg.horizon_days)
    last_test = end - horizon                       # la cible doit pouvoir être observée
    test = [last_test - k * step for k in range(cfg.n_test_snapshots)][::-1]
    train, s = [], test[0] - horizon                # cibles d'entraînement < 1re date de test
    while s >= start + pd.Timedelta(days=cfg.min_history_days) and len(train) < cfg.max_train_snapshots:
        train.append(s)
        s -= step
    if not train:
        raise ValueError("Historique trop court pour constituer un échantillon d'entraînement.")
    return SnapshotPlan(train=train[::-1], test=test, inference=end)


def snapshot_features(s: pd.Timestamp, events: pd.DataFrame, orders: pd.DataFrame,
                      customers: pd.DataFrame, products: pd.DataFrame, cfg: ModelConfig,
                      with_target: bool = True) -> pd.DataFrame:
    hist = events[events["date"] <= s]
    keys = ["customer_id", "product_id"]
    g = hist.groupby(keys)
    f = g.agg(n_purchases=("date", "size"),
              first_purchase=("date", "min"),
              last_purchase=("date", "max"),
              mean_purchase_interval=("interval", "mean"),
              median_purchase_interval=("interval", "median"),
              std_purchase_interval=("interval", "std"),
              average_quantity=("quantity", "mean")).reset_index()
    f = f[f["n_purchases"] >= cfg.min_purchases]
    if f.empty:
        return f
    last_int = hist.groupby(keys)["interval"].last().rename("last_purchase_interval")
    f = f.merge(last_int, left_on=keys, right_index=True, how="left")

    f["days_since_last_purchase"] = (s - f["last_purchase"]).dt.days
    span = (f["last_purchase"] - f["first_purchase"]).dt.days.clip(lower=1)
    f["purchase_frequency"] = (f["n_purchases"] - 1) / span * 30      # achats par 30 jours
    f["overdue_ratio"] = f["days_since_last_purchase"] / f["mean_purchase_interval"].clip(lower=1)
    f["std_purchase_interval"] = f["std_purchase_interval"].fillna(0)

    # niveau client (commandes passées uniquement)
    oh = orders[orders["date"] <= s].groupby("customer_id").agg(
        customer_n_orders=("date", "size"), customer_mean_order_gap=("gap", "mean"),
        customer_last_order=("date", "max")).reset_index()
    oh["customer_days_since_last_order"] = (s - oh.pop("customer_last_order")).dt.days
    f = f.merge(oh, on="customer_id", how="left")

    # niveau produit : encodage adapté de product_id (statistiques passées, tous clients)
    ph = hist.groupby("product_id").agg(product_median_interval=("interval", "median"),
                                        product_n_buyers=("customer_id", "nunique")).reset_index()
    f = f.merge(ph, on="product_id", how="left")
    f = f.merge(customers, on="customer_id", how="left").merge(
        products[["product_id", "unit_price", "category_code"]], on="product_id", how="left")
    f["snapshot"] = s

    if with_target:
        fut = events[events["date"] > s].groupby(keys)["date"].min().rename("next_purchase")
        f = f.merge(fut, left_on=keys, right_index=True, how="left")
        days = (f["next_purchase"] - s).dt.days
        f["days_until_next_purchase"] = days
        f["censored"] = days.isna() | (days > cfg.horizon_days)
        f["buy_in_window"] = (days <= cfg.purchase_window_days).fillna(False).astype(int)
    return f.drop(columns=["first_purchase"])


def build_dataset(snapshots: list[pd.Timestamp], events, orders, customers, products,
                  cfg: ModelConfig) -> pd.DataFrame:
    frames = [snapshot_features(s, events, orders, customers, products, cfg) for s in snapshots]
    df = pd.concat([x for x in frames if not x.empty], ignore_index=True)
    df[FEATURES] = df[FEATURES].astype(float).fillna(0.0)   # imputation fixe : aucune info du test
    log.info("  %d snapshots -> %s observations (%.1f %% censurées)",
             len(snapshots), f"{len(df):,}", 100 * df["censored"].mean())
    return df
