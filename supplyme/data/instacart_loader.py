"""
Chargement et préparation du dataset Instacart Market Basket Analysis.

Points méthodologiques gérés ici :
- Instacart ne fournit PAS de dates : seulement `order_number` et
  `days_since_prior_order` (plafonné à 30 jours). On reconstruit un
  calendrier par client (somme cumulée des écarts), ancré sur END_DATE.
  Les intervalles entre achats sont donc réels ; les dates absolues sont
  une convention.
- Les commandes `test` n'ont pas de produits associés : elles sont exclues.
- Les fichiers produits sont volumineux (~32 M lignes) : lecture par blocs
  en ne gardant que les commandes des clients échantillonnés.
- Quantités, prix, coûts et taille des foyers sont absents : ils sont
  simulés (cf. data/synthetic.py) et doivent être présentés comme tels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import END_DATE, INSTACART_FILES, RANDOM_SEED, DataConfig
from data.synthetic import enrich_customers, enrich_products, simulate_quantities

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    source: str
    quality: dict = field(default_factory=dict)


def instacart_available(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in INSTACART_FILES)


def _read_items(path: Path, order_ids: set[int], chunksize: int = 2_000_000) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in ("order_id", "product_id", "add_to_cart_order", "quantity") if c in header]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype={"order_id": "int32", "product_id": "int32"}):
        parts.append(chunk[chunk["order_id"].isin(order_ids)])
    return pd.concat(parts, ignore_index=True)


def reconstruct_dates(orders: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Transforme la séquence de commandes de chaque client en dates calendaires."""
    rng = np.random.default_rng(seed)
    o = orders.sort_values(["user_id", "order_number"]).copy()
    gap = o["days_since_prior_order"].fillna(0)
    o["day_index"] = gap.groupby(o["user_id"]).cumsum()
    last = o.groupby("user_id")["day_index"].transform("max")
    users = o["user_id"].unique()
    # décalage aléatoire de 0 à 6 jours : tous les clients ne commandent pas le dernier jour
    offset = pd.Series(rng.integers(0, 7, len(users)), index=users)
    days_before_end = (last - o["day_index"]) + o["user_id"].map(offset)
    o["order_date"] = pd.Timestamp(END_DATE) - pd.to_timedelta(days_before_end, unit="D")
    return o


def load_instacart(raw_dir: Path, cfg: DataConfig, source: str = "instacart",
                   seed: int = RANDOM_SEED) -> Dataset:
    log.info("Lecture des fichiers Instacart depuis %s", raw_dir)
    orders = pd.read_csv(raw_dir / "orders.csv")
    orders = orders[orders["eval_set"] != "test"]

    n_orders = orders.groupby("user_id").size()
    eligible = n_orders[n_orders >= cfg.min_orders_per_customer].index.to_numpy()
    rng = np.random.default_rng(seed)
    n = min(cfg.n_customers, len(eligible))
    sampled = np.sort(rng.choice(eligible, size=n, replace=False))
    orders = orders[orders["user_id"].isin(sampled)]
    log.info("Clients éligibles : %s — échantillon : %s", f"{len(eligible):,}", f"{n:,}")

    orders = reconstruct_dates(orders, seed)
    capped_share = float((orders["days_since_prior_order"] == 30).mean())

    order_ids = set(orders["order_id"].astype(int))
    items = pd.concat([_read_items(raw_dir / "order_products__prior.csv", order_ids),
                       _read_items(raw_dir / "order_products__train.csv", order_ids)],
                      ignore_index=True)
    quantity_simulated = "quantity" not in items.columns
    if quantity_simulated:
        items["quantity"] = simulate_quantities(len(items), cfg.max_quantity, seed)

    products = pd.read_csv(raw_dir / "products.csv") \
        .merge(pd.read_csv(raw_dir / "aisles.csv"), on="aisle_id", how="left") \
        .merge(pd.read_csv(raw_dir / "departments.csv"), on="department_id", how="left")
    products = products[products["product_id"].isin(items["product_id"].unique())]
    products = products.rename(columns={"department": "category"})[
        ["product_id", "product_name", "category", "aisle"]]
    products = enrich_products(products, raw_dir, seed)

    customers = orders.groupby("user_id").agg(
        registration_date=("order_date", "min"),
        preferred_hour=("order_hour_of_day", lambda s: int(s.mode().iloc[0])),
    ).reset_index().rename(columns={"user_id": "customer_id"})
    customers = enrich_customers(customers, raw_dir, seed)

    orders_out = orders.rename(columns={"user_id": "customer_id"})[
        ["order_id", "customer_id", "order_number", "order_date", "order_hour_of_day"]]
    items_out = items[["order_id", "product_id", "quantity"]].copy()

    quality = {
        "source": source,
        "customers": int(len(customers)),
        "products": int(len(products)),
        "orders": int(len(orders_out)),
        "order_items": int(len(items_out)),
        "date_min": str(orders_out["order_date"].min().date()),
        "date_max": str(orders_out["order_date"].max().date()),
        "share_gaps_capped_at_30_days": round(capped_share, 4),
        "avg_basket_size": round(len(items_out) / max(len(orders_out), 1), 2),
        "quantity_simulated": quantity_simulated,
        "prices_simulated": not (raw_dir / "product_economics.csv").exists(),
        "household_size_simulated": not (raw_dir / "households.csv").exists(),
        "orders_without_items": int(len(order_ids - set(items_out["order_id"]))),
    }
    log.info("Données prêtes : %s", quality)
    return Dataset(customers, products, orders_out, items_out, source, quality)
