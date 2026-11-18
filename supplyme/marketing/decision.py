"""
Moteur de décision marketing : transforme les prédictions en actions.

1. Sélection : couples client × produit dont le besoin est prévu dans les
   `contact_window_days` prochains jours.
2. Pour chaque produit : profit incrémental du scénario A (relance) et du
   meilleur scénario B (grille de réductions).
3. Pour chaque client : UN message regroupe les produits rentables. Il est
   envoyé seulement si la somme des profits incrémentaux couvre le coût de
   contact ; sinon scénario C (aucune relance).
4. Moment d'envoi : `lead_days` avant le premier besoin prévu, une heure
   avant l'heure habituelle de commande du client.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import DB_PATH, MarketingConfig
from database.db import query, set_metadata, write_table
from marketing import economics as eco

log = logging.getLogger(__name__)


def load_candidates(cfg: MarketingConfig, db_path=DB_PATH) -> pd.DataFrame:
    df = query("""
        SELECT p.customer_id, p.product_id, p.as_of_date, p.predicted_date, p.predicted_interval,
               p.purchase_proba, pr.product_name, pr.category, pr.unit_price, pr.unit_cost,
               c.household_size, c.preferred_hour, h.avg_quantity, h.n_purchases, h.mean_interval,
               julianday(p.as_of_date) - julianday(p.last_purchase_date) AS days_since_last
        FROM predictions p
        JOIN products  pr ON pr.product_id = p.product_id
        JOIN customers c  ON c.customer_id = p.customer_id
        JOIN v_customer_product_habits h
             ON h.customer_id = p.customer_id AND h.product_id = p.product_id
    """, db_path=db_path, parse_dates=["as_of_date", "predicted_date"])
    as_of = df["as_of_date"].max()
    horizon = as_of + pd.Timedelta(days=cfg.contact_window_days)
    imminent = (df["predicted_date"] > as_of) & (df["predicted_date"] <= horizon)
    # besoin dépassé : le client aurait dû racheter mais ne l'a pas fait (risque d'achat ailleurs)
    overdue = (df["days_since_last"] >= cfg.overdue_ratio * df["mean_interval"]) & \
              (df["days_since_last"] <= cfg.abandon_ratio * df["mean_interval"]) & \
              (df["days_since_last"] <= 60)
    df["segment"] = np.where(overdue, "besoin_depasse", "besoin_imminent")
    df = df[imminent | overdue].copy()
    # un besoin dépassé est traité comme immédiat : relance dès le lendemain
    df.loc[df["segment"] == "besoin_depasse", "predicted_date"] = as_of + pd.Timedelta(days=cfg.lead_days + 1)
    df["quantity"] = df["avg_quantity"].round().clip(lower=1)
    df["revenue"] = df["unit_price"] * df["quantity"]
    df["margin"] = (df["unit_price"] - df["unit_cost"]) * df["quantity"]
    return df.reset_index(drop=True)


def score_products(df: pd.DataFrame, cfg: MarketingConfig) -> pd.DataFrame:
    p0, m, r = df["purchase_proba"].to_numpy(), df["margin"].to_numpy(), df["revenue"].to_numpy()
    out = df.copy()
    out["profit_A"] = eco.gross_profit_reminder(p0, m, cfg)
    grid = np.array(cfg.discount_grid)
    profits_B = np.column_stack([eco.gross_profit_discount(p0, m, r, d, cfg) for d in grid])
    best = profits_B.argmax(axis=1)
    out["best_discount"] = grid[best]
    out["profit_B"] = profits_B[np.arange(len(df)), best]
    out["option"] = np.where(out["profit_B"] > out["profit_A"], "B_promotion", "A_relance")
    out["gross_profit"] = out[["profit_A", "profit_B"]].max(axis=1)
    out["discount_rate"] = np.where(out["option"] == "B_promotion", out["best_discount"], 0.0)
    out["expected_proba"] = np.where(out["option"] == "B_promotion",
                                     eco.proba_with_discount(p0, out["discount_rate"], cfg),
                                     eco.proba_with_reminder(p0, cfg))
    # un produit non rentable n'est pas inclus dans le message
    out.loc[out["gross_profit"] <= 0, ["option", "discount_rate"]] = ["C_aucune", 0.0]
    # au plus `max_products_per_message` produits par message : on garde les plus rentables
    rank = out["gross_profit"].where(out["option"] != "C_aucune") \
        .groupby(out["customer_id"]).rank(ascending=False, method="first")
    out.loc[rank > cfg.max_products_per_message, ["option", "discount_rate"]] = ["C_aucune", 0.0]
    return out


def decide(df: pd.DataFrame, cfg: MarketingConfig) -> pd.DataFrame:
    df = score_products(df, cfg)
    included = df["option"] != "C_aucune"
    per_customer = df[included].groupby("customer_id")["gross_profit"].agg(["sum", "size"])
    send = per_customer.index[per_customer["sum"] > cfg.contact_cost]
    sent = df["customer_id"].isin(send) & included

    df["decision"] = np.where(sent, df["option"], "C_aucune")
    df.loc[~sent, "discount_rate"] = 0.0
    df.loc[~sent, "expected_proba"] = df.loc[~sent, "purchase_proba"]
    n_items = df[sent].groupby("customer_id")["product_id"].transform("size")
    df["expected_profit"] = 0.0
    df.loc[sent, "expected_profit"] = df.loc[sent, "gross_profit"] - cfg.contact_cost / n_items

    # moment d'envoi : lead_days avant le 1er besoin, 1 h avant l'heure habituelle
    need = df["predicted_date"] - pd.Timedelta(days=cfg.lead_days)
    send_day = need.where(need > df["as_of_date"], df["as_of_date"] + pd.Timedelta(days=1))
    first_day = send_day.where(sent).groupby(df["customer_id"]).transform("min")
    send_day = first_day.fillna(send_day)
    hour = (df["preferred_hour"].fillna(10) - 1).clip(lower=7).astype(int)
    df["campaign_date"] = send_day.dt.strftime("%Y-%m-%d") + " " + hour.map("{:02d}:00".format)
    df["base_proba"] = df["purchase_proba"]
    return df


def benchmarks(df: pd.DataFrame, cfg: MarketingConfig) -> pd.DataFrame:
    """Compare SupplyMe à des stratégies naïves sur la même population de besoins prévus."""
    p0, m, r = df["purchase_proba"].to_numpy(), df["margin"].to_numpy(), df["revenue"].to_numpy()
    n_cust = df["customer_id"].nunique()
    d = cfg.benchmark_discount
    sent = df["decision"] != "C_aucune"

    def row(name, profit, contacts, discount_cost, extra_orders):
        contact_spend = contacts * cfg.contact_cost
        spend = contact_spend + discount_cost
        return {"strategie": name, "clients_contactes": int(contacts),
                "achats_incrementaux": round(float(extra_orders), 1),
                "cout_contacts": round(contact_spend, 2), "cout_reductions": round(float(discount_cost), 2),
                "profit_incremental": round(float(profit), 2),
                "roi": round(float(profit / spend), 3) if spend > 0 else None}

    pa = eco.proba_with_reminder(p0, cfg)
    pb = eco.proba_with_discount(p0, d, cfg)
    sm_disc = eco.expected_discount_cost(p0[sent], r[sent], df.loc[sent, "discount_rate"].to_numpy(), cfg)
    return pd.DataFrame([
        row("Aucune relance", 0.0, 0, 0.0, 0.0),
        row("Relance de masse sans réduction",
            eco.gross_profit_reminder(p0, m, cfg).sum() - n_cust * cfg.contact_cost,
            n_cust, 0.0, (pa - p0).sum()),
        row(f"Relance de masse -{int(d * 100)} %",
            eco.gross_profit_discount(p0, m, r, d, cfg).sum() - n_cust * cfg.contact_cost,
            n_cust, eco.expected_discount_cost(p0, r, d, cfg).sum(), (pb - p0).sum()),
        row("SupplyMe (ciblé et optimisé)", df["expected_profit"].sum(),
            df.loc[sent, "customer_id"].nunique(), sm_disc.sum(),
            (df.loc[sent, "expected_proba"] - df.loc[sent, "base_proba"]).sum()),
    ])


def run_marketing(cfg: MarketingConfig, db_path=DB_PATH) -> pd.DataFrame:
    cand = load_candidates(cfg, db_path)
    if cand.empty:
        raise RuntimeError("Aucun besoin prévu dans la fenêtre de contact : vérifier les prédictions.")
    df = decide(cand, cfg)
    write_table(df[["customer_id", "product_id", "campaign_date", "segment", "discount_rate", "decision",
                    "base_proba", "expected_proba", "expected_profit"]],
                "campaigns", db_path, replace_rows=True)
    bench = benchmarks(df, cfg)
    log.info("Décisions : %s", df["decision"].value_counts().to_dict())
    log.info("Comparaison des stratégies :\n%s", bench.to_string(index=False))
    set_metadata({"campaign_customers": int(df.loc[df["decision"] != "C_aucune", "customer_id"].nunique()),
                  "campaign_expected_profit": round(float(df["expected_profit"].sum()), 2)}, db_path)
    return bench
