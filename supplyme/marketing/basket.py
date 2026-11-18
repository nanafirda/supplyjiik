"""Paniers personnalisés préremplis et aperçu du message de relance (non envoyé)."""
from __future__ import annotations

import pandas as pd

from config import DB_PATH
from database.db import query

BASKET_SQL = """
    SELECT c.customer_id, c.product_id, pr.product_name, pr.category, pr.unit_price,
           MAX(1, CAST(ROUND(h.avg_quantity) AS INTEGER)) AS quantity,
           c.discount_rate, c.decision, c.campaign_date, c.expected_profit,
           p.predicted_date
    FROM campaigns c
    JOIN products pr ON pr.product_id = c.product_id
    JOIN predictions p ON p.customer_id = c.customer_id AND p.product_id = c.product_id
    JOIN v_customer_product_habits h ON h.customer_id = c.customer_id AND h.product_id = c.product_id
    WHERE c.decision != 'C_aucune' {extra}
    ORDER BY c.customer_id, p.predicted_date, pr.product_name
"""


def build_baskets(customer_id: int | None = None, db_path=DB_PATH) -> pd.DataFrame:
    extra, params = ("AND c.customer_id = ?", (customer_id,)) if customer_id is not None else ("", ())
    b = query(BASKET_SQL.format(extra=extra), params, db_path=db_path)
    b["final_unit_price"] = (b["unit_price"] * (1 - b["discount_rate"])).round(2)
    b["line_total"] = (b["final_unit_price"] * b["quantity"]).round(2)
    b["savings"] = ((b["unit_price"] - b["final_unit_price"]) * b["quantity"]).round(2)
    return b


def basket_summary(db_path=DB_PATH) -> pd.DataFrame:
    b = build_baskets(db_path=db_path)
    return b.groupby("customer_id").agg(
        campaign_date=("campaign_date", "first"), n_products=("product_id", "size"),
        basket_value=("line_total", "sum"), savings=("savings", "sum"),
        expected_profit=("expected_profit", "sum")).round(2).reset_index()


def compose_message(basket: pd.DataFrame) -> str:
    """Texte d'e-mail de démonstration (aucun envoi réel : hors périmètre)."""
    if basket.empty:
        return "Aucune relance prévue pour ce client."
    lines = []
    for r in basket.itertuples():
        promo = f"  (-{r.discount_rate:.0%})" if r.discount_rate > 0 else ""
        lines.append(f"  • {r.product_name} × {r.quantity} — {r.line_total:.2f} €{promo}")
    total, savings = basket["line_total"].sum(), basket["savings"].sum()
    promo_line = f"\nVous économisez {savings:.2f} € grâce à vos offres personnalisées." if savings > 0 else ""
    return (
        "Objet : Votre panier habituel est prêt 🛒\n\n"
        "Bonjour,\n\n"
        "D'après vos habitudes, certains produits devraient bientôt vous manquer. "
        "Nous avons préparé votre panier :\n\n"
        + "\n".join(lines)
        + f"\n\nTotal : {total:.2f} €{promo_line}\n\n"
        "Validez en un clic ou modifiez-le avant la livraison.\n— L'équipe SupplyMe"
    )
