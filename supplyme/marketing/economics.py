"""
Fonctions économiques du moteur de décision.

Raisonnement INCRÉMENTAL : une relance ne crée de la valeur que sur les
achats qu'elle provoque. Les clients qui auraient acheté de toute façon
(probabilité de base p0) ne doivent pas être comptés, et une réduction
qui leur est accordée est un coût pur (effet d'aubaine).

    m  = (prix - coût) × quantité              marge sans réduction
    R  = prix × quantité                        chiffre d'affaires
    pA = p0 + u·(1 - p0)                        relance simple
    pB = p0 + min(u + e·d, 1)·(1 - p0)          relance avec réduction d

    Profit_A    = (pA - p0)·m                          - coût de contact
    Profit_B(d) = pB·(m - d·R) - p0·m                  - coût de contact
    Profit_C    = 0                                    (référence)

u (uplift d'une relance) et e (élasticité à la réduction) sont des
HYPOTHÈSES de simulation (config.MarketingConfig), pas des estimations.
"""
from __future__ import annotations

import numpy as np

from config import MarketingConfig


def proba_with_reminder(p0, cfg: MarketingConfig):
    return np.minimum(p0 + cfg.reminder_uplift * (1 - p0), cfg.max_probability)


def proba_with_discount(p0, d, cfg: MarketingConfig):
    lift = np.minimum(cfg.reminder_uplift + cfg.discount_elasticity * d, 1.0)
    return np.minimum(p0 + lift * (1 - p0), cfg.max_probability)


def gross_profit_reminder(p0, margin, cfg: MarketingConfig):
    """Profit incrémental du scénario A, hors coût de contact."""
    return (np.maximum(proba_with_reminder(p0, cfg), p0) - p0) * margin


def gross_profit_discount(p0, margin, revenue, d, cfg: MarketingConfig):
    """Profit incrémental du scénario B pour un taux d, hors coût de contact."""
    p = np.maximum(proba_with_discount(p0, d, cfg), p0)
    return p * (margin - d * revenue) - p0 * margin


def expected_discount_cost(p0, revenue, d, cfg: MarketingConfig):
    """Coût attendu de la réduction (versée à tous les acheteurs, y compris organiques)."""
    return np.maximum(proba_with_discount(p0, d, cfg), p0) * d * revenue
