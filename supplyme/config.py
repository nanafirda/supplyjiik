"""
Configuration centrale de SupplyMe.

Tous les chemins, paramètres de modélisation et HYPOTHÈSES ÉCONOMIQUES
sont regroupés ici pour être modifiés en un seul endroit et cités
tels quels dans le rapport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"                 # CSV Instacart (Kaggle) à déposer ici
SYNTH_RAW_DIR = ROOT / "data" / "raw_synthetic"  # CSV synthétiques au format Instacart
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DB_PATH = ROOT / "supplyme.db"
SCHEMA_PATH = ROOT / "database" / "schema.sql"

INSTACART_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)

# Instacart ne fournit pas de dates absolues : on ancre la reconstruction
# des calendriers clients sur cette date de fin d'historique.
END_DATE = date(2026, 10, 31)

RANDOM_SEED = 42


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    n_customers: int = 3000            # clients échantillonnés (Instacart ou synthétiques)
    min_orders_per_customer: int = 5   # on ignore les clients trop peu actifs
    max_quantity: int = 4              # quantités simulées (absentes d'Instacart)


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    min_purchases: int = 3        # nb min d'achats passés d'un couple client×produit
    horizon_days: int = 60        # fenêtre d'observation de la cible (au-delà : censuré)
    snapshot_step_days: int = 14  # écart entre deux dates de prédiction (snapshots)
    min_history_days: int = 60    # historique minimal avant le premier snapshot
    n_test_snapshots: int = 3     # snapshots réservés au test (les plus récents)
    max_train_snapshots: int = 12 # on garde les snapshots d'entraînement les plus récents
    max_train_rows: int = 120_000
    rf_n_estimators: int = 100
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 20
    purchase_window_days: int = 7  # cible du classifieur : achat organique sous 7 jours


# --------------------------------------------------------------------------
# Marketing — HYPOTHÈSES (à présenter comme telles, non apprises sur Instacart)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketingConfig:
    contact_window_days: int = 7          # on cible les besoins prévus dans les 7 prochains jours
    lead_days: int = 1                    # on relance 1 jour avant la date prévue
    overdue_ratio: float = 1.5            # besoin « dépassé » : >1,5× l'intervalle habituel sans achat
    abandon_ratio: float = 3.0            # au-delà de 3× l'intervalle habituel : produit considéré abandonné
    max_products_per_message: int = 12   # un panier relancé reste lisible
    contact_cost: float = 0.05            # coût d'un message (€ : routage, outil CRM, création)
    reminder_uplift: float = 0.08         # hausse de proba d'achat due à une simple relance
    discount_elasticity: float = 2.0      # +1 pt de réduction -> +2 pts de conversion des non-acheteurs
    max_probability: float = 0.95
    discount_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    benchmark_discount: float = 0.10      # relance de masse utilisée comme point de comparaison


# Taux de marge brute moyen par rayon (department) — hypothèses issues
# d'ordres de grandeur de la grande distribution alimentaire.
DEPARTMENT_MARGIN = {
    "produce": 0.35, "dairy eggs": 0.25, "snacks": 0.38, "beverages": 0.32,
    "frozen": 0.30, "pantry": 0.33, "bakery": 0.40, "canned goods": 0.30,
    "deli": 0.36, "dry goods pasta": 0.30, "household": 0.34, "breakfast": 0.32,
    "meat seafood": 0.22, "personal care": 0.42, "babies": 0.28, "international": 0.35,
    "alcohol": 0.27, "pets": 0.33, "missing": 0.30, "other": 0.30, "bulk": 0.30,
}
# Prix unitaire moyen (€) par rayon, utilisé pour simuler les prix Instacart.
DEPARTMENT_PRICE = {
    "produce": 2.2, "dairy eggs": 2.6, "snacks": 3.0, "beverages": 2.8,
    "frozen": 4.0, "pantry": 3.2, "bakery": 2.5, "canned goods": 1.9,
    "deli": 4.5, "dry goods pasta": 1.8, "household": 5.0, "breakfast": 3.5,
    "meat seafood": 8.0, "personal care": 5.5, "babies": 7.0, "international": 3.0,
    "alcohol": 9.0, "pets": 6.0, "missing": 3.0, "other": 3.0, "bulk": 4.0,
}


@dataclass(frozen=True)
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    marketing: MarketingConfig = field(default_factory=MarketingConfig)


SETTINGS = Settings()
