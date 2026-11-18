"""
Modèles de prédiction du prochain achat.

- Baselines statistiques : prochain achat = dernier achat + intervalle
  médian (ou moyen) du couple client × produit.
- Random Forest Regressor : délai (jours) avant le prochain achat.
- Random Forest Classifier : P(achat organique dans les 7 jours),
  utilisée par le moteur marketing comme probabilité de base.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (brier_score_loss, mean_absolute_error, mean_squared_error,
                             roc_auc_score)

from config import RANDOM_SEED, ModelConfig
from prediction.features import FEATURES

log = logging.getLogger(__name__)
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_predict(df: pd.DataFrame, stat: str = "median", horizon: int = 60) -> np.ndarray:
    interval = df[f"{stat}_purchase_interval"].to_numpy()
    return np.clip(interval - df["days_since_last_purchase"].to_numpy(), 1, horizon)


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": float(np.median(err)),
        "within_3_days": float((err <= 3).mean()),
        "within_7_days": float((err <= 7).mean()),
    }


def classification_metrics(y_true, proba) -> dict:
    return {"AUC": float(roc_auc_score(y_true, proba)),
            "Brier": float(brier_score_loss(y_true, proba)),
            "base_rate": float(np.mean(y_true)),
            "mean_proba": float(np.mean(proba))}


# ---------------------------------------------------------------------------
# Entraînement & évaluation
# ---------------------------------------------------------------------------
def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n, random_state=RANDOM_SEED) if len(df) > n else df


def make_regressor(cfg: ModelConfig) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                                 min_samples_leaf=cfg.rf_min_samples_leaf, max_features=0.5,
                                 n_jobs=-1, random_state=RANDOM_SEED)


def make_classifier(cfg: ModelConfig) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                                  min_samples_leaf=cfg.rf_min_samples_leaf, max_features=0.5,
                                  class_weight=None, n_jobs=-1, random_state=RANDOM_SEED)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, cfg: ModelConfig) -> dict:
    """Entraîne sur les snapshots passés, évalue sur les snapshots futurs."""
    tr_reg = _sample(train[~train["censored"]], cfg.max_train_rows)
    te_reg = test[~test["censored"]]
    y_te = te_reg["days_until_next_purchase"].to_numpy()

    reg = make_regressor(cfg).fit(tr_reg[FEATURES], tr_reg["days_until_next_purchase"])
    pred_rf = np.clip(reg.predict(te_reg[FEATURES]), 1, cfg.horizon_days)

    results = {
        "baseline_mean": regression_metrics(y_te, baseline_predict(te_reg, "mean", cfg.horizon_days)),
        "baseline_median": regression_metrics(y_te, baseline_predict(te_reg, "median", cfg.horizon_days)),
        "random_forest": regression_metrics(y_te, pred_rf),
    }
    gain = 1 - results["random_forest"]["MAE"] / results["baseline_median"]["MAE"]
    results["random_forest"]["MAE_gain_vs_median"] = float(gain)

    tr_cls = _sample(train, cfg.max_train_rows)
    clf = make_classifier(cfg).fit(tr_cls[FEATURES], tr_cls["buy_in_window"])
    proba = clf.predict_proba(test[FEATURES])[:, 1]
    results["rf_classifier_7d"] = classification_metrics(test["buy_in_window"], proba)

    # diagnostics pour le rapport et l'interface
    te_reg = te_reg.assign(pred_rf=pred_rf, pred_baseline=baseline_predict(te_reg, "median", cfg.horizon_days))
    by_cat = te_reg.assign(err_rf=(te_reg["pred_rf"] - y_te).abs(),
                           err_base=(te_reg["pred_baseline"] - y_te).abs()) \
        .groupby("category_code")[["err_rf", "err_base"]].mean()
    importances = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False)

    ARTIFACTS.mkdir(exist_ok=True)
    te_reg[["customer_id", "product_id", "snapshot", "days_until_next_purchase",
            "pred_rf", "pred_baseline"]].to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    importances.to_csv(ARTIFACTS / "feature_importances.csv", header=["importance"])
    by_cat.to_csv(ARTIFACTS / "mae_by_category_code.csv")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("Évaluation (test temporel, %s obs.) :", f"{len(te_reg):,}")
    for name, m in results.items():
        log.info("  %-17s %s", name, {k: round(v, 3) for k, v in m.items()})
    return results


def fit_final(full: pd.DataFrame, cfg: ModelConfig) -> tuple[RandomForestRegressor, RandomForestClassifier]:
    """Ré-entraîne sur l'ensemble des données étiquetées pour la prédiction de production."""
    reg_df = _sample(full[~full["censored"]], cfg.max_train_rows)
    reg = make_regressor(cfg).fit(reg_df[FEATURES], reg_df["days_until_next_purchase"])
    cls_df = _sample(full, cfg.max_train_rows)
    clf = make_classifier(cfg).fit(cls_df[FEATURES], cls_df["buy_in_window"])
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(reg, ARTIFACTS / "rf_regressor.joblib", compress=3)
    joblib.dump(clf, ARTIFACTS / "rf_classifier.joblib", compress=3)
    return reg, clf


def predict_next(snap: pd.DataFrame, reg, clf, cfg: ModelConfig) -> pd.DataFrame:
    """Prédictions à la date courante (as_of) pour tous les couples client × produit éligibles."""
    X = snap[FEATURES].astype(float).fillna(0.0)
    out = snap[["customer_id", "product_id", "snapshot", "last_purchase"]].copy()
    out["predicted_interval"] = np.clip(reg.predict(X), 1, cfg.horizon_days)
    out["baseline_interval"] = baseline_predict(snap, "median", cfg.horizon_days)
    out["purchase_proba"] = clf.predict_proba(X)[:, 1]
    out["predicted_date"] = out["snapshot"] + pd.to_timedelta(np.round(out["predicted_interval"]), unit="D")
    out["model_name"] = "random_forest"
    return out.rename(columns={"snapshot": "as_of_date", "last_purchase": "last_purchase_date"})
