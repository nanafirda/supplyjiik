"""Génère reports/summary.md et les figures PNG réutilisables dans le rapport technique."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import REPORTS_DIR, SETTINGS  # noqa: E402
from database.db import get_metadata, query  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "prediction" / "artifacts"
BLUE, ORANGE, GREY = "#2a78d4", "#e07b39", "#9aa3ad"


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def make_figures(fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = []

    iv = query("SELECT days_since_prev FROM v_purchase_intervals WHERE days_since_prev IS NOT NULL")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(iv["days_since_prev"].clip(upper=90), bins=45, color=BLUE)
    ax.set_xlabel("Jours entre deux achats du même produit")
    _style(ax, "Distribution des intervalles de réachat")
    fig.tight_layout(); fig.savefig(fig_dir / "01_intervalles.png", dpi=150); plt.close(fig)
    files.append("01_intervalles.png")

    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    names = ["baseline_mean", "baseline_median", "random_forest"]
    labels = ["Baseline\nmoyenne", "Baseline\nmédiane", "Random\nForest"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [m[n]["MAE"] for n in names], 0.4, label="MAE", color=BLUE)
    ax.bar([i + 0.2 for i in x], [m[n]["RMSE"] for n in names], 0.4, label="RMSE", color=GREY)
    ax.set_xticks(list(x), labels); ax.set_ylabel("Jours"); ax.legend(frameon=False)
    _style(ax, "Erreur de prédiction sur la période de test")
    fig.tight_layout(); fig.savefig(fig_dir / "02_performance.png", dpi=150); plt.close(fig)
    files.append("02_performance.png")

    imp = pd.read_csv(ARTIFACTS / "feature_importances.csv", index_col=0)["importance"].head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE)
    _style(ax, "Importance des variables (Random Forest)")
    fig.tight_layout(); fig.savefig(fig_dir / "03_importances.png", dpi=150); plt.close(fig)
    files.append("03_importances.png")

    tp = pd.read_csv(ARTIFACTS / "test_predictions.csv").sample(frac=1, random_state=0).head(4000)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tp["days_until_next_purchase"], tp["pred_rf"], s=6, alpha=0.2, color=BLUE)
    lim = SETTINGS.model.horizon_days
    ax.plot([0, lim], [0, lim], color=ORANGE, lw=1.5)
    ax.set_xlabel("Délai réel (jours)"); ax.set_ylabel("Délai prédit (jours)")
    _style(ax, "Réel vs prédit (test)")
    fig.tight_layout(); fig.savefig(fig_dir / "04_reel_vs_predit.png", dpi=150); plt.close(fig)
    files.append("04_reel_vs_predit.png")

    bench = pd.read_csv(REPORTS_DIR / "benchmark_strategies.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [GREY, GREY, GREY, BLUE]
    ax.bar(bench["strategie"].str.replace(" (", "\n(", regex=False).str.replace("sans", "\nsans"),
           bench["profit_incremental"], color=colors[:len(bench)])
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("€")
    _style(ax, "Profit incrémental estimé par stratégie")
    fig.tight_layout(); fig.savefig(fig_dir / "05_strategies.png", dpi=150); plt.close(fig)
    files.append("05_strategies.png")
    return files


def write_summary(bench: pd.DataFrame | None = None) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    if bench is not None:
        bench.to_csv(REPORTS_DIR / "benchmark_strategies.csv", index=False)
    bench = pd.read_csv(REPORTS_DIR / "benchmark_strategies.csv")
    md = get_metadata()
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    figs = make_figures(REPORTS_DIR / "figures")
    dec = query("SELECT segment, decision, COUNT(*) n, ROUND(AVG(discount_rate), 3) avg_discount, "
                "ROUND(SUM(expected_profit), 2) profit FROM campaigns GROUP BY segment, decision")
    mk = SETTINGS.marketing

    reg = pd.DataFrame({k: m[k] for k in ("baseline_mean", "baseline_median", "random_forest")}).T
    reg = reg[["MAE", "RMSE", "MedAE", "within_3_days", "within_7_days"]].round(3)
    lines = [
        "# SupplyMe — synthèse des résultats",
        "",
        f"*Source des données : **{md.get('data_source')}** · date d'analyse : **{md.get('as_of_date')}***",
        "",
        "## 1. Données",
        "",
        "| Indicateur | Valeur |", "|---|---|",
        *[f"| {k.removeprefix('data_')} | {v} |" for k, v in md.items() if k.startswith("data_")],
        "",
        "## 2. Prédiction du prochain achat",
        "",
        f"Découpage temporel : {md.get('train_snapshots')} dates d'entraînement, {md.get('test_snapshots')} dates de test ; "
        f"{int(md.get('train_rows', 0)):_} observations d'entraînement, {int(md.get('test_rows', 0)):_} de test. "
        .replace("_", " ")
        + f"Part d'observations censurées (pas de réachat sous {SETTINGS.model.horizon_days} j) : "
        f"{float(md.get('censored_share', 0)):.1%}.",
        "",
        reg.to_markdown(),
        "",
        f"Gain de MAE du Random Forest vs baseline médiane : **{m['random_forest']['MAE_gain_vs_median']:.1%}**.",
        "",
        f"Classifieur « achat sous {SETTINGS.model.purchase_window_days} jours » : AUC = "
        f"**{m['rf_classifier_7d']['AUC']:.3f}**, Brier = {m['rf_classifier_7d']['Brier']:.3f} "
        f"(taux de base {m['rf_classifier_7d']['base_rate']:.1%}).",
        "",
        "## 3. Moteur de décision marketing",
        "",
        f"Hypothèses : effet d'une relance = +{mk.reminder_uplift:.0%} de conversion des non-acheteurs ; "
        f"élasticité à la réduction = {mk.discount_elasticity} ; coût d'un contact = {mk.contact_cost:.2f} € ; "
        f"réductions testées : {', '.join(f'{d:.0%}' for d in mk.discount_grid)}.",
        "",
        dec.to_markdown(index=False),
        "",
        "### Comparaison des stratégies",
        "",
        bench.to_markdown(index=False),
        "",
        "## 4. Figures",
        "",
        *[f"![{f}](figures/{f})" for f in figs],
        "",
        "## 5. Limites",
        "",
        "- Instacart ne contient ni dates absolues, ni prix, ni quantités, ni taille des foyers : ces éléments "
        "sont reconstruits ou simulés.",
        "- `days_since_prior_order` est plafonné à 30 jours dans Instacart : les longs intervalles sont sous-estimés.",
        "- Le réachat est utilisé comme approximation du besoin (pas de données de stock).",
        "- Les effets des relances et des réductions sont des hypothèses : une campagne test (A/B test) "
        "serait nécessaire pour les estimer.",
    ]
    out = REPORTS_DIR / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
