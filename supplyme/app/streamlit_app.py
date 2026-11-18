"""
SupplyMe — interface de démonstration (Streamlit).

Lancement :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from config import DB_PATH, SETTINGS  # noqa: E402
from database.db import get_metadata, query  # noqa: E402
from marketing.basket import build_baskets, compose_message  # noqa: E402
from marketing.decision import benchmarks, decide, load_candidates  # noqa: E402

ARTIFACTS = ROOT / "prediction" / "artifacts"
DECISION_LABELS = {"A_relance": "A — Relance classique", "B_promotion": "B — Relance promotionnelle",
                   "C_aucune": "C — Aucune relance"}
COLORS = {"A — Relance classique": "#2a78d4", "B — Relance promotionnelle": "#e07b39",
          "C — Aucune relance": "#9aa3ad"}

st.set_page_config(page_title="SupplyMe", page_icon="🛒", layout="wide")

if not DB_PATH.exists() and DB_PATH.with_suffix(".db.gz").exists():
    # version en ligne : la base est stockée compressée sur GitHub (limite de 25 Mo par fichier)
    import gzip
    import shutil
    with gzip.open(DB_PATH.with_suffix(".db.gz"), "rb") as src, open(DB_PATH, "wb") as dst:
        shutil.copyfileobj(src, dst)

if not DB_PATH.exists():
    st.error("Base introuvable. Lancez d'abord :  `python main.py run`")
    st.stop()


@st.cache_data(show_spinner=False)
def load(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query(sql, params)


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return get_metadata()


def eur(x: float) -> str:
    return f"{x:,.2f} €".replace(",", " ")


md = meta()
st.title("🛒 SupplyMe")
st.caption("Predict. Anticipate. Convert. — prédiction des besoins et optimisation des relances "
           f"· source : **{md.get('data_source', '?')}** · date d'analyse : **{md.get('as_of_date', '?')}**")

tab_dash, tab_cli, tab_pred, tab_mkt, tab_code = st.tabs(
    ["01 · Dashboard", "02 · Clients", "03 · Prédictions", "04 · Marketing", "05 · Méthode & code"])

# ---------------------------------------------------------------------------
# 01 Dashboard
# ---------------------------------------------------------------------------
with tab_dash:
    k = load("""SELECT (SELECT COUNT(*) FROM customers) c, (SELECT COUNT(*) FROM orders) o,
                       (SELECT COUNT(*) FROM products) p, (SELECT COUNT(*) FROM predictions) pr""").iloc[0]
    camp = load("SELECT decision, COUNT(*) n, COUNT(DISTINCT customer_id) cust, SUM(expected_profit) profit "
                "FROM campaigns GROUP BY decision")
    sent = camp[camp["decision"] != "C_aucune"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clients", f"{int(k.c):,}".replace(",", " "))
    c2.metric("Commandes", f"{int(k.o):,}".replace(",", " "))
    c3.metric("Prédictions actives", f"{int(k.pr):,}".replace(",", " "))
    c4.metric("Clients à relancer", f"{int(load('SELECT COUNT(DISTINCT customer_id) n FROM campaigns WHERE decision != \"C_aucune\"').n[0]):,}".replace(",", " "))
    c5.metric("Bénéfice incrémental estimé", eur(float(sent["profit"].sum())))

    left, right = st.columns(2)
    monthly = load("""SELECT strftime('%Y-%m', o.order_date) month, COUNT(DISTINCT o.order_id) orders,
                             ROUND(SUM(oi.quantity * p.unit_price), 2) revenue
                      FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
                      JOIN products p ON p.product_id = oi.product_id GROUP BY month ORDER BY month""")
    left.plotly_chart(px.bar(monthly, x="month", y="orders", title="Commandes par mois",
                             labels={"month": "", "orders": "Commandes"}), width="stretch")
    cat = load("""SELECT p.category, ROUND(SUM(oi.quantity * p.unit_price), 0) revenue
                  FROM order_items oi JOIN products p ON p.product_id = oi.product_id
                  GROUP BY p.category ORDER BY revenue DESC LIMIT 12""")
    right.plotly_chart(px.bar(cat, x="revenue", y="category", orientation="h", title="Chiffre d'affaires par rayon (€)",
                              labels={"revenue": "€", "category": ""}).update_yaxes(autorange="reversed"),
                       width="stretch")
    camp["Décision"] = camp["decision"].map(DECISION_LABELS)
    st.plotly_chart(px.bar(camp, x="Décision", y="n", color="Décision", color_discrete_map=COLORS,
                           title="Décisions marketing (couples client × produit)", labels={"n": "Produits"}),
                    width="stretch")
    with st.expander("Qualité des données"):
        st.json({k_: v for k_, v in md.items() if k_.startswith("data_")})

# ---------------------------------------------------------------------------
# 02 Clients
# ---------------------------------------------------------------------------
with tab_cli:
    candidates = load("SELECT DISTINCT customer_id FROM campaigns WHERE decision != 'C_aucune' ORDER BY customer_id")
    all_ids = load("SELECT customer_id FROM customers ORDER BY customer_id")["customer_id"].tolist()
    default = int(candidates["customer_id"].iloc[0]) if not candidates.empty else all_ids[0]
    cid = st.selectbox("Client", all_ids, index=all_ids.index(default), key="client")
    info = load("SELECT * FROM customers WHERE customer_id = ?", (cid,)).iloc[0]
    hist = load("""SELECT o.order_date, COUNT(*) n_items, ROUND(SUM(oi.quantity * p.unit_price), 2) amount
                   FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
                   JOIN products p ON p.product_id = oi.product_id
                   WHERE o.customer_id = ? GROUP BY o.order_id ORDER BY o.order_date""", (cid,))
    a, b, c, d = st.columns(4)
    a.metric("Taille du foyer", int(info.household_size))
    b.metric("Commandes", len(hist))
    c.metric("Panier moyen", eur(hist["amount"].mean()))
    d.metric("Heure habituelle", f"{int(info.preferred_hour)} h")
    st.plotly_chart(px.line(hist, x="order_date", y="amount", markers=True, title="Historique des commandes",
                            labels={"order_date": "", "amount": "Montant (€)"}), width="stretch")
    habits = load("""SELECT p.product_name AS Produit, p.category AS Rayon, h.n_purchases AS Achats,
                            h.mean_interval AS "Intervalle moyen (j)", h.avg_quantity AS "Qté moy.",
                            h.last_purchase AS "Dernier achat"
                     FROM v_customer_product_habits h JOIN products p ON p.product_id = h.product_id
                     WHERE h.customer_id = ? AND h.n_purchases >= 2 ORDER BY h.n_purchases DESC""", (cid,))
    st.subheader("Habitudes de consommation")
    st.dataframe(habits, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 03 Prédictions
# ---------------------------------------------------------------------------
with tab_pred:
    metrics_path = ARTIFACTS / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.subheader("Performance sur la période de test (découpage temporel)")
        reg = pd.DataFrame({k_: m[k_] for k_ in ("baseline_mean", "baseline_median", "random_forest")}).T
        reg = reg[["MAE", "RMSE", "MedAE", "within_3_days", "within_7_days"]]
        reg.index = ["Baseline : intervalle moyen", "Baseline : intervalle médian", "Random Forest"]
        x1, x2, x3 = st.columns(3)
        x1.metric("MAE Random Forest", f"{m['random_forest']['MAE']:.2f} j",
                  f"{-m['random_forest']['MAE_gain_vs_median']:.1%} vs baseline", delta_color="inverse")
        x2.metric("RMSE Random Forest", f"{m['random_forest']['RMSE']:.2f} j")
        x3.metric("AUC achat sous 7 j", f"{m['rf_classifier_7d']['AUC']:.3f}")
        st.dataframe(reg.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MedAE": "{:.2f}",
                                       "within_3_days": "{:.1%}", "within_7_days": "{:.1%}"}),
                     width="stretch")
        l, r = st.columns(2)
        imp = pd.read_csv(ARTIFACTS / "feature_importances.csv", index_col=0).reset_index(names="variable")
        l.plotly_chart(px.bar(imp.head(12), x="importance", y="variable", orientation="h",
                              title="Importance des variables").update_yaxes(autorange="reversed"),
                       width="stretch")
        tp = pd.read_csv(ARTIFACTS / "test_predictions.csv").sample(frac=1, random_state=0).head(3000)
        r.plotly_chart(px.scatter(tp, x="days_until_next_purchase", y="pred_rf", opacity=0.25,
                                  title="Réel vs prédit (échantillon test)",
                                  labels={"days_until_next_purchase": "Délai réel (j)", "pred_rf": "Délai prédit (j)"}),
                       width="stretch")
    st.subheader("Prochains achats estimés")
    win = st.slider("Besoins prévus dans les N prochains jours", 1, 30, 7)
    upcoming = load("""SELECT p.customer_id AS Client, pr.product_name AS Produit, p.last_purchase_date AS "Dernier achat",
                              p.predicted_date AS "Date prévue", ROUND(p.predicted_interval, 1) AS "Dans (j)",
                              ROUND(p.baseline_interval, 1) AS "Baseline (j)", ROUND(p.purchase_proba, 3) AS "P(achat 7 j)"
                       FROM predictions p JOIN products pr ON pr.product_id = p.product_id
                       WHERE p.predicted_interval <= ? ORDER BY p.predicted_date LIMIT 2000""", (win,))
    st.dataframe(upcoming, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 04 Marketing
# ---------------------------------------------------------------------------
with tab_mkt:
    st.info("Les effets d'une relance et d'une réduction sur la probabilité d'achat sont des **hypothèses de "
            "simulation**, ajustables ci-dessous ; la probabilité d'achat de base est estimée par le modèle.")
    base = SETTINGS.marketing
    s1, s2, s3 = st.columns(3)
    uplift = s1.slider("Effet d'une relance (pts de proba)", 0.0, 0.20, base.reminder_uplift, 0.01)
    elasticity = s2.slider("Élasticité à la réduction", 0.0, 3.0, base.discount_elasticity, 0.1)
    cost = s3.slider("Coût d'un contact (€)", 0.0, 1.0, base.contact_cost, 0.01)

    @st.cache_data(show_spinner="Simulation des campagnes…")
    def simulate(u: float, e: float, c: float):
        conf = type(base)(**{**base.__dict__, "reminder_uplift": u, "discount_elasticity": e, "contact_cost": c})
        dec = decide(load_candidates(conf), conf)
        return dec, benchmarks(dec, conf)

    dec, bench = simulate(uplift, elasticity, cost)
    st.subheader("Rentabilité comparée des stratégies")
    st.dataframe(bench.rename(columns={"strategie": "Stratégie", "clients_contactes": "Clients contactés",
                                       "achats_incrementaux": "Achats incrémentaux", "cout_contacts": "Coût contacts (€)",
                                       "cout_reductions": "Coût réductions (€)", "profit_incremental": "Profit incrémental (€)",
                                       "roi": "ROI"}), width="stretch", hide_index=True)
    st.plotly_chart(px.bar(bench, x="strategie", y="profit_incremental", color="strategie",
                           title="Profit incrémental estimé par stratégie (€)",
                           labels={"strategie": "", "profit_incremental": "€"}).update_layout(showlegend=False),
                    width="stretch")

    st.subheader("Parcours client : du besoin prévu au panier")
    sent_ids = sorted(dec.loc[dec["decision"] != "C_aucune", "customer_id"].unique().tolist())
    if sent_ids:
        cid_m = st.selectbox("Client relancé", sent_ids, key="mkt_client")
        view = dec[dec["customer_id"] == cid_m][["product_name", "segment", "predicted_date", "purchase_proba", "profit_A",
                                                 "profit_B", "best_discount", "decision", "expected_profit",
                                                 "campaign_date"]].copy()
        view["decision"] = view["decision"].map(DECISION_LABELS)
        st.dataframe(view.rename(columns={"product_name": "Produit", "segment": "Segment", "predicted_date": "Besoin prévu",
                                          "purchase_proba": "P0 (organique)", "profit_A": "Gain A (€)",
                                          "profit_B": "Gain B (€)", "best_discount": "Meilleure réduc.",
                                          "decision": "Décision", "expected_profit": "Profit net (€)",
                                          "campaign_date": "Envoi"}).round(
                         {"P0 (organique)": 3, "Gain A (€)": 3, "Gain B (€)": 3, "Profit net (€)": 3}),
                     width="stretch", hide_index=True)
        if (uplift, elasticity, cost) == (base.reminder_uplift, base.discount_elasticity, base.contact_cost):
            basket = build_baskets(cid_m)
            st.markdown("**Panier prérempli & message (aperçu, non envoyé)**")
            st.code(compose_message(basket), language=None)
        else:
            st.caption("Le panier et le message correspondent aux paramètres par défaut enregistrés en base.")
    else:
        st.warning("Avec ces hypothèses, aucune relance n'est rentable.")

# ---------------------------------------------------------------------------
# 05 Méthode & code
# ---------------------------------------------------------------------------
with tab_code:
    import inspect

    from data import instacart_loader
    from marketing import decision as decision_mod
    from marketing import economics
    from prediction import features as features_mod
    from prediction import models as models_mod

    def show(obj, caption: str) -> None:
        st.caption(caption)
        st.code(inspect.getsource(obj), language="python")

    st.markdown("""
Cet onglet montre **comment SupplyMe fonctionne** : chaque étape est expliquée simplement, et le
**vrai code exécuté** est affiché juste en dessous (il est lu directement dans les fichiers du projet).
Code complet : [github.com/firdaoussiadamou57-bit/supplyme](https://github.com/firdaoussiadamou57-bit/supplyme)
""")
    st.markdown("#### Le pipeline en 4 étapes")
    st.code("""
 1. DONNÉES        Instacart (CSV)  ──►  nettoyage + dates  ──►  base SQLite
 2. PRÉDICTION     historique d'achat ──► variables ──► Random Forest ──► date du prochain achat
 3. MARKETING      prédiction + marge ──► profit A / B / C ──► décision + moment d'envoi
 4. RESTITUTION    panier prérempli, message, rapport, application Streamlit
 Lancement :  python main.py run   (≈ 1 min 30)
""", language=None)

    with st.expander("① Données : Instacart et reconstruction des dates", expanded=True):
        st.markdown("""
- **Source** : Instacart Market Basket Analysis (Kaggle) — 3,4 M de commandes, 206 000 clients.
  On tire un **échantillon aléatoire de 3 000 clients** (graine fixe → résultats reproductibles).
- **Problème** : Instacart ne donne **pas de dates**, seulement le nombre de jours depuis la commande
  précédente. On reconstruit un calendrier par client en **additionnant ces écarts**.
  Les intervalles entre achats sont donc réels ; les dates exactes sont une convention.
- **Données absentes d'Instacart** (prix, coûts, quantités, taille du foyer) : **simulées** par rayon,
  et signalées comme telles dans la base.
- Tout est stocké dans une **base SQLite** (clients, produits, commandes, prédictions, campagnes).
""")
        show(instacart_loader.reconstruct_dates, "data/instacart_loader.py — reconstruction des dates")
        st.caption("database/schema.sql — calcul des habitudes d'achat directement en SQL (fonction LAG)")
        schema = (ROOT / "database" / "schema.sql").read_text(encoding="utf-8")
        st.code(schema[schema.index("CREATE VIEW v_purchase_intervals"):], language="sql")

    with st.expander("② Prédiction : quand le client va-t-il racheter ?"):
        st.markdown("""
- **Cible** : `days_until_next_purchase` = nombre de jours avant le prochain achat d'un produit par un client.
- **Variables** : temps écoulé depuis le dernier achat, intervalle moyen / médian entre achats, fréquence,
  quantité moyenne, régularité du client, popularité du produit…
- **Pas de fuite d'information** : on se place à une date passée (*snapshot*) et on calcule les variables
  **uniquement avec les achats antérieurs**. Le modèle est entraîné sur le passé et **testé sur les
  périodes les plus récentes**, jamais vues pendant l'entraînement.
- **Deux modèles comparés** :
  - **Baseline** (règle simple) : prochain achat = dernier achat + intervalle médian habituel ;
  - **Random Forest** : apprend des combinaisons de variables (ex. client irrégulier + produit rare).
- **Évaluation** : MAE (erreur moyenne en jours) et RMSE (pénalise les grosses erreurs).
- Un second Random Forest (classification) estime la **probabilité d'achat dans les 7 jours** :
  elle sert au moteur marketing.
""")
        mpath = ARTIFACTS / "metrics.json"
        if mpath.exists():
            mm = json.loads(mpath.read_text())
            st.markdown(
                f"**Résultat** : MAE baseline = {mm['baseline_median']['MAE']:.1f} j → Random Forest = "
                f"**{mm['random_forest']['MAE']:.1f} j** (soit **{mm['random_forest']['MAE_gain_vs_median']:.0%} "
                f"d'erreur en moins**) · AUC achat sous 7 jours = **{mm['rf_classifier_7d']['AUC']:.2f}**.")
        show(features_mod.make_snapshots, "prediction/features.py — découpage temporel train / test")
        show(features_mod.snapshot_features, "prediction/features.py — calcul des variables et de la cible")
        show(models_mod.baseline_predict, "prediction/models.py — baseline statistique")
        show(models_mod.make_regressor, "prediction/models.py — Random Forest")
        show(models_mod.evaluate, "prediction/models.py — entraînement et évaluation")

    with st.expander("③ Marketing : relancer, avec ou sans réduction ?"):
        st.markdown("""
On compare 3 scénarios pour chaque produit et on garde le plus rentable :

| Scénario | Action | Profit incrémental estimé |
|---|---|---|
| **A** | Relance simple | (pA − p0) × marge − coût du message |
| **B** | Relance + réduction d | pB × (marge − réduction) − p0 × marge − coût du message |
| **C** | Ne rien envoyer | 0 |

- **p0** = probabilité que le client achète **sans** relance (estimée par le modèle).
- On ne compte que les achats **provoqués** par la relance (*profit incrémental*). Une réduction donnée à un
  client qui aurait acheté de toute façon est une **perte** (effet d'aubaine) : c'est pourquoi la promo
  de masse est moins rentable.
- **Hypothèses** (non apprises sur les données, réglables dans l'onglet Marketing) : effet d'une relance,
  effet d'une réduction, coût d'un message. Il faudrait un **A/B test** réel pour les mesurer.
- **Moment d'envoi** : 1 jour avant le besoin prévu, 1 h avant l'heure habituelle de commande du client.
""")
        show(economics, "marketing/economics.py — fonctions de profit")
        show(decision_mod.decide, "marketing/decision.py — moteur de décision A / B / C")

    with st.expander("④ Limites et améliorations"):
        st.markdown("""
- Dates, prix, quantités et taille des foyers **reconstruits ou simulés** (absents d'Instacart).
- Instacart plafonne les écarts entre commandes à **30 jours** → les longs cycles sont sous-estimés.
- Le **réachat** sert d'approximation du besoin (pas de données de stock réel).
- Effets marketing **supposés** → à mesurer par A/B test, puis *uplift modeling*.
- Pistes : modèles de survie (gestion de la censure), gradient boosting, intervalles de confiance.
""")

    with st.expander("💬 Questions possibles du jury — réponses courtes"):
        st.markdown("""
**Pourquoi un Random Forest ?** Il capte des effets non linéaires et des interactions sans hypothèse
de forme, il est robuste aux valeurs extrêmes et donne l'importance des variables (interprétabilité).

**Comment savez-vous que le modèle est bon ?** On le compare à une règle simple (baseline) sur une période
future jamais vue : il réduit l'erreur moyenne d'environ un quart.

**Pourquoi un découpage temporel et pas aléatoire ?** Un découpage aléatoire mélangerait passé et futur :
le modèle « verrait l'avenir » et ses performances seraient surestimées (fuite d'information).

**Que faites-vous des clients qui ne rachètent jamais ?** Ce sont des observations censurées : exclues de la
régression (délai inconnu) mais gardées pour la probabilité d'achat sous 7 jours.

**Les probabilités de conversion sont-elles réelles ?** La probabilité d'achat spontané est estimée par le
modèle ; l'effet de la relance et de la réduction sont des hypothèses de simulation, clairement indiquées.

**Pourquoi ne pas faire des promotions à tout le monde ?** Parce qu'on paie la réduction aux clients qui
auraient acheté quand même : la stratégie ciblée de SupplyMe rapporte plus.
""")
