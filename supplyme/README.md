# 🛒 SupplyMe — *Predict. Anticipate. Convert.*

Plateforme de prédiction des besoins et d'optimisation des relances marketing pour une épicerie en ligne (MVP).

SupplyMe analyse l'historique d'achat des clients, prédit **quand** chaque produit habituel sera racheté, décide **s'il faut relancer** le client (relance simple, relance avec réduction ou aucune relance), choisit **le moment d'envoi**, prépare un **panier prérempli** et estime la **rentabilité incrémentale** de la campagne.

---

## 1. Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python main.py run                   # pipeline complet (≈ 2 min)
python main.py app                   # interface Streamlit -> http://localhost:8501
python -m pytest -q                  # tests
```

Sans les fichiers Instacart, `run` génère automatiquement un jeu **synthétique au format Instacart** : tout le projet fonctionne immédiatement.

Étapes séparées : `python main.py data | predict | marketing | report`
Options : `--source instacart|synthetic|auto` · `--n-customers 3000`

## 2. Utiliser le vrai dataset Instacart

1. Sur Kaggle, ouvrir la compétition **Instacart Market Basket Analysis** et accepter le règlement.
2. Télécharger les données (bouton *Download All*, ou `kaggle competitions download -c instacart-market-basket-analysis`).
3. Décompresser **tous** les fichiers (l'archive contient des `.csv.zip`) dans `data/raw/` :

```
data/raw/orders.csv
data/raw/order_products__prior.csv
data/raw/order_products__train.csv
data/raw/products.csv
data/raw/aisles.csv
data/raw/departments.csv
```

4. Lancer `python main.py run --source instacart --n-customers 3000`.

Le chargeur lit les gros fichiers par blocs et ne garde que les clients échantillonnés.

## 3. Arborescence

```
SupplyMe/
├── main.py                  Point d'entrée (CLI) — orchestre les 4 étapes
├── config.py                Chemins, paramètres des modèles, HYPOTHÈSES marketing
├── data/
│   ├── catalog.py           Catalogue synthétique (69 produits, cycles, prix)
│   ├── synthetic.py         Générateur au format Instacart + enrichissement économique
│   ├── instacart_loader.py  Chargement, échantillonnage, reconstruction des dates
│   ├── raw/                 CSV Instacart (à déposer)
│   └── raw_synthetic/       CSV synthétiques générés
├── database/
│   ├── schema.sql           Tables, contraintes, index, vues SQL (LAG)
│   └── db.py                Connexion, import, requêtes, contrôles d'intégrité
├── prediction/
│   ├── features.py          Snapshots temporels, variables, cibles (sans fuite)
│   ├── models.py            Baselines, Random Forest (régression + classification), métriques
│   ├── pipeline.py          Orchestration -> table predictions
│   └── artifacts/           Modèles, métriques, importances, prédictions de test
├── marketing/
│   ├── economics.py         Fonctions de profit incrémental
│   ├── decision.py          Moteur A / B / C, moment d'envoi, benchmarks
│   └── basket.py            Paniers préremplis + aperçu du message
├── app/
│   ├── streamlit_app.py     Interface 4 onglets
│   └── report.py            reports/summary.md + figures PNG
├── reports/                 Synthèse, figures, CSV de résultats (générés)
├── notebooks/
│   └── SupplyMe_Colab.ipynb Notebook Google Colab (pipeline + résultats + Streamlit)
├── tests/                   Tests unitaires (pytest)
└── supplyme.db              Base SQLite (générée)
```

## 4. Base de données

Tables : `customers`, `products`, `orders`, `order_items`, `predictions`, `campaigns`, `model_metrics`, `metadata`.
Le schéma du cahier des charges est respecté et complété (heure préférée, rayon fin, date de prédiction, probabilité de base, segment).
Deux vues SQL calculent les habitudes directement en base :

- `v_purchase_intervals` : intervalle entre deux achats d'un même produit (fonction de fenêtre `LAG`) ;
- `v_customer_product_habits` : nombre d'achats, intervalle moyen, quantité moyenne, dernier achat.

Contraintes `CHECK` (prix > coût, quantités > 0, décisions autorisées) et contrôle `PRAGMA foreign_key_check` après import.

## 5. Méthodologie

### 5.1 Données
- **Dates** : Instacart ne donne que l'ordre des commandes et `days_since_prior_order`. Le calendrier de chaque client est reconstruit par somme cumulée et ancré sur `END_DATE`. Les intervalles sont réels, les dates absolues sont une convention.
- **Variables absentes d'Instacart** (prix, coûts, quantités, taille du foyer) : simulées par rayon et signalées dans `metadata` (`*_simulated`).
- **Synthétique** : mécanisme de stock explicite (un produit est racheté quand il est épuisé), cycles dépendant de la taille du foyer, oublis, achats d'impulsion et abandons de produits.
- Le **réachat** est utilisé comme approximation du besoin (pas de données de stock réel).

### 5.2 Prédiction (sans fuite temporelle)
- À chaque date de prédiction *s* (*snapshot*, tous les 14 jours), les variables sont calculées **uniquement avec les achats ≤ s** ; la cible est `days_until_next_purchase` = délai jusqu'au premier réachat après *s*.
- **Découpage temporel** : les 3 snapshots les plus récents servent de test ; toutes les cibles d'entraînement sont observées **avant** la première date de test.
- **Censure** : sans réachat sous 60 jours, l'observation est exclue de la régression (mais gardée pour le classifieur).
- `customer_id` et `product_id` ne sont pas utilisés bruts : ils sont remplacés par des statistiques passées du client et du produit (encodage adapté).
- **Modèles** : baselines (intervalle moyen / médian − temps écoulé) vs **Random Forest Regressor** ; un **Random Forest Classifier** estime P(achat organique sous 7 jours), utilisée par le moteur marketing.
- **Évaluation** : MAE, RMSE, erreur médiane, part des prédictions à ±3 et ±7 jours ; AUC et score de Brier pour le classifieur.

### 5.3 Moteur de décision (raisonnement incrémental)

```
Profit_A    = (pA − p0)·marge                         − coût de contact
Profit_B(d) = pB·(marge − d·CA) − p0·marge            − coût de contact
Profit_C    = 0
```

`p0` = probabilité d'achat sans relance (estimée par le modèle). La réduction est versée à **tous** les acheteurs, y compris ceux qui auraient acheté de toute façon : c'est l'effet d'aubaine, qui rend les promotions de masse non rentables.
Deux segments : **besoin imminent** (achat prévu sous 7 jours) et **besoin dépassé** (aucun achat depuis plus de 1,5 × l'intervalle habituel : risque d'achat ailleurs).
Un seul message par client, envoyé si la somme des gains couvre le coût de contact, **1 jour avant le besoin** et **1 h avant son heure habituelle de commande**.

⚠️ L'effet d'une relance (+8 pts), l'élasticité à la réduction (2,0) et le coût de contact (0,05 €) sont des **hypothèses** (`config.py`), modifiables en direct dans l'onglet Marketing. Les estimer demanderait un A/B test.

## 6. Résultats (Instacart réel, échantillon de 3 000 clients)

54 832 commandes, 535 691 lignes produits, 25 404 produits. 9,5 % des écarts entre commandes sont plafonnés à 30 jours par Instacart.

| Modèle | MAE (j) | RMSE (j) |
|---|---|---|
| Baseline intervalle médian | 16,3 | 22,3 |
| Baseline intervalle moyen | 16,6 | 22,3 |
| **Random Forest** | **12,1** | **15,2** |

Random Forest : −26 % de MAE et −32 % de RMSE par rapport à la baseline médiane. Classifieur « achat sous 7 jours » : AUC = 0,77.

| Stratégie | Clients contactés | Profit incrémental |
|---|---|---|
| Aucune relance | 0 | 0 € |
| Relance de masse sans réduction | 1 485 | 329 € |
| Relance de masse −10 % | 1 485 | 479 € |
| **SupplyMe** | **1 428** | **650 €** |

Lecture marketing : sur les vraies données, les achats sont moins réguliers et la probabilité d'achat spontané est plus faible. Les réductions ciblées sur les **besoins dépassés** deviennent donc rentables, alors que les besoins imminents reçoivent une simple relance. Chaque message contient au plus 12 produits.
Les chiffres exacts sont régénérés dans `reports/summary.md` à chaque exécution (données synthétiques : `--source synthetic`).

## 7. Checklist de validation

| Critère | Où le vérifier |
|---|---|
| Dataset réel importé et exploitable | `data/instacart_loader.py`, `metadata` |
| Base SQL avec clients, produits, commandes | `supplyme.db`, `check_integrity()` |
| Habitudes d'achat calculées | vue `v_customer_product_habits`, onglet Clients |
| Le modèle prédit les prochains achats | table `predictions`, onglet Prédictions |
| Performances comparées à une baseline | `prediction/artifacts/metrics.json` |
| Trois décisions marketing distinguées | table `campaigns`, tests |
| Bénéfices estimés calculés | `marketing/economics.py`, `reports/benchmark_strategies.csv` |
| Panier personnalisé généré | `marketing/basket.py`, onglet Marketing |
| Interface sans erreur | `python main.py app` |
| Parcours client complet | onglet Marketing → « Parcours client » |

## 8. Limites et perspectives
- Dates, prix, quantités et foyers reconstruits ou simulés pour Instacart ; `days_since_prior_order` plafonné à 30 jours.
- Effets des relances et réductions supposés : à mesurer par A/B test, puis à modéliser en *uplift modeling*.
- Pistes : modèles de survie (gestion native de la censure), gradient boosting, intervalles de prédiction, table `campaign_results` pour comparer bénéfices estimés et observés.
