-- =====================================================================
-- SupplyMe — schéma relationnel (SQLite)
-- Dates stockées au format ISO 8601 (TEXT), comme recommandé par SQLite.
-- =====================================================================
PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS v_customer_product_habits;
DROP VIEW  IF EXISTS v_purchase_intervals;
DROP TABLE IF EXISTS campaigns;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_metrics;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS metadata;

CREATE TABLE customers (
    customer_id        INTEGER PRIMARY KEY,
    household_size     INTEGER NOT NULL CHECK (household_size BETWEEN 1 AND 10),
    registration_date  DATE    NOT NULL,
    preferred_hour     INTEGER CHECK (preferred_hour BETWEEN 0 AND 23)
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT,
    aisle         TEXT,
    unit_price    REAL NOT NULL CHECK (unit_price > 0),
    unit_cost     REAL NOT NULL CHECK (unit_cost >= 0),
    CHECK (unit_cost < unit_price)
);

CREATE TABLE orders (
    order_id           INTEGER PRIMARY KEY,
    customer_id        INTEGER NOT NULL,
    order_number       INTEGER,
    order_date         DATE    NOT NULL,
    order_hour_of_day  INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_item_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       INTEGER NOT NULL,
    product_id     INTEGER NOT NULL,
    quantity       INTEGER NOT NULL CHECK (quantity > 0),
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL,
    product_id          INTEGER NOT NULL,
    as_of_date          DATE    NOT NULL,   -- date à laquelle la prédiction est faite
    last_purchase_date  DATE,
    predicted_date      DATE    NOT NULL,   -- date estimée du prochain achat
    predicted_interval  REAL    NOT NULL,   -- jours entre as_of_date et le prochain achat
    baseline_interval   REAL,               -- même quantité selon la baseline médiane
    purchase_proba      REAL,               -- P(achat organique dans la fenêtre)
    model_name          TEXT    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES products(product_id)
);

CREATE TABLE campaigns (
    campaign_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id        INTEGER NOT NULL,
    product_id         INTEGER NOT NULL,
    campaign_date      TEXT    NOT NULL,    -- date + heure d'envoi recommandées
    segment            TEXT,                -- besoin_imminent | besoin_depasse
    discount_rate      REAL    NOT NULL DEFAULT 0,
    decision           TEXT    NOT NULL CHECK (decision IN ('A_relance', 'B_promotion', 'C_aucune')),
    base_proba         REAL,
    expected_proba     REAL,
    expected_profit    REAL,                -- bénéfice INCRÉMENTAL estimé (€)
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES products(product_id)
);

CREATE TABLE model_metrics (
    model_name  TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL,
    split       TEXT NOT NULL,
    PRIMARY KEY (model_name, metric, split)
);

CREATE TABLE metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_items_order          ON order_items(order_id);
CREATE INDEX idx_items_product        ON order_items(product_id);
CREATE INDEX idx_pred_customer        ON predictions(customer_id);
CREATE INDEX idx_camp_customer        ON campaigns(customer_id);

-- Historique d'achat client × produit avec l'intervalle depuis l'achat précédent
-- (fonction de fenêtre LAG).
CREATE VIEW v_purchase_intervals AS
SELECT
    o.customer_id,
    oi.product_id,
    o.order_date,
    oi.quantity,
    julianday(o.order_date) - julianday(
        LAG(o.order_date) OVER (PARTITION BY o.customer_id, oi.product_id ORDER BY o.order_date)
    ) AS days_since_prev
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id;

-- Habitudes de consommation par couple client × produit.
CREATE VIEW v_customer_product_habits AS
SELECT
    customer_id,
    product_id,
    COUNT(*)                         AS n_purchases,
    MIN(order_date)                  AS first_purchase,
    MAX(order_date)                  AS last_purchase,
    ROUND(AVG(days_since_prev), 2)   AS mean_interval,
    ROUND(AVG(quantity), 2)          AS avg_quantity
FROM v_purchase_intervals
GROUP BY customer_id, product_id;
