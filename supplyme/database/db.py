"""Connexion SQLite, création du schéma, import et requêtes."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from config import DB_PATH, SCHEMA_PATH

log = logging.getLogger(__name__)


@contextmanager
def connect(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """(Re)crée toutes les tables, index et vues."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    log.info("Schéma SQLite initialisé : %s", db_path)


def _to_iso(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def write_table(df: pd.DataFrame, table: str, db_path: Path = DB_PATH, replace_rows: bool = False) -> None:
    """Insère un DataFrame dans une table existante (le schéma SQL fait foi)."""
    with connect(db_path) as conn:
        if replace_rows:
            conn.execute(f"DELETE FROM {table}")
        _to_iso(df).to_sql(table, conn, if_exists="append", index=False, chunksize=50_000)
    log.info("%-14s : %s lignes écrites", table, f"{len(df):,}")


def load_dataset(ds, db_path: Path = DB_PATH) -> None:
    """Importe un `Dataset` (data.instacart_loader) dans une base vierge, dans l'ordre des clés étrangères."""
    init_db(db_path)
    write_table(ds.customers[["customer_id", "household_size", "registration_date", "preferred_hour"]],
                "customers", db_path)
    write_table(ds.products[["product_id", "product_name", "category", "aisle", "unit_price", "unit_cost"]],
                "products", db_path)
    write_table(ds.orders, "orders", db_path)
    write_table(ds.order_items, "order_items", db_path)
    meta = {f"data_{k}": str(v) for k, v in ds.quality.items()}
    set_metadata(meta, db_path)


def set_metadata(values: dict, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                         [(k, str(v)) for k, v in values.items()])


def get_metadata(db_path: Path = DB_PATH) -> dict:
    with connect(db_path) as conn:
        return dict(conn.execute("SELECT key, value FROM metadata").fetchall())


def query(sql: str, params: tuple | dict = (), db_path: Path = DB_PATH,
          parse_dates: list[str] | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)


def check_integrity(db_path: Path = DB_PATH) -> dict:
    """Contrôles de cohérence rapides, utilisés à la fin de l'import."""
    with connect(db_path) as conn:
        cur = conn.cursor()
        checks = {
            "customers": cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
            "products": cur.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "orders": cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "order_items": cur.execute("SELECT COUNT(*) FROM order_items").fetchone()[0],
            "fk_violations": len(cur.execute("PRAGMA foreign_key_check").fetchall()),
            "orphan_orders": cur.execute(
                "SELECT COUNT(*) FROM orders o WHERE NOT EXISTS "
                "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)").fetchone()[0],
        }
    return checks
