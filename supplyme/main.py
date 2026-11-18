"""
SupplyMe — point d'entrée.

Exemples :
    python main.py run                          # pipeline complet (Instacart si présent, sinon synthétique)
    python main.py run --source synthetic --n-customers 2000
    python main.py data | predict | marketing | report
    python main.py app                          # lance l'interface Streamlit
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
import time

from config import RAW_DIR, ROOT, SETTINGS, SYNTH_RAW_DIR, DataConfig

log = logging.getLogger("supplyme")


def step_data(source: str, cfg: DataConfig) -> None:
    from data.instacart_loader import instacart_available, load_instacart
    from data.synthetic import generate_instacart_like
    from database.db import check_integrity, load_dataset

    if source == "auto":
        source = "instacart" if instacart_available(RAW_DIR) else "synthetic"
        if source == "synthetic":
            log.warning("Fichiers Instacart absents de %s -> données synthétiques au format Instacart.", RAW_DIR)
    if source == "instacart":
        if not instacart_available(RAW_DIR):
            sys.exit(f"Fichiers Instacart manquants dans {RAW_DIR} (voir README).")
        ds = load_instacart(RAW_DIR, cfg, source="instacart")
    else:
        generate_instacart_like(SYNTH_RAW_DIR, n_customers=cfg.n_customers)
        ds = load_instacart(SYNTH_RAW_DIR, cfg, source="synthetic")
    load_dataset(ds)
    checks = check_integrity()
    log.info("Contrôles d'intégrité : %s", checks)
    if checks["fk_violations"] or checks["orphan_orders"]:
        raise RuntimeError("Base incohérente : voir les contrôles ci-dessus.")


def step_predict() -> None:
    from prediction.pipeline import run_prediction
    run_prediction(SETTINGS.model)


def step_marketing():
    from config import REPORTS_DIR
    from marketing.basket import basket_summary
    from marketing.decision import run_marketing
    bench = run_marketing(SETTINGS.marketing)
    REPORTS_DIR.mkdir(exist_ok=True)
    bench.to_csv(REPORTS_DIR / "benchmark_strategies.csv", index=False)
    basket_summary().to_csv(REPORTS_DIR / "baskets_summary.csv", index=False)
    return bench


def step_report() -> None:
    from app.report import write_summary
    out = write_summary()
    log.info("Synthèse écrite : %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="SupplyMe — pipeline de bout en bout")
    parser.add_argument("command", choices=["run", "data", "predict", "marketing", "report", "app"])
    parser.add_argument("--source", choices=["auto", "instacart", "synthetic"], default="auto")
    parser.add_argument("--n-customers", type=int, default=SETTINGS.data.n_customers)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    if args.command == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")])
        return

    data_cfg = dataclasses.replace(SETTINGS.data, n_customers=args.n_customers)
    steps = {
        "data": lambda: step_data(args.source, data_cfg),
        "predict": step_predict,
        "marketing": step_marketing,
        "report": step_report,
    }
    todo = list(steps) if args.command == "run" else [args.command]
    t0 = time.time()
    for name in todo:
        log.info("=" * 22 + f" Étape : {name} " + "=" * 22)
        t = time.time()
        steps[name]()
        log.info("Étape %s terminée en %.1f s", name, time.time() - t)
    log.info("Terminé en %.1f s. Interface : python main.py app", time.time() - t0)


if __name__ == "__main__":
    main()
