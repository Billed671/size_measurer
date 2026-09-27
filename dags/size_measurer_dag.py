"""
Airflow DAG: измерение габаритов каждые 3 секунды.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_pipeline, to_wms_payload  # noqa: E402
from generate_cloud import generate_cloud          # noqa: E402
from logging_utils import log_measurement          # noqa: E402


def task_measure(**_):
    cloud, _ = generate_cloud(seed=None)
    results = run_pipeline(cloud, source="airflow")
    for rec in to_wms_payload(results):
        log_measurement(rec)


def task_metrics(**_):
    from metrics import evaluate
    from logging_utils import read_measurements
    ms = read_measurements()
    # пример: без ground_truth просто считаем статистику
    if not ms:
        return
    n = len(ms)
    ok = sum(1 for m in ms if m.get("status") == "ok")
    print(f"[metrics] всего={n}, ok={ok}, "
          f"hit_rate≈{ok / n:.3f}")


default_args = {
    "owner": "cv-team",
    "retries": 1,
    "retry_delay": timedelta(seconds=2),
}

with DAG(
    dag_id="size_measurer",
    description="Измерение габаритов каждые 3 с",
    start_date=datetime(2026, 1, 1),
    schedule_interval=timedelta(seconds=3),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["cv", "conveyor"],
) as dag:

    measure = PythonOperator(
        task_id="measure",
        python_callable=task_measure,
    )
    metrics = PythonOperator(
        task_id="metrics",
        python_callable=task_metrics,
    )

    measure >> metrics