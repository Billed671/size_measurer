import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
MEASURE_LOG = LOG_DIR / "measurements.jsonl"
APP_LOG = LOG_DIR / "app.log"

_lock = Lock()
_initialized = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _initialized:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    fh = logging.FileHandler(APP_LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    _initialized.add(name)
    return logger


def log_measurement(record: dict) -> None:
    record = dict(record)
    record.setdefault(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )
    with _lock:
        with MEASURE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_measurements() -> list[dict]:
    if not MEASURE_LOG.exists():
        return []
    out = []
    with MEASURE_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out