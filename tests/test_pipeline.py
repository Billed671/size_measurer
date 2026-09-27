import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_cloud import generate_cloud  # noqa: E402
from pipeline import run_pipeline          # noqa: E402


def test_single_object_recovered():
    cloud, n_obj = generate_cloud(n_objects=1, seed=1,
                                  floor_points=5000,
                                  object_points=3000)
    results = run_pipeline(cloud, seed=1)
    assert len(results) == 1
    r = results[0]
    assert 10 < r["L"] < 400
    assert 10 < r["W"] < 400
    assert 10 < r["H"] < 400


def test_three_objects_recovered():
    cloud, _ = generate_cloud(n_objects=3, seed=2,
                              floor_points=5000,
                              object_points=3000)
    results = run_pipeline(cloud, seed=2)
    assert len(results) >= 2