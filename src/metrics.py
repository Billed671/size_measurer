import json
from pathlib import Path
from typing import Iterable

import numpy as np


def mae(pred: Iterable[float], gt: Iterable[float]) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def rmse(pred: Iterable[float], gt: Iterable[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2)))


def hit_rate(pred, gt, tol_frac: float = 0.05, tol_abs: float = 5.0) -> float:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    tol = np.maximum(tol_frac * np.abs(gt), tol_abs)
    return float(np.mean(np.abs(pred - gt) <= tol))


def evaluate(measurements: list[dict],
             ground_truth: dict[str, tuple[float, float, float]]) -> dict:
    pairs = [(m, ground_truth[m["id"]])
             for m in measurements if m["id"] in ground_truth]
    if not pairs:
        return {}

    out = {"n": len(pairs)}
    for i, axis in enumerate(("length_mm", "width_mm", "height_mm")):
        pred = [m[axis] for m, _ in pairs]
        gt = [g[i] for _, g in pairs]
        out[f"{axis}_mae"] = mae(pred, gt)
        out[f"{axis}_rmse"] = rmse(pred, gt)
        out[f"{axis}_hit"] = hit_rate(pred, gt)

    all_pred, all_gt = [], []
    for m, g in pairs:
        all_pred += [m["length_mm"], m["width_mm"], m["height_mm"]]
        all_gt += [g[0], g[1], g[2]]
    out["overall_hit"] = hit_rate(all_pred, all_gt)
    out["overall_mae"] = mae(all_pred, all_gt)
    out["overall_rmse"] = rmse(all_pred, all_gt)
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Метрики качества измерений.")
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path,
                   help="JSON: {id: [L, W, H]}")
    args = p.parse_args()

    from logging_utils import read_measurements  # noqa
    ms = read_measurements() if args.measurements.name == "auto" \
        else [json.loads(line) for line in args.measurements.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    if not args.ground_truth:
        print("Метрик без ground_truth не построить.")
        return
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    gt = {k: tuple(v) for k, v in gt.items()}

    report = evaluate(ms, gt)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()