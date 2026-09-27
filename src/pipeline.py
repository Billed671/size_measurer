from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

from logging_utils import get_logger, log_measurement
from generate_cloud import generate_cloud

log = get_logger("pipeline")


# ---------------------------------------------------------------- RANSAC
def ransac_plane(points, n_iter=300, thresh=2.0, seed=0):
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    best_inliers = None
    best_model = None
    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (n, d)
    n, d = best_model
    if n[2] < 0:
        n, d = -n, -d
    return n, d, best_inliers


# ---------------------------------------------------------------- SOR
def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    if points.shape[0] <= k + 1:
        return np.ones(points.shape[0], dtype=bool)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d < thresh


# ---------------------------------------------------------------- voxel
def voxel_downsample(points, voxel=5.0):
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


# ---------------------------------------------------------------- 2D OBB
def minimal_obb_2d(proj):
    if proj.shape[0] < 3:
        mn, mx = proj.min(axis=0), proj.max(axis=0)
        corners = np.array([[mn[0], mn[1]], [mx[0], mn[1]],
                            [mx[0], mx[1]], [mn[0], mx[1]]])
        return float(mx[0] - mn[0]), float(mx[1] - mn[1]), corners, 0.0

    try:
        hull = ConvexHull(proj)
    except Exception:
        mn, mx = proj.min(axis=0), proj.max(axis=0)
        corners = np.array([[mn[0], mn[1]], [mx[0], mn[1]],
                            [mx[0], mx[1]], [mn[0], mx[1]]])
        return float(mx[0] - mn[0]), float(mx[1] - mn[1]), corners, 0.0

    hp = proj[hull.vertices]
    m = hp.shape[0]
    best_area = np.inf
    best = None
    for i in range(m):
        p1, p2 = hp[i], hp[(i + 1) % m]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        d = edge / length
        n = np.array([-d[1], d[0]])
        u = hp @ d
        v = hp @ n
        w = u.max() - u.min()
        h = v.max() - v.min()
        area = w * h
        if area < best_area:
            best_area = area
            best = (u.min(), u.max(), v.min(), v.max(), d, n)

    u0, u1, v0, v1, d, n = best
    corners = np.array([
        u0 * d + v0 * n,
        u1 * d + v0 * n,
        u1 * d + v1 * n,
        u0 * d + v1 * n,
    ])
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i])
             for i in range(4)]
    L = float(max(sides))
    W = float(min(sides))
    theta = float(np.arctan2(d[1], d[0]))
    return L, W, corners, theta


# ---------------------------------------------------------------- measure
def measure_cluster(cluster):
    proj = cluster[:, :2]
    H = float(cluster[:, 2].max())
    L, W, corners, theta = minimal_obb_2d(proj)
    area = max(L * W, 1.0)
    density = cluster.shape[0] / area
    conf = float(np.clip(density / 0.5, 0.0, 1.0))
    return dict(L=L, W=W, H=H,
                n_points=int(cluster.shape[0]),
                confidence=conf,
                corners=corners.tolist(),
                theta=theta)


# ---------------------------------------------------------------- status
def classify_status(L, W, H, confidence):
    if not all(10.0 <= v <= 400.0 for v in (L, W, H)):
        return "recheck"
    if confidence < 0.5:
        return "recheck"
    if confidence < 0.8:
        return "suspicious"
    return "ok"


# ---------------------------------------------------------------- pipeline
def run_pipeline(cloud,
                 voxel=5.0,
                 dbscan_eps=15.0,
                 dbscan_min_samples=10,
                 seed=0,
                 source="unknown"):
    t0 = time.perf_counter()
    log.info("старт: %d точек, источник=%s", cloud.shape[0], source)

    n_plane, d_plane, _ = ransac_plane(cloud, seed=seed)
    dist = cloud @ n_plane + d_plane
    obj = cloud[dist > 3.0]

    keep = statistical_outlier_removal(obj)
    obj = obj[keep]

    obj = voxel_downsample(obj, voxel=voxel)

    if obj.shape[0] == 0:
        log.warning("нет точек для кластеризации")
        return []

    labels = DBSCAN(eps=dbscan_eps,
                    min_samples=dbscan_min_samples).fit_predict(obj)
    uniq = [int(l) for l in np.unique(labels) if l != -1]
    log.info("кластеров: %d", len(uniq))

    results = []
    for lbl in uniq:
        cluster = obj[labels == lbl]
        if cluster.shape[0] < 50:
            continue
        r = measure_cluster(cluster)
        r["cluster_id"] = lbl
        r["status"] = classify_status(r["L"], r["W"], r["H"],
                                      r["confidence"])
        r["run_id"] = str(uuid.uuid4())
        r["source"] = source
        r["duration_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        results.append(r)
        log.info("кластер %d: L=%.1f W=%.1f H=%.1f conf=%.2f [%s]",
                 lbl, r["L"], r["W"], r["H"], r["confidence"], r["status"])
    return results


def to_wms_payload(results, timestamp=None):
    from datetime import datetime, timezone
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": f"SKU-{1000 + i}",
            "timestamp": ts,
            "length_mm": round(r["L"], 1),
            "width_mm":  round(r["W"], 1),
            "height_mm": round(r["H"], 1),
            "confidence": round(r["confidence"], 3),
            "status": r["status"],
            "duration_ms": r.get("duration_ms"),
            "source": r.get("source", "unknown"),
        }
        for i, r in enumerate(results)
    ]


# ---------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path)
    src.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-objects", type=int, default=None)
    p.add_argument("--voxel", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=15.0)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--out-json", type=Path, default=Path("measurements.json"))
    p.add_argument("--log-measurements", action="store_true",
                   help="писать результаты в logs/measurements.jsonl")
    args = p.parse_args()

    if args.input is not None:
        cloud = np.load(args.input)["points"].astype(np.float64)
        source = str(args.input)
    elif args.random:
        cloud, n_obj = generate_cloud(n_objects=args.n_objects,
                                      seed=args.seed)
        source = f"random(seed={args.seed},n={n_obj})"
    else:
        p.error("укажите --input или --random")

    results = run_pipeline(
        cloud,
        voxel=args.voxel,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        seed=args.seed,
        source=source,
    )
    payload = to_wms_payload(results)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.log_measurements:
        for rec in payload:
            log_measurement(rec)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()