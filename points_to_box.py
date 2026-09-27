"""
pipeline.py

Пайплайн измерения габаритов товаров на конвейере по облаку точек.

Этапы:
    0. Источник точек: --input cloud.npz  или  --random
    1. RANSAC — удаление плоскости ленты
    2. Statistical Outlier Removal
    3. Воксельная фильтрация
    4. DBSCAN — сегментация объектов
    5. Минимальный 2D-OBB (вращающиеся калиперы по выпуклой оболочке)
    6. Валидация диапазона + confidence
    7. JSON для WMS

Использование:
    python pipeline.py --input cloud.npz --out-json measurements.json
    python pipeline.py --random --seed 7 --out-json measurements.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

# импорт генератора, чтобы можно было запускать без файла
from generate_cloud import generate_cloud


# ---------------------------------------------------------------------------
# 1. RANSAC — плоскость ленты
# ---------------------------------------------------------------------------

def ransac_plane(points: np.ndarray,
                 n_iter: int = 300,
                 thresh: float = 2.0,
                 seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Ищет доминирующую плоскость ax+by+cz+d=0.
    Возвращает (нормаль n с n_z >= 0, d, маску инлайеров).
    """
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
    # нормаль всегда "вверх"
    if n[2] < 0:
        n, d = -n, -d
    return n, d, best_inliers


# ---------------------------------------------------------------------------
# 2. Statistical Outlier Removal
# ---------------------------------------------------------------------------

def statistical_outlier_removal(points: np.ndarray,
                                k: int = 20,
                                std_ratio: float = 2.0) -> np.ndarray:
    if points.shape[0] <= k + 1:
        return np.ones(points.shape[0], dtype=bool)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d < thresh


# ---------------------------------------------------------------------------
# 3. Воксельная фильтрация
# ---------------------------------------------------------------------------

def voxel_downsample(points: np.ndarray, voxel: float = 5.0) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


# ---------------------------------------------------------------------------
# 5. Минимальный 2D-OBB
# ---------------------------------------------------------------------------

def minimal_obb_2d(proj: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """
    Минимальный по площади описывающий прямоугольник для 2D-точек.
    Возвращает (L, W, corners (4,2), theta).

    Если точек < 3 или они вырождены — возвращает bbox.
    """
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
    # длины сторон
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    L = float(max(sides))
    W = float(min(sides))
    theta = float(np.arctan2(d[1], d[0]))
    return L, W, corners, theta


# ---------------------------------------------------------------------------
# 6. Измерение одного кластера
# ---------------------------------------------------------------------------

def measure_cluster(cluster: np.ndarray) -> dict:
    proj = cluster[:, :2]
    H = float(cluster[:, 2].max())
    L, W, corners, theta = minimal_obb_2d(proj)

    # confidence: эвристика по плотности (точек / мм^2 проекции)
    area = max(L * W, 1.0)
    density = cluster.shape[0] / area
    conf = float(np.clip(density / 0.5, 0.0, 1.0))

    return dict(L=L, W=W, H=H,
                n_points=int(cluster.shape[0]),
                confidence=conf,
                corners=corners.tolist(),
                theta=theta)


# ---------------------------------------------------------------------------
# Полный пайплайн
# ---------------------------------------------------------------------------

def run_pipeline(cloud: np.ndarray,
                 voxel: float = 5.0,
                 dbscan_eps: float = 15.0,
                 dbscan_min_samples: int = 10,
                 seed: int = 0,
                 verbose: bool = True) -> list[dict]:
    log = print if verbose else (lambda *a, **k: None)
    log(f"[pipeline] вход: {cloud.shape[0]} точек")

    # 1. RANSAC — плоскость
    n_plane, d_plane, floor_mask = ransac_plane(cloud, seed=seed)
    log(f"[pipeline] плоскость: n={n_plane.round(3)}, d={d_plane:.2f}, "
        f"инлайеров={floor_mask.sum()}")

    # оставляем точки ВЫШЕ плоскости
    dist = cloud @ n_plane + d_plane
    above = dist > 3.0
    obj = cloud[above]
    log(f"[pipeline] над плоскостью: {obj.shape[0]}")

    # 2. SOR
    keep = statistical_outlier_removal(obj, k=20, std_ratio=2.0)
    obj = obj[keep]
    log(f"[pipeline] после SOR: {obj.shape[0]}")

    # 3. Воксель
    obj = voxel_downsample(obj, voxel=voxel)
    log(f"[pipeline] после вокселя: {obj.shape[0]}")

    # 4. DBSCAN
    if obj.shape[0] == 0:
        log("[pipeline] нет точек для кластеризации")
        return []
    labels = DBSCAN(eps=dbscan_eps,
                    min_samples=dbscan_min_samples).fit_predict(obj)
    uniq = [int(l) for l in np.unique(labels) if l != -1]
    log(f"[pipeline] кластеров: {len(uniq)}")

    # 5-6. Измерение + валидация
    results = []
    for lbl in uniq:
        cluster = obj[labels == lbl]
        if cluster.shape[0] < 50:
            continue
        r = measure_cluster(cluster)
        r["cluster_id"] = lbl

        ok = all(10.0 <= r[k] <= 400.0 for k in ("L", "W", "H"))
        if not ok:
            log(f"[pipeline] кластер {lbl}: габариты вне диапазона "
                f"L={r['L']:.1f} W={r['W']:.1f} H={r['H']:.1f}")
            continue
        if r["confidence"] < 0.5:
            log(f"[pipeline] кластер {lbl}: низкий confidence "
                f"{r['confidence']:.2f}")
            continue
        results.append(r)
        log(f"[pipeline] кластер {lbl}: L={r['L']:.1f} W={r['W']:.1f} "
            f"H={r['H']:.1f} conf={r['confidence']:.2f}")

    return results


def to_wms_payload(results: list[dict],
                   timestamp: str = "2026-09-17T12:00:00Z") -> list[dict]:
    payload = []
    for i, r in enumerate(results):
        payload.append({
            "id": f"SKU-{1000 + i}",
            "timestamp": timestamp,
            "length_mm": round(r["L"], 1),
            "width_mm":  round(r["W"], 1),
            "height_mm": round(r["H"], 1),
            "confidence": round(r["confidence"], 3),
        })
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Пайплайн измерения габаритов.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path,
                     help="Путь к .npz с массивом 'points'")
    src.add_argument("--random", action="store_true",
                     help="Сгенерировать облако случайно")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-objects", type=int, default=None,
                        help="Только для --random")
    parser.add_argument("--voxel", type=float, default=5.0)
    parser.add_argument("--eps", type=float, default=15.0)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--out-json", type=Path, default=Path("measurements.json"))
    args = parser.parse_args()

    if args.input is not None:
        data = np.load(args.input)
        cloud = data["points"].astype(np.float64)
        print(f"[main] загружено {cloud.shape[0]} точек из {args.input}")
    elif args.random:
        cloud, n_obj = generate_cloud(n_objects=args.n_objects, seed=args.seed)
        print(f"[main] сгенерировано {cloud.shape[0]} точек, объектов: {n_obj}")
    else:
        parser.error("укажите --input или --random")

    results = run_pipeline(
        cloud,
        voxel=args.voxel,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        seed=args.seed,
    )
    payload = to_wms_payload(results)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[main] {len(payload)} объектов -> {args.out_json}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()