"""
generate_cloud.py

Генератор синтетического облака точек для задачи измерения габаритов
на конвейере.

Точки генерируются на поверхности:
  * плоскости ленты конвейера (с шумом);
  * одного или нескольких (с малой вероятностью) выпуклых многогранников
    произвольной формы — углы между гранями не фиксированы;
  * плюс немного случайных выбросов.

Система координат:
    X — поперёк ленты,  мм
    Y — вдоль ленты,    мм
    Z — вверх,          Z = 0 — плоскость ленты

Использование:
    python generate_cloud.py --out cloud.npz --seed 42
    python generate_cloud.py --out cloud.npz --n-objects 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Публичные функции — их же использует pipeline.py при --random
# ---------------------------------------------------------------------------

def sample_triangle(p0: np.ndarray,
                    p1: np.ndarray,
                    p2: np.ndarray,
                    n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Равномерно семплирует n точек внутри треугольника (p0, p1, p2)."""
    r1 = np.sqrt(rng.uniform(0.0, 1.0, n))
    r2 = rng.uniform(0.0, 1.0, n)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2
    return a[:, None] * p0 + b[:, None] * p1 + c[:, None] * p2


def sample_convex_polyhedron(vertices: np.ndarray,
                             n: int,
                             rng: np.random.Generator,
                             noise: float = 0.5) -> np.ndarray:
    """
    Семплирует n точек равномерно по поверхности выпуклого многогранника,
    заданного вершинами. Грани триангулируются через ConvexHull,
    количество точек на каждой грани пропорционально её площади.
    """
    hull = ConvexHull(vertices)
    tris = hull.simplices            # (M, 3) — треугольные грани

    # площади треугольников
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / areas.sum()

    counts = rng.multinomial(n, probs)
    chunks = []
    for tri, cnt in zip(tris, counts):
        if cnt == 0:
            continue
        p0, p1, p2 = vertices[tri]
        chunks.append(sample_triangle(p0, p1, p2, int(cnt), rng))

    pts = np.vstack(chunks)
    pts += rng.normal(0.0, noise, pts.shape)
    return pts


def random_convex_object(center: np.ndarray,
                         size_scale: float,
                         rng: np.random.Generator,
                         n_vertices: int = 10) -> np.ndarray:
    """
    Случайный выпуклый многогранник с произвольными углами между гранями.
    Строится как выпуклая оболочка случайных точек в шаре.
    Нижняя грань прижимается к плоскости Z = 0.
    """
    dirs = rng.normal(size=(n_vertices, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.uniform(0.5, 1.0, n_vertices) * size_scale
    verts = dirs * radii[:, None]
    verts[:, 2] -= verts[:, 2].min()      # низ на Z = 0
    verts += center
    return verts


def make_floor(x_range: tuple[float, float],
               y_range: tuple[float, float],
               n: int,
               noise: float,
               rng: np.random.Generator) -> np.ndarray:
    """Плоскость ленты с шумом."""
    return np.column_stack([
        rng.uniform(*x_range, n),
        rng.uniform(*y_range, n),
        rng.normal(0.0, noise, n),
    ])


def generate_cloud(n_objects: int | None = None,
                   seed: int = 42,
                   floor_points: int = 20_000,
                   object_points: int = 5_000,
                   x_range: tuple[float, float] = (-300.0, 300.0),
                   y_range: tuple[float, float] = (0.0, 2000.0),
                   outlier_frac: float = 0.02
                   ) -> tuple[np.ndarray, int]:
    """
    Возвращает (облако точек (N,3) float32, число сгенерированных объектов).

    Если n_objects is None — число объектов выбирается случайно:
    1 объект с вероятностью 0.7, 2 — 0.2, 3 — 0.1.
    """
    rng = np.random.default_rng(seed)

    if n_objects is None:
        n_objects = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))

    parts = [make_floor(x_range, y_range, floor_points, 0.8, rng)]

    # размещаем объекты вдоль Y без пересечений
    if n_objects == 1:
        y_centers = [0.5 * (y_range[0] + y_range[1])]
    else:
        y_centers = np.linspace(
            y_range[0] + 400.0, y_range[1] - 400.0, n_objects
        )

    for yc in y_centers:
        xc = float(rng.uniform(-150.0, 150.0))
        verts = random_convex_object(
            center=np.array([xc, float(yc), 0.0]),
            size_scale=float(rng.uniform(80.0, 180.0)),
            rng=rng,
            n_vertices=int(rng.integers(8, 14)),
        )
        pts = sample_convex_polyhedron(verts, object_points, rng, noise=0.8)
        # не вылезаем за пределы конвейера
        m = ((pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
             (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]))
        parts.append(pts[m])

    # выбросы
    n_out = int(outlier_frac * sum(p.shape[0] for p in parts))
    parts.append(rng.uniform(
        [x_range[0], y_range[0], -50.0],
        [x_range[1], y_range[1], 400.0],
        (n_out, 3),
    ))

    cloud = np.vstack(parts).astype(np.float32)
    return cloud, n_objects


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генератор облака точек для измерения габаритов."
    )
    parser.add_argument("--out", type=Path, default=Path("cloud.npz"),
                        help="Путь для .npz файла с облаком")
    parser.add_argument("--n-objects", type=int, default=None,
                        help="Число объектов (по умолчанию — случайно 1/2/3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--floor-points", type=int, default=20_000)
    parser.add_argument("--object-points", type=int, default=5_000)
    args = parser.parse_args()

    cloud, n_obj = generate_cloud(
        n_objects=args.n_objects,
        seed=args.seed,
        floor_points=args.floor_points,
        object_points=args.object_points,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        points=cloud,
        n_objects=n_obj,
        x_range=np.array([-300.0, 300.0]),
        y_range=np.array([0.0, 2000.0]),
    )
    print(f"[generate_cloud] объектов: {n_obj}, "
          f"точек: {cloud.shape[0]} -> {args.out}")


if __name__ == "__main__":
    main()