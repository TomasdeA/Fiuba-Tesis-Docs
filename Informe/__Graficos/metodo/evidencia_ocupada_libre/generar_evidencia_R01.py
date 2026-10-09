#!/usr/bin/env python3
"""Compara la evidencia actual del mapper en escenas E1-C y E1-G."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

UTILS = Path(__file__).resolve().parents[1] / "auxiliares"
sys.path.insert(0, str(UTILS))
from escenas_r01 import (  # noqa: E402
    cloud_array,
    raw_depth_at_phase,
    records_at_target,
)

OUT = Path(__file__).resolve().parent / "salida" / "evidencia_ocupada_libre_R01.png"
ANGULAR_BINS = 360
CELL = 0.10
SIZE = 100
HALF = SIZE * CELL / 2
MAX_RANGE = 5.0


def point_cell(point):
    if np.hypot(point[0], point[2]) > MAX_RANGE:
        return None
    ci = int(np.floor((point[0] + HALF) / CELL))
    cj = int(np.floor((point[2] + HALF) / CELL))
    return (ci, cj) if 0 <= ci < SIZE and 0 <= cj < SIZE else None


def cells(points):
    return {cell for point in points if (cell := point_cell(point)) is not None}


def ray(endpoint, occupied):
    """Bresenham desde el sensor, detenido antes de la primera celda ocupada."""
    x0 = z0 = SIZE // 2
    x1, z1 = endpoint
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx, sz = (1 if x0 < x1 else -1), (1 if z0 < z1 else -1)
    error = dx - dz
    result = set()
    while x0 != x1 or z0 != z1:
        cell = (x0, z0)
        if cell in occupied:
            break
        result.add(cell)
        twice = 2 * error
        if twice > -dz:
            error -= dz
            x0 += sx
        if twice < dx:
            error += dx
            z0 += sz
    return result


def xy(cell_set):
    if not cell_set:
        return np.empty((0, 2))
    array = np.asarray(sorted(cell_set))
    return np.column_stack(((array[:, 0] + .5) * CELL - HALF,
                            (array[:, 1] + .5) * CELL - HALF))


def load_scene(phase):
    messages = records_at_target(published_phase=phase, offset_s=8)
    return (
        raw_depth_at_phase(published_phase=phase, offset_s=8),
        cloud_array(messages["/depth_obstacle_filter/debug/ground_cloud"]),
        cloud_array(messages["/depth_obstacle_filter/debug/ceiling_cloud"]),
        cloud_array(messages["/depth_obstacle_filter/obstacle_cloud"]),
    )


def plot_depth(axis, depth, scene):
    shown = np.ma.masked_invalid(depth)
    image = axis.imshow(shown, cmap="turbo", vmin=.25, vmax=MAX_RANGE, aspect="auto")
    axis.set(xlabel="u [píxeles]", ylabel="v [píxeles]",
             title=f"{scene}: profundidad cruda")
    axis.set_box_aspect(depth.shape[0] / depth.shape[1])
    return image


def map_evidence(ground, obstacles):
    ground_evidence = ground[::4]
    observed = np.vstack([ground_evidence, obstacles])
    angles = np.arctan2(observed[:, 0], observed[:, 2])
    bins = np.floor((angles + np.pi) * ANGULAR_BINS / (2 * np.pi)).astype(int)
    bins = np.clip(bins, 0, ANGULAR_BINS - 1)
    distances = np.hypot(observed[:, 0], observed[:, 2])
    first = {}
    for index, distance, point in zip(bins, distances, observed):
        if distance <= MAX_RANGE and (index not in first or distance < first[index][0]):
            first[index] = (distance, point)

    occupied = cells(obstacles)
    ground_cells = cells(ground_evidence) - occupied
    swept = set()
    for _, point in first.values():
        endpoint = point_cell(point)
        if endpoint is not None:
            swept.update(ray(endpoint, occupied))
    return ground_cells, swept - ground_cells - occupied, occupied


def plot_classification(axis, ground, upper, obstacles, scene):
    series = [(ground, "#7f8c8d", "suelo"), (obstacles, "#e4572e", "obstáculo")]
    if len(upper):
        series.insert(1, (upper, "#8e7cc3", "región superior"))
    for points, color, label in series:
        sample = points[::max(1, len(points) // 7000)]
        axis.scatter(sample[:, 0], sample[:, 1], s=2, alpha=.55, c=color,
                     label=f"{label} ({len(points)})")
    axis.invert_yaxis()
    axis.axhline(0, color="k", ls="--", lw=1)
    axis.set(xlabel="X [m]", ylabel="Y alineado [m]",
             title=f"{scene}: clasificación geométrica 3D")
    axis.legend(fontsize=8)


def plot_evidence(axis, ground, obstacles, scene):
    ground_cells, ray_cells, occupied = map_evidence(ground, obstacles)
    ground_xy, ray_xy, occupied_xy = map(xy, (ground_cells, ray_cells, occupied))
    axis.scatter(ground_xy[:, 0], ground_xy[:, 1], marker="s", s=8,
                 c="#72b7a1", label="ground → libre")
    axis.scatter(ray_xy[:, 0], ray_xy[:, 1], marker="s", s=11,
                 c="#f2a541", label="aporte del raycasting")
    axis.scatter(occupied_xy[:, 0], occupied_xy[:, 1], marker="s", s=11,
                 c="#d62728", label="obstáculo → ocupado")
    axis.scatter([0], [0], marker="^", s=90, c="black", label="sensor")
    axis.set(xlabel="X [m]", ylabel="Z [m]",
             title=f"{scene}: evidencia combinada para el mapa",
             xlim=(-5.2, 5.2), ylim=(-.4, 5.4))
    axis.set_aspect("equal", "box")
    axis.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for row, (scene, phase) in enumerate((("E1-C", 4), ("E1-G", 9))):
        depth, ground, upper, obstacles = load_scene(phase)
        depth_image = plot_depth(axes[row, 0], depth, scene)
        plot_classification(axes[row, 1], ground, upper, obstacles, scene)
        plot_evidence(axes[row, 2], ground, obstacles, scene)
        fig.colorbar(depth_image, ax=axes[row, 0], label="profundidad axial [m]", shrink=.62)
    fig.suptitle("De la observación 3D a la evidencia del mapa 2D",
                 fontsize=17, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
