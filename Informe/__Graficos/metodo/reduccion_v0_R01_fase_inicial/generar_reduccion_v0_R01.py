#!/usr/bin/env python3
"""Genera la reducción v0 de la fase lógica 1 del rosbag maestro R01."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

HELPERS = Path(__file__).resolve().parents[1] / "auxiliares"
sys.path.insert(0, str(HELPERS))
from rosbag_depth_grid import (  # noqa: E402
    decode_depth,
    read_grid,
    read_nearest_image,
    roi_bounds,
    roi_metrics,
    validate_cell,
)

BAG = Path("/home/tomasdea/Tesis/bags/experimentos_tesis/derivados/R01_2_fase_inicial_v0")
OUTPUT = Path(__file__).resolve().parent / "salida" / "reduccion_v0_R01_fase_inicial.png"
GRID_INDEX = 170
Z_MIN = 0.25
Z_MAX = 5.0
HAPTIC_Z_MAX = 3.0

CASES = {
    "near": ("A", "#d81b60"),
    "middle": ("B", "#00a676"),
    "above": ("C", "#6f42c1"),
}


def main() -> int:
    grid_ns, grid = read_grid(BAG, "sqlite3", "/perception/depth_grid", GRID_INDEX)
    delta_ns, _, image = read_nearest_image(
        BAG, "sqlite3", "/camera/camera/depth/image_rect_raw", grid_ns
    )
    depth = decode_depth(image)
    rows, cols = int(grid.rows), int(grid.cols)
    height, width = depth.shape
    x_edges = np.arange(cols + 1, dtype=int) * width // cols
    y_edges = np.arange(rows + 1, dtype=int) * height // rows
    values = np.array([float(cell.min_m) for cell in grid.cells]).reshape(rows, cols)
    counts = np.array([int(cell.count) for cell in grid.cells]).reshape(rows, cols)

    candidates = []
    for row in range(rows):
        for col in range(cols):
            x0, x1, y0, y1 = roi_bounds(row, col, x_edges, y_edges)
            metrics = roi_metrics(depth[y0:y1, x0:x1], Z_MIN, Z_MAX)
            candidates.append((row, col, metrics))

    usable = [item for item in candidates if item[2]["valid"] > 0]
    near = min(usable, key=lambda item: item[2]["min"])
    remaining = [item for item in usable if item[:2] != near[:2]]
    middle = min(remaining, key=lambda item: abs(item[2]["min"] - 2.0))
    remaining = [item for item in candidates if item[:2] not in {near[:2], middle[:2]}]
    above = max(remaining, key=lambda item: item[2]["above"])
    if above[2]["above"] == 0:
        raise RuntimeError("El fotograma seleccionado no contiene muestras por encima de z_max.")
    selected = {"near": near, "middle": middle, "above": above}

    for _, (row, col, metrics) in selected.items():
        ok, _ = validate_cell(grid, row, col, metrics)
        if not ok:
            raise RuntimeError(f"La ROI ({row},{col}) no coincide con la DepthGrid publicada.")

    display_depth = np.ma.masked_where((~np.isfinite(depth)) | (depth <= 0), depth)
    haptic_active = (counts > 0) & np.isfinite(values) & (values > 0)
    haptic = np.zeros_like(values)
    haptic[haptic_active] = (
        (HAPTIC_Z_MAX - np.clip(values[haptic_active], Z_MIN, HAPTIC_Z_MAX))
        / (HAPTIC_Z_MAX - Z_MIN) * 100.0
    )

    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    layout = fig.add_gridspec(3, 1, height_ratios=(1.15, 1.0, 1.0))
    image_ax = fig.add_subplot(layout[0, 0])
    image_plot = image_ax.imshow(
        display_depth, cmap="turbo", vmin=0, vmax=Z_MAX * 1.1, aspect="auto"
    )
    for edge in x_edges:
        image_ax.axvline(edge - 0.5, color="white", lw=0.65, alpha=0.8)
    for edge in y_edges:
        image_ax.axhline(edge - 0.5, color="white", lw=0.65, alpha=0.8)

    for case, (row, col, _) in selected.items():
        label, color = CASES[case]
        x0, x1, y0, y1 = roi_bounds(row, col, x_edges, y_edges)
        image_ax.add_patch(Rectangle(
            (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, lw=3,
        ))
        image_ax.text(
            x0 + 5, y0 + 22, label, color="white", weight="bold", fontsize=13,
            bbox=dict(facecolor=color, edgecolor="none", pad=3),
        )
    image_ax.set_title(
        f"Profundidad {width}×{height}: partición en {rows}×{cols} regiones",
        weight="bold",
    )
    image_ax.set_xlabel("u [píxeles]")
    image_ax.set_ylabel("v [píxeles]")
    fig.colorbar(image_plot, ax=image_ax, label="Profundidad medida [m]", shrink=0.82)

    grid_ax = fig.add_subplot(layout[1, 0])
    masked = np.ma.masked_where((counts <= 0) | ~np.isfinite(values), values)
    grid_plot = grid_ax.imshow(masked, cmap="turbo", vmin=Z_MIN, vmax=Z_MAX, aspect="auto")
    for row in range(rows):
        for col in range(cols):
            label = "--" if counts[row, col] <= 0 else f"{values[row, col]:.2f}"
            color = "white" if counts[row, col] <= 0 or values[row, col] < 2.5 else "black"
            grid_ax.text(col, row, label, ha="center", va="center", fontsize=10, color=color)
    for case, (row, col, _) in selected.items():
        label, color = CASES[case]
        grid_ax.add_patch(Rectangle(
            (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color, lw=2.8
        ))
        grid_ax.text(
            col - 0.37, row - 0.28, label, ha="center", va="center",
            color="white", weight="bold", fontsize=9,
            bbox=dict(facecolor=color, edgecolor="none", pad=2),
        )
    grid_ax.set_title("DepthGrid v0 publicada: mínimo válido por región [m]", weight="bold")
    grid_ax.set_xlabel("columna de salida")
    grid_ax.set_ylabel("fila de salida")
    grid_ax.set_xticks(range(cols))
    grid_ax.set_yticks(range(rows))
    fig.colorbar(grid_plot, ax=grid_ax, label="Distancia mínima válida [m]", shrink=0.88)

    haptic_ax = fig.add_subplot(layout[2, 0])
    haptic_cmap = plt.get_cmap("magma_r").copy()
    haptic_plot = haptic_ax.imshow(
        haptic, cmap=haptic_cmap, vmin=0, vmax=100, aspect="auto"
    )
    for row in range(rows):
        for col in range(cols):
            label = f"{haptic[row,col]:.0f}"
            color = "white" if haptic[row,col] >= 55 else "black"
            haptic_ax.text(col, row, label, ha="center", va="center", fontsize=9, color=color)
    haptic_ax.set_title(
        "HapticGrid v0 resultante: intensidad háptica [%]", weight="bold"
    )
    haptic_ax.set_xlabel("columna de salida")
    haptic_ax.set_ylabel("fila de salida")
    haptic_ax.set_xticks(range(cols))
    haptic_ax.set_yticks(range(rows))
    fig.colorbar(haptic_plot, ax=haptic_ax, label="Intensidad háptica [%]", shrink=0.88)

    near_value = near[2]["min"]
    middle_value = middle[2]["min"]
    image_ax.legend(
        handles=[
            Patch(facecolor=CASES["near"][1],
                  label=f"A: válida más próxima a zmin ({near_value:.3f} m)"),
            Patch(facecolor=CASES["middle"][1],
                  label=f"B: medición intermedia ({middle_value:.2f} m)"),
            Patch(facecolor=CASES["above"][1],
                  label=f"C: región con muestras z > zmax ({Z_MAX:g} m)"),
        ],
        loc="upper left", ncol=3, frameon=True, fontsize=8.5,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada en: {OUTPUT}")
    for case, (row, col, metrics) in selected.items():
        print(
            f"{case}: celda ({row},{col}), publicado={values[row,col]:.3f} m, "
            f"valid={metrics['valid']}, above={metrics['above']}"
        )
    positive = depth[np.isfinite(depth) & (depth > 0)]
    print(f"Mínimo positivo del fotograma: {positive.min():.3f} m")
    print(f"Muestras positivas bajo z_min: {np.count_nonzero(positive < Z_MIN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
