#!/usr/bin/env python3
"""Compara mínimo, media y máximo de una misma DepthGrid grabada en rosbag2."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from generar_reduccion_depth_grid import grid_array, read_selected_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera tres grillas comparables desde un único mensaje DepthGrid."
    )
    parser.add_argument("bag", type=Path, help="Directorio del rosbag2.")
    parser.add_argument("--grid-topic", default="/perception/depth_grid")
    parser.add_argument("--grid-index", type=int, default=30)
    parser.add_argument("--roi-row", type=int, default=2)
    parser.add_argument("--roi-col", type=int, default=5)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "salida" / "comparacion_estadisticos.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, grid = read_selected_grid(
        args.bag, args.storage_id, args.grid_topic, args.grid_index
    )
    rows, cols = int(grid.rows), int(grid.cols)
    if not (0 <= args.roi_row < rows and 0 <= args.roi_col < cols):
        raise RuntimeError(
            f"ROI ({args.roi_row}, {args.roi_col}) fuera de la grilla {rows}x{cols}."
        )

    fields = (
        ("min_m", "Mínimo por celda"),
        ("mean_m", "Media por celda"),
        ("max_m", "Máximo por celda"),
    )
    arrays = [(field, title, grid_array(grid, field)) for field, title in fields]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    last_image = None

    for axis, (_, title, values) in zip(axes, arrays):
        last_image = axis.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            vmin=args.z_min,
            vmax=args.z_max,
            interpolation="nearest",
            aspect="auto",
        )
        for row in range(rows):
            for col in range(cols):
                value = values[row, col]
                label = "--" if not math.isfinite(value) else f"{value:.2f}"
                color = (
                    "white"
                    if not math.isfinite(value)
                    or value > (args.z_min + args.z_max) / 2
                    else "black"
                )
                axis.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
        axis.add_patch(
            Rectangle(
                (args.roi_col - 0.5, args.roi_row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ff2d55",
                lw=3,
            )
        )
        selected = values[args.roi_row, args.roi_col]
        axis.set_title(f"{title}\nROI ({args.roi_row}, {args.roi_col}): {selected:.3f} m")
        axis.set_xlabel("columna")
        axis.set_ylabel("fila")
        axis.set_xticks(range(cols))
        axis.set_yticks(range(rows))

    fig.colorbar(last_image, ax=axes, label="Distancia [m]", shrink=0.86, pad=0.015)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura escrita en: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}")
