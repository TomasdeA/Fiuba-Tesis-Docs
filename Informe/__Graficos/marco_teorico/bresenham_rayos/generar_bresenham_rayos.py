#!/usr/bin/env python3
"""Genera el esquema de recorrido discreto de rayos con Bresenham."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent / "salida/bresenham_rayos.png"


def discrete_ray(x0, y0, x1, y1):
    cells = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx-dy
    while x0 != x1 or y0 != y1:
        cells.append((x0, y0))
        e2 = 2*err
        if e2 > -dy: err, x0 = err-dy, x0+sx
        if e2 < dx: err, y0 = err+dx, y0+sy
    return cells


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    origin = (2, 2)
    cases = [(True, (11, 7), "Medición con obstáculo"),
             (False, (14, 8), "Medición sin obstáculo dentro del rango")]
    for ax, (occupied, endpoint, title) in zip(axes, cases):
        intermediate = discrete_ray(*origin, *endpoint)
        free_cells = intermediate if occupied else intermediate + [endpoint]
        ax.set_xticks(np.arange(-.5, 15, 1), minor=True); ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which="minor", color="#b8b8b8", linewidth=.55)
        for x, y in free_cells:
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor="#a8d8a8", edgecolor="none", alpha=.9, zorder=1))
        if occupied:
            ax.add_patch(Rectangle((endpoint[0]-.5, endpoint[1]-.5), 1, 1, facecolor="#e45756", edgecolor="none", alpha=.95, zorder=1))
        route = intermediate + [endpoint]
        route_x = [point[0] for point in route]; route_y = [point[1] for point in route]
        ax.plot(route_x, route_y, color="#2457a6", lw=2.3, marker="o", markersize=3.2, zorder=3)
        ax.annotate("", xy=route[-1], xytext=route[-2], arrowprops=dict(arrowstyle="->", color="#2457a6", lw=2.3), zorder=4)
        ax.scatter(*origin, s=75, color="#2457a6", edgecolor="white", linewidth=.8, zorder=4)
        ax.text(origin[0]-.15, origin[1]-.75, "celda de la cámara", ha="center", fontsize=9)
        ax.text(endpoint[0]+.15, endpoint[1]+.55, "celda ocupada" if occupied else "límite del rango", ha="center", fontsize=9)
        ax.set_title(title); ax.set(aspect="equal", xlim=(-.5, 14.5), ylim=(-.5, 9.5)); ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(handles=[Patch(facecolor="#a8d8a8", label="evidencia libre"), Patch(facecolor="#e45756", label="evidencia ocupada")], loc="lower right", framealpha=.95, fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
