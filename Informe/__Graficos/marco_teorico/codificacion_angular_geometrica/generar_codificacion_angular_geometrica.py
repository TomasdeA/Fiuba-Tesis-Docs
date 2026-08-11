#!/usr/bin/env python3
"""Genera un esquema teórico de la codificación angular y vertical.

La figura no usa datos experimentales. Muestra cómo un punto 3D se asigna a
una columna por su ángulo lateral, a una fila por su altura y cómo cada celda
conserva la distancia radial mínima.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Wedge
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "salida/codificacion_angular_geometrica.png"

N_COLS = 10
N_ROWS = 5
APERTURE_DEG = 90.0
Y_MIN = -0.20
Y_MAX = 1.50

# Único punto usado en los tres paneles para preservar continuidad visual.
POINT_X = 0.92
POINT_Y = 0.63
POINT_Z = 2.05

BLUE = "#2457a6"
ORANGE = "#e68a00"
GREEN = "#287a59"
GRID = "#aeb9ca"
LIGHT_BLUE = "#dbe9ff"


def direction(angle_deg: float, radius: float = 1.0) -> np.ndarray:
    """Vector XZ; cero grados apunta hacia +Z."""
    angle = math.radians(angle_deg)
    return radius * np.array([math.sin(angle), math.cos(angle)])


def draw_top_view(ax: plt.Axes) -> None:
    """Vista superior: apertura, columnas, beta y distancia radial."""
    radius = 2.8
    half_aperture = APERTURE_DEG / 2.0
    edges = np.linspace(-half_aperture, half_aperture, N_COLS + 1)

    # Sectores alternados para distinguir las columnas sin sobrecargar.
    for index in range(N_COLS):
        theta1 = 90.0 - edges[index + 1]
        theta2 = 90.0 - edges[index]
        color = LIGHT_BLUE if index % 2 == 0 else "#eef4ff"
        ax.add_patch(Wedge((0, 0), radius, theta1, theta2,
                           facecolor=color, edgecolor="none", alpha=0.85))

    for angle in edges:
        endpoint = direction(angle, radius)
        linewidth = 1.5 if angle in (edges[0], edges[-1]) else 0.65
        ax.plot([0, endpoint[0]], [0, endpoint[1]], color=BLUE,
                linewidth=linewidth, alpha=0.9)

    # Punto de ejemplo dentro de una columna.
    point = np.array([POINT_X, POINT_Z])
    beta_deg = math.degrees(math.atan2(point[0], point[1]))
    distance = float(np.linalg.norm(point))
    ax.plot([0, point[0]], [0, point[1]], color=ORANGE, linewidth=2.1)
    ax.scatter(*point, s=65, color=ORANGE, edgecolor="white",
               linewidth=0.8, zorder=5)
    ax.text(point[0] + 0.08, point[1] + 0.04, r"$P=(x,z)$", fontsize=9)

    # Arco de beta medido desde +Z hacia el punto.
    arc_radius = 0.72
    ax.add_patch(Arc((0, 0), 2 * arc_radius, 2 * arc_radius,
                     theta1=90.0 - beta_deg, theta2=90.0,
                     color=ORANGE, linewidth=1.6))
    beta_label = direction(beta_deg / 2.0, 0.89)
    ax.text(beta_label[0] + 0.02, beta_label[1], r"$\beta$",
            color=ORANGE, fontsize=10, ha="left", weight="bold")
    midpoint = 0.57 * point
    ax.text(midpoint[0] + 0.05, midpoint[1] - 0.04, r"$d$",
            color=ORANGE, fontsize=10, ha="left", va="center", weight="bold")

    equations = (
        rf"$\beta=\operatorname{{atan2}}(x,z)={beta_deg:.1f}^\circ$"
        "\n"
        rf"$d=\sqrt{{x^2+z^2}}={distance:.2f}\,\mathrm{{m}}$"
    ).replace(".", ",")
    ax.text(-2.02, 2.60, equations, fontsize=8.5, color=ORANGE,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=.35", facecolor="white",
                      edgecolor=ORANGE, alpha=.96), zorder=10)

    ax.scatter(0, 0, marker="^", s=95, color="black", zorder=6)
    ax.text(0, -0.19, "cámara", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(0, 1.15), xytext=(0, 0.20),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
    ax.text(0.08, 0.72, "+z", fontsize=8)
    ax.annotate("", xy=(1.15, 0), xytext=(0.20, 0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
    ax.text(1.17, 0, "+x", fontsize=8, va="center")

    ax.text(0, radius + 0.13, "10 columnas angulares",
            color=BLUE, fontsize=9, ha="center", weight="bold")
    ax.set_title("(a) Asignación horizontal", fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-0.35, 3.12)
    ax.axis("off")


def draw_side_view(ax: plt.Axes) -> None:
    """Vista lateral: bandas métricas y asignación de fila."""
    z_max = 3.0
    row_edges = np.linspace(Y_MIN, Y_MAX, N_ROWS + 1)
    row_colors = ["#eef4ff", "#dbe9ff"]

    for row in range(N_ROWS):
        lower = row_edges[row]
        upper = row_edges[row + 1]
        ax.add_patch(Rectangle((0, lower), z_max, upper - lower,
                               facecolor=row_colors[row % 2],
                               edgecolor=GRID, linewidth=0.8))
        ax.text(z_max + 0.08, (lower + upper) / 2.0,
                f"fila {row}", fontsize=8.5, va="center", color=BLUE)

    # Punto de ejemplo ubicado en la banda vertical 2.
    point_z = POINT_Z
    point_y = POINT_Y
    row = int((point_y - Y_MIN) / (Y_MAX - Y_MIN) * N_ROWS)
    ax.scatter(point_z, point_y, s=65, color=ORANGE,
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.plot([0, point_z], [0, point_y], color=ORANGE, linewidth=1.8)
    ax.text(point_z + 0.07, point_y + 0.06,
            rf"$P\rightarrow$ fila {row}", color=ORANGE, fontsize=8.5)

    ax.scatter(0, 0, marker=">", s=95, color="black", zorder=6)
    ax.text(-0.06, -0.11, "cámara", ha="left", va="top", fontsize=9)
    ax.annotate("", xy=(0, 1.25), xytext=(0, 0.18),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
    ax.text(-0.08, 1.27, "+y", fontsize=8, ha="right")
    ax.annotate("", xy=(1.25, 0), xytext=(0.18, 0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
    ax.text(1.28, -0.04, "+z", fontsize=8, va="top")

    ax.text(-0.05, Y_MIN, r"$y_{\min}$", fontsize=8.5, ha="right", va="center")
    ax.text(-0.05, Y_MAX, r"$y_{\max}$", fontsize=8.5, ha="right", va="center")
    ax.set_title("(b) Asignación vertical", fontsize=11, weight="bold")
    ax.set_xlim(-0.34, 3.62)
    # En la convención óptica empleada por el pipeline, +y apunta hacia abajo.
    # Invertir los límites coloca y_min/fila 0 arriba y y_max/fila 4 abajo.
    ax.set_ylim(Y_MAX + 0.22, Y_MIN - 0.22)
    ax.axis("off")


def draw_output_grid(ax: plt.Axes) -> None:
    """Grilla final: ejemplo de reducción por distancia radial mínima."""
    values = np.full((N_ROWS, N_COLS), np.nan)
    distance = math.hypot(POINT_X, POINT_Z)
    beta_deg = math.degrees(math.atan2(POINT_X, POINT_Z))
    selected_row = int((POINT_Y - Y_MIN) / (Y_MAX - Y_MIN) * N_ROWS)
    selected_col = int((beta_deg + APERTURE_DEG / 2.0) / APERTURE_DEG * N_COLS)
    selected_row = min(max(selected_row, 0), N_ROWS - 1)
    selected_col = min(max(selected_col, 0), N_COLS - 1)
    values[selected_row, selected_col] = distance

    ax.imshow(values, cmap="YlOrRd_r", vmin=0.5, vmax=3.0,
              origin="lower", alpha=0.72)
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="#555555", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticklabels([str(index) for index in range(N_COLS)], fontsize=7.5)
    ax.set_yticklabels([str(index) for index in range(N_ROWS)], fontsize=8)
    ax.set_xlabel("columna angular", fontsize=9)
    ax.set_ylabel("fila vertical", fontsize=9)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            if np.isfinite(values[row, col]):
                label = f"{values[row, col]:.2f}".replace(".", ",")
                ax.text(col, row, label, ha="center", va="center",
                        fontsize=7.5, weight="bold")

    # Única celda del ejemplo, correspondiente al mismo punto de (a) y (b).
    ax.add_patch(Rectangle((selected_col - 0.5, selected_row - 0.5), 1, 1,
                           fill=False, edgecolor=GREEN, linewidth=2.3))
    ax.set_title("(c) Grilla resultante 5 × 10", fontsize=11, weight="bold")
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(-0.5, N_ROWS - 0.5)


def main() -> None:
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.2, 4.35),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.15]},
        constrained_layout=True,
    )

    draw_top_view(axes[0])
    draw_side_view(axes[1])
    draw_output_grid(axes[2])

    fig.suptitle(
        "Codificación angular y vertical de una nube de obstáculos",
        fontsize=13,
        weight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gráfico escrito en {OUTPUT}")


if __name__ == "__main__":
    main()
