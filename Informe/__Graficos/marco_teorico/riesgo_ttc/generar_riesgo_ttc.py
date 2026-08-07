#!/usr/bin/env python3
"""Genera el esquema teórico propuesto de selección de riesgo y cálculo de TTC.

Los límites y umbrales se leen de la configuración vigente de
``spatial_awareness``. La pose, la velocidad y el obstáculo forman un ejemplo
geométrico declarado: no se presentan como mediciones experimentales. La
distancia longitudinal se mide desde el contorno corporal, según el cambio
previsto para ``RiskEvaluator``.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Patch
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
TESIS_ROOT = HERE.parents[4]
DEFAULT_PARAMS = (
    TESIS_ROOT
    / "develop/nav_mapper/src/spatial_awareness/config/params.yaml"
)
LOCAL_MAPPER_PARAMS = (
    TESIS_ROOT
    / "develop/nav_mapper/src/local_mapper/config/params.yaml"
)
OUTPUT = HERE / "salida/riesgo_ttc.png"


def load_parameters(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return document["spatial_awareness"]["ros__parameters"]


def load_local_mapper_parameters(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return document["local_mapper"]["ros__parameters"]


def direction(angle_deg: float) -> np.ndarray:
    """Vector XZ; el ángulo se mide desde +Z y es positivo hacia la derecha."""
    angle = math.radians(angle_deg)
    return np.array([math.sin(angle), math.cos(angle)])


def sector(ax, start_deg, end_deg, radius, color, alpha, label=None):
    angles = np.linspace(start_deg, end_deg, 120)
    boundary = [radius * direction(angle) for angle in angles]
    vertices = np.vstack(([0.0, 0.0], boundary, [0.0, 0.0]))
    ax.add_patch(Polygon(vertices, closed=True, facecolor=color,
                         edgecolor="none", alpha=alpha, label=label))


def main() -> None:
    params = load_parameters(DEFAULT_PARAMS)
    mapper_params = load_local_mapper_parameters(LOCAL_MAPPER_PARAMS)

    aperture_sensor_deg = float(params["default_aperture_deg"])
    aperture_total_deg = aperture_sensor_deg
    rear_half_deg = float(params["rear_half_angle_deg"])
    body_radius_m = float(params["body_radius_m"])
    cell_size_m = float(mapper_params["occupancy_cell_size_m"])
    corridor_half_width = body_radius_m + cell_size_m / math.sqrt(2.0)

    # Ejemplo declarado para que la proyección no sea trivial.
    speed_mps = 0.80
    velocity_angle_deg = 75.0
    obstacle_angle_deg = 60.0
    obstacle_distance_m = 0.90

    velocity_dir = direction(velocity_angle_deg)
    obstacle_dir = direction(obstacle_angle_deg)
    velocity = speed_mps * velocity_dir
    obstacle = obstacle_distance_m * obstacle_dir
    body_contact = body_radius_m * obstacle_dir
    distance_m = obstacle_distance_m - body_radius_m
    closing_speed = max(0.0, float(np.dot(velocity, obstacle_dir)))
    ttc = distance_m / closing_speed

    fig, ax = plt.subplots(figsize=(9.4, 8.0), constrained_layout=True)
    radius = 1.22

    # Regiones definidas respecto del frente corporal.
    sector(ax, -aperture_total_deg, aperture_total_deg, radius,
           "#6baed6", .20, "zona frontal excluida")
    sector(ax, aperture_total_deg, 180.0 - rear_half_deg, radius,
           "#74c476", .15, "región derecha")
    sector(ax, -180.0 + rear_half_deg, -aperture_total_deg, radius,
           "#9e9ac8", .15, "región izquierda")
    sector(ax, 180.0 - rear_half_deg, 180.0 + rear_half_deg, radius,
           "#fdae6b", .18, "región posterior")

    # Semiapertura frontal observable.
    for sign in (-1.0, 1.0):
        sensor_edge = radius * direction(sign * aperture_sensor_deg)
        ax.plot([0, sensor_edge[0]], [0, sensor_edge[1]], "-",
                color="#2171b5", lw=1.4)
    ax.text(-.31, .78, r"apertura observable $45^{\circ}$", color="#2171b5",
            rotation=43, ha="right", fontsize=9)

    # Pose y frente corporal.
    ax.add_patch(Circle((0, 0), body_radius_m, facecolor="white",
                        edgecolor="black", lw=1.5, zorder=6))
    ax.annotate("", xy=(0, .55), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
                zorder=7)
    ax.text(.04, .51, "frente observable", fontsize=9, va="bottom")
    ax.text(0, -.08, "usuario", ha="center", va="top", fontsize=9, zorder=8)

    # Corredor orientado por el movimiento.
    normal = np.array([-velocity_dir[1], velocity_dir[0]])
    length = 1.15
    corridor = np.array([
        corridor_half_width * normal,
        length * velocity_dir + corridor_half_width * normal,
        length * velocity_dir - corridor_half_width * normal,
        -corridor_half_width * normal,
    ])
    ax.add_patch(Polygon(corridor, closed=True, facecolor="#31a354",
                         edgecolor="#238b45", linestyle="--", lw=1.4,
                         alpha=.12, zorder=2))
    # Eje del corredor: prolongación de la dirección de movimiento.
    velocity_axis_end = length * velocity_dir
    ax.plot([0, velocity_axis_end[0]], [0, velocity_axis_end[1]],
            linestyle=":", color="#54278f", lw=1.5, zorder=3)
    width_origin = 1.08 * velocity_dir
    width_edge = width_origin + corridor_half_width * normal
    ax.annotate("", xy=width_edge, xytext=width_origin,
                arrowprops=dict(arrowstyle="<->", color="#238b45", lw=1.5),
                zorder=7)
    width_text = width_origin + .62 * corridor_half_width * normal
    ax.text(width_text[0]+.03, width_text[1]+.02,
            rf"$r_c={corridor_half_width:.2f}$ m".replace(".", ","),
            color="#238b45", fontsize=8.5, ha="left")

    # Distancia al obstáculo.
    ax.scatter(*obstacle, s=105, color="#e6550d", edgecolor="white",
               linewidth=1.0, zorder=8)
    ax.text(obstacle[0]+.035, obstacle[1]+.035, "obstáculo",
            color="#a63603", fontsize=9)
    # Cota d paralela a la dirección radial, desplazada para no ocultar v_c.
    obstacle_normal = np.array([-obstacle_dir[1], obstacle_dir[0]])
    dimension_offset = .095 * obstacle_normal
    dimension_start = body_contact + dimension_offset
    dimension_end = obstacle + dimension_offset
    ax.annotate("", xy=dimension_end, xytext=dimension_start,
                arrowprops=dict(arrowstyle="<->", color="#a63603", lw=1.6),
                zorder=5)
    ax.plot([body_contact[0], dimension_start[0]],
            [body_contact[1], dimension_start[1]],
            color="#a63603", lw=.9, zorder=4)
    ax.plot([obstacle[0], dimension_end[0]],
            [obstacle[1], dimension_end[1]], color="#a63603", lw=.9, zorder=4)
    midpoint = .5 * (body_contact + obstacle) + dimension_offset
    ax.text(midpoint[0], midpoint[1]+.015,
            rf"$d={distance_m:.2f}$ m".replace(".", ","),
            color="#a63603", fontsize=9, rotation=-obstacle_angle_deg+90,
            ha="center")

    # Vector velocidad y su proyección sobre la dirección del obstáculo.
    velocity_scale = .70
    velocity_tip = velocity_scale * velocity
    projected_tip = velocity_scale * closing_speed * obstacle_dir
    ax.annotate("", xy=velocity_tip, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.4, color="#54278f"),
                zorder=9)
    ax.text(velocity_tip[0]+.04, velocity_tip[1]-.08,
            r"$\mathbf{v}$: velocidad", color="#54278f", fontsize=9, ha="left")
    ax.annotate("", xy=projected_tip, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=3.0, color="#de2d26"),
                zorder=10)
    ax.plot([projected_tip[0], obstacle[0]],
            [projected_tip[1], obstacle[1]], linestyle=":",
            color="#de2d26", lw=1.6, zorder=6)
    ax.plot([velocity_tip[0], projected_tip[0]],
            [velocity_tip[1], projected_tip[1]], "--", color="#756bb1", lw=1.1)
    ax.text(.34, .255, "$v_c$", color="#de2d26", fontsize=9.5,
            ha="center")

    # Resultado numérico de la escena.
    result = (
        rf"$r_c=0{{,}}20+0{{,}}10/\sqrt{{2}}={corridor_half_width:.2f}$ m"
        "\n"
        rf"$v_c=\mathbf{{v}}\cdot\hat{{\mathbf{{r}}}}={closing_speed:.3f}$ m/s"
        "\n"
        rf"$TTC=d/v_c={ttc:.2f}$ s"
    ).replace(".", ",")
    ax.text(-1.33, -1.10, result, fontsize=10,
            bbox=dict(boxstyle="round,pad=.35", facecolor="white",
                      edgecolor="#636363"))

    ax.text(-.92, .64, "izquierda", color="#756bb1", fontsize=10,
            ha="center", weight="bold")
    ax.text(.92, .64, "derecha", color="#238b45", fontsize=10,
            ha="center", weight="bold")
    ax.text(0, -1.02, "posterior", color="#d95f0e", fontsize=10,
            ha="center", weight="bold")

    legend = [
        Patch(facecolor="#6baed6", alpha=.35,
              label="zona frontal observable (atendida por percepción frontal)"),
        Patch(facecolor="#31a354", alpha=.20,
              label="zona compatible con colisión"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8.5)
    ax.set_title("Geometría de selección de riesgo y cálculo de TTC", fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.30, 1.34)
    ax.set_xlabel("$x$ [m] — derecha")
    ax.set_ylabel("$z$ [m] — frente")
    ax.grid(alpha=.18)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico escrito en {OUTPUT}")


if __name__ == "__main__":
    main()
