#!/usr/bin/env python3
"""Genera la reducción v1 del mismo fotograma fuente usado por la figura v0 de R01."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

HELPERS = Path(__file__).resolve().parents[1] / "auxiliares"
sys.path.insert(0, str(HELPERS))
from rosbag_depth_grid import (  # noqa: E402
    open_reader, read_grid, ros_imports, topic_types
)

V0_BAG = Path("/home/tomasdea/Tesis/bags/experimentos_tesis/derivados/R01_2_fase_inicial_v0")
V1_BAG = Path("/home/tomasdea/Tesis/bags/experimentos_tesis/derivados/R01_2_fase_inicial_v1")
OUTPUT = Path(__file__).resolve().parent / "salida" / "reduccion_v1_R01_fase_inicial.png"
V0_GRID_INDEX = 170
Z_MIN, Z_MAX = 0.25, 5.0
HAPTIC_Z_MAX = 3.0


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def cloud_array(message):
    offsets = {field.name: field.offset for field in message.fields}
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + "f4"] * 3,
        "offsets": [offsets[name] for name in ("x", "y", "z")],
        "itemsize": int(message.point_step),
    })
    data = np.frombuffer(bytes(message.data), dtype=dtype)
    points = np.column_stack((data["x"], data["y"], data["z"]))
    return points[np.isfinite(points).all(axis=1)]


def first_bag_timestamp(bag):
    reader = open_reader(bag, "sqlite3")
    if not reader.has_next():
        raise RuntimeError(f"Rosbag vacío: {bag}")
    return reader.read_next()[2]


def nearest_messages(target_relative_ns):
    _, deserialize, get_message = ros_imports()
    reader = open_reader(V1_BAG, "sqlite3")
    types = topic_types(reader)
    wanted = {
        "/depth_obstacle_filter/obstacle_cloud",
        "/perception/depth_grid",
    }
    best = {topic: None for topic in wanted}
    bag_start = first_bag_timestamp(V1_BAG)
    while reader.has_next():
        topic, raw, bag_stamp = reader.read_next()
        if topic not in wanted:
            continue
        message = deserialize(raw, get_message(types[topic]))
        delta = abs((bag_stamp - bag_start) - target_relative_ns)
        if best[topic] is None or delta < best[topic][0]:
            best[topic] = (delta, bag_stamp, message)
    if any(value is None for value in best.values()):
        raise RuntimeError("Faltan tópicos requeridos en el rosbag derivado v1.")
    return best


def main():
    v0_grid_bag_stamp, v0_grid = read_grid(
        V0_BAG, "sqlite3", "/perception/depth_grid", V0_GRID_INDEX
    )
    target_relative_ns = v0_grid_bag_stamp - first_bag_timestamp(V0_BAG)
    matched = nearest_messages(target_relative_ns)
    cloud_delta, _, cloud_msg = matched["/depth_obstacle_filter/obstacle_cloud"]
    grid_delta, _, grid = matched["/perception/depth_grid"]
    points = cloud_array(cloud_msg)
    if not len(points):
        raise RuntimeError("La nube de obstáculos sincronizada está vacía.")

    rows, cols = int(grid.rows), int(grid.cols)
    values = np.array([float(cell.min_m) for cell in grid.cells]).reshape(rows, cols)
    counts = np.array([int(cell.count) for cell in grid.cells]).reshape(rows, cols)
    radial = np.hypot(points[:, 0], points[:, 2])

    haptic_active = (counts > 0) & np.isfinite(values) & (values > 0)
    haptic = np.zeros_like(values)
    haptic[haptic_active] = (
        (HAPTIC_Z_MAX - np.clip(values[haptic_active], Z_MIN, HAPTIC_Z_MAX))
        / (HAPTIC_Z_MAX - Z_MIN) * 100.0
    )

    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    layout = fig.add_gridspec(3, 1, height_ratios=(1.15, 1.0, 1.0))
    cloud_ax = fig.add_subplot(layout[0, 0])
    stride = max(1, len(points) // 60000)
    shown = points[::stride]
    shown_radial = radial[::stride]
    azimuth_deg = np.degrees(np.arctan2(shown[:, 0], shown[:, 2]))
    cloud_plot = cloud_ax.scatter(
        azimuth_deg, shown[:, 1], c=shown_radial, cmap="turbo",
        vmin=Z_MIN, vmax=Z_MAX, s=2.8, alpha=0.82, linewidths=0,
    )
    cloud_ax.invert_yaxis()
    cloud_ax.axhline(0, color="0.25", ls="--", lw=1)
    angular_limit = max(45.0, float(np.percentile(np.abs(azimuth_deg), 99.5)))
    angular_limit = min(80.0, angular_limit)
    cloud_ax.set_xlim(-angular_limit, angular_limit)
    cloud_ax.set_xlabel("ángulo horizontal respecto del frente [grados]")
    cloud_ax.set_ylabel("Y alineado: positivo hacia abajo [m]")
    cloud_ax.set_title(
        "Reducción v1 — mismo fotograma de la configuración multirrango inicial\n"
        "Proyección angular de la nube alineada: suelo y región superior eliminados",
        weight="bold",
    )
    fig.colorbar(cloud_plot, ax=cloud_ax, label="Distancia radial [m]", shrink=0.82)

    grid_ax = fig.add_subplot(layout[1, 0])
    masked = np.ma.masked_where((counts <= 0) | ~np.isfinite(values), values)
    grid_plot = grid_ax.imshow(masked, cmap="turbo", vmin=Z_MIN, vmax=Z_MAX, aspect="auto")
    for row in range(rows):
        for col in range(cols):
            label = "--" if counts[row, col] <= 0 else f"{values[row,col]:.2f}"
            color = "white" if counts[row, col] <= 0 or values[row, col] < 2.5 else "black"
            grid_ax.text(col, row, label, ha="center", va="center", fontsize=10, color=color)
    grid_ax.set_title("DepthGrid v1 publicada: mínimo radial por sector angular [m]", weight="bold")
    grid_ax.set_xlabel("columna angular")
    grid_ax.set_ylabel("banda vertical respecto del suelo")
    grid_ax.set_xticks(range(cols))
    grid_ax.set_yticks(range(rows))
    fig.colorbar(grid_plot, ax=grid_ax, label="Distancia radial mínima [m]", shrink=0.88)

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
        "HapticGrid v1 resultante: intensidad háptica [%]", weight="bold"
    )
    haptic_ax.set_xlabel("columna angular")
    haptic_ax.set_ylabel("banda vertical respecto del suelo")
    haptic_ax.set_xticks(range(cols))
    haptic_ax.set_yticks(range(rows))
    fig.colorbar(haptic_plot, ax=haptic_ax, label="Intensidad háptica [%]", shrink=0.88)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {OUTPUT}")
    print(f"Tiempo relativo objetivo: {target_relative_ns/1e9:.3f} s")
    print(f"Nube: delta={cloud_delta/1e6:.3f} ms, puntos={len(points)}")
    print(f"Grid: delta={grid_delta/1e6:.3f} ms, celdas activas={np.count_nonzero(counts > 0)}/50")


if __name__ == "__main__":
    main()
