#!/usr/bin/env python3
"""Visualiza y verifica la codificación angular de obstacle_grid_encoder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


def ros_imports():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except ImportError as exc:
        raise RuntimeError(
            "Faltan bibliotecas de ROS 2. Cargue /opt/ros/humble/setup.bash y el workspace."
        ) from exc
    return rosbag2_py, deserialize_message, get_message, point_cloud2


def open_reader(bag: Path, storage_id: str):
    rosbag2_py, _, _, _ = ros_imports()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def decode_depth(message) -> np.ndarray:
    encoding = message.encoding.upper()
    if encoding in {"16UC1", "MONO16"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = 1e-3
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise RuntimeError(f"Codificación de profundidad no soportada: {message.encoding!r}.")
    elements_per_row = int(message.step) // dtype.itemsize
    raw = np.frombuffer(bytes(message.data), dtype=dtype)
    expected = int(message.height) * elements_per_row
    if raw.size < expected:
        raise RuntimeError("La imagen de profundidad contiene menos datos que los declarados.")
    image = raw[:expected].reshape(int(message.height), elements_per_row)
    return image[:, : int(message.width)].astype(np.float32) * scale


def read_sample(args):
    _, deserialize_message, get_message, point_cloud2 = ros_imports()
    reader = open_reader(args.bag, args.storage_id)
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    required = (args.cloud_topic, args.grid_topic, args.depth_topic)
    for topic in required:
        if topic not in types:
            raise RuntimeError(f"El rosbag no contiene {topic!r}.")

    cloud_type = get_message(types[args.cloud_topic])
    grid_type = get_message(types[args.grid_topic])
    depth_type = get_message(types[args.depth_topic])
    scalar_types = {
        topic: get_message(types[topic])
        for topic in (args.aperture_topic, args.height_topic)
        if topic in types
    }
    cloud = None
    cloud_index = 0
    grids = {}
    depth_images = {}
    aperture = None
    height = None
    selected_aperture = None
    selected_height = None

    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == args.cloud_topic:
            if cloud_index == args.cloud_index:
                cloud = deserialize_message(raw, cloud_type)
                selected_aperture = aperture
                selected_height = height
            cloud_index += 1
        elif topic == args.grid_topic:
            message = deserialize_message(raw, grid_type)
            grids.setdefault(stamp_ns(message), message)
        elif topic == args.depth_topic:
            message = deserialize_message(raw, depth_type)
            depth_images.setdefault(stamp_ns(message), message)
        elif topic == args.aperture_topic:
            aperture = float(deserialize_message(raw, scalar_types[topic]).data)
        elif topic == args.height_topic:
            height = float(deserialize_message(raw, scalar_types[topic]).data)

    if cloud is None:
        raise RuntimeError(
            f"Solo hay {cloud_index} nubes; no existe --cloud-index={args.cloud_index}."
        )
    cloud_stamp = stamp_ns(cloud)
    if cloud_stamp not in grids:
        raise RuntimeError(
            "No existe una DepthGrid con el mismo header.stamp que la nube seleccionada."
        )
    if cloud_stamp not in depth_images:
        raise RuntimeError(
            "No existe una imagen de profundidad con el mismo header.stamp que la nube "
            "seleccionada."
        )
    points = point_cloud2.read_points_numpy(
        cloud, field_names=["x", "y", "z"], skip_nans=False
    ).astype(np.float64, copy=False)
    return (
        points,
        grids[cloud_stamp],
        decode_depth(depth_images[cloud_stamp]),
        selected_aperture,
        selected_height,
    )


def bin_index(values: np.ndarray, low: float, high: float, bins: int) -> np.ndarray:
    """Replica binIndex() de C++ incluido su truncamiento hacia cero."""
    indices = ((values - low) / (high - low) * bins).astype(np.int64)
    indices[(indices < 0) | (indices >= bins)] = -1
    return indices


def reproduce(points: np.ndarray, grid, args, aperture_deg: float, y_max: float):
    rows, cols = int(grid.rows), int(grid.cols)
    finite = np.all(np.isfinite(points), axis=1)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    base = finite & (z >= args.z_min) & (z <= args.z_max)
    base &= (y >= args.y_min) & (y < y_max)

    row = np.full(points.shape[0], -1, dtype=np.int64)
    row[base] = ((y[base] - args.y_min) / (y_max - args.y_min) * rows).astype(np.int64)
    base &= (row >= 0) & (row < rows)

    angle = np.arctan2(x, z)
    sensor_half = math.radians(args.sensor_aperture_deg)
    high_cols = max(cols, math.ceil(2.0 * args.sensor_aperture_deg / args.resolution_deg))
    high_col = bin_index(angle, -sensor_half, sensor_half, high_cols)
    admitted_sensor = base & (high_col >= 0)

    high_step = 2.0 * sensor_half / high_cols
    high_center = -sensor_half + (high_col + 0.5) * high_step
    active_half = math.radians(aperture_deg)
    col = bin_index(high_center, -active_half, active_half, cols)
    kept = admitted_sensor & (col >= 0)
    distance = np.hypot(x, z)

    count = np.zeros((rows, cols), dtype=np.int64)
    minimum = np.full((rows, cols), np.nan)
    maximum = np.full((rows, cols), np.nan)
    mean = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            values = distance[kept & (row == r) & (col == c)]
            if values.size:
                count[r, c] = values.size
                minimum[r, c] = np.min(values)
                maximum[r, c] = np.max(values)
                mean[r, c] = np.mean(values)

    published_count = np.array([cell.count for cell in grid.cells]).reshape(rows, cols)
    published_min = np.array([cell.min_m for cell in grid.cells]).reshape(rows, cols)
    published_mean = np.array([cell.mean_m for cell in grid.cells]).reshape(rows, cols)
    published_max = np.array([cell.max_m for cell in grid.cells]).reshape(rows, cols)
    same = np.array_equal(count, published_count)
    same &= np.allclose(minimum, published_min, atol=2e-3, equal_nan=True)
    same &= np.allclose(mean, published_mean, atol=2e-3, equal_nan=True)
    same &= np.allclose(maximum, published_max, atol=2e-3, equal_nan=True)
    return {
        "row": row,
        "col": col,
        "angle": angle,
        "admitted_sensor": admitted_sensor,
        "kept": kept,
        "minimum": minimum,
        "count": count,
        "verified": bool(same),
        "high_cols": high_cols,
    }


def subsample(mask: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= maximum:
        return indices
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def plot(points, grid, depth, result, args, aperture_deg: float, y_max: float):
    rows, cols = int(grid.rows), int(grid.cols)
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), width_ratios=(1.0, 1.25))
    depth_ax = fig.add_subplot(layout[0, 0])
    top = fig.add_subplot(layout[0, 1])
    side = fig.add_subplot(layout[1, 0])
    matrix = fig.add_subplot(layout[1, 1])

    column_cmap = ListedColormap(plt.get_cmap("tab10").colors[:cols])
    column_norm = BoundaryNorm(np.arange(-0.5, cols + 0.5), cols)
    row_cmap = ListedColormap(plt.get_cmap("Set1").colors[:rows])
    row_norm = BoundaryNorm(np.arange(-0.5, rows + 0.5), rows)

    depth_cmap_input = plt.get_cmap("turbo").copy()
    depth_cmap_input.set_bad("black")
    valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth <= args.z_max)
    depth_image = depth_ax.imshow(
        np.ma.masked_where(~valid_depth, depth), cmap=depth_cmap_input,
        vmin=args.z_min, vmax=args.z_max, interpolation="nearest", aspect="auto",
    )
    depth_ax.set_title("Entrada: imagen de profundidad")
    depth_ax.set_xlabel("u [píxeles]")
    depth_ax.set_ylabel("v [píxeles]")
    fig.colorbar(depth_image, ax=depth_ax, label="Profundidad [m]", pad=0.01)

    outside = result["admitted_sensor"] & ~result["kept"]
    outside_indices = subsample(outside, args.max_points // 5, rng)
    kept_indices = subsample(result["kept"], args.max_points, rng)
    if outside_indices.size:
        top.scatter(
            points[outside_indices, 0], points[outside_indices, 2],
            s=1, c="0.78", alpha=0.35, rasterized=True,
            label="fuera de la apertura activa",
        )
    scatter_top = top.scatter(
        points[kept_indices, 0], points[kept_indices, 2],
        s=1.2, c=result["col"][kept_indices], cmap=column_cmap,
        norm=column_norm, alpha=0.65, rasterized=True,
    )
    max_range = args.z_max
    for angle_deg in np.linspace(-aperture_deg, aperture_deg, cols + 1):
        angle = math.radians(angle_deg)
        top.plot(
            [0, max_range * math.sin(angle)], [0, max_range * math.cos(angle)],
            color="black", lw=0.65, alpha=0.8,
        )
    for angle_deg in (-args.sensor_aperture_deg, args.sensor_aperture_deg):
        angle = math.radians(angle_deg)
        top.plot(
            [0, max_range * math.sin(angle)], [0, max_range * math.cos(angle)],
            color="0.35", lw=1.2, ls="--",
        )
    output_step_deg = 2.0 * aperture_deg / cols
    effective_low_deg = -aperture_deg - output_step_deg
    effective_low = math.radians(effective_low_deg)
    top.plot(
        [0, max_range * math.sin(effective_low)],
        [0, max_range * math.cos(effective_low)],
        color="crimson", lw=1.4, ls=":",
        label=f"límite inferior efectivo ({effective_low_deg:g}°)",
    )
    top.plot(0, 0, marker="^", color="black", ms=7)
    top.set_title(
        f"Vista superior: {result['high_cols']} sectores internos de {args.resolution_deg:g}° "
        f"y reducción a {cols} columnas (apertura nominal ±{aperture_deg:g}°)"
    )
    top.set_xlabel("x [m]")
    top.set_ylabel("z [m]")
    top.set_aspect("equal", adjustable="box")
    top.set_xlim(-args.z_max, args.z_max)
    top.set_ylim(0, args.z_max)
    top.grid(alpha=0.2)
    top.legend(loc="upper right", fontsize=8)
    cbar_top = fig.colorbar(scatter_top, ax=top, ticks=range(cols), pad=0.01)
    cbar_top.set_label("columna asignada")

    scatter_side = side.scatter(
        points[kept_indices, 2], points[kept_indices, 1],
        s=1.2, c=result["row"][kept_indices], cmap=row_cmap,
        norm=row_norm, alpha=0.65, rasterized=True,
    )
    y_edges = np.linspace(args.y_min, y_max, rows + 1)
    for edge in y_edges:
        side.axhline(edge, color="black", lw=0.8)
    for r in range(rows):
        side.text(args.z_max * 0.98, (y_edges[r] + y_edges[r + 1]) / 2,
                  f"fila {r}", ha="right", va="center", fontsize=8,
                  bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    side.set_title(f"Vista lateral: 5 bandas métricas, y=[{args.y_min:.2f}, {y_max:.2f}) m")
    side.set_xlabel("z [m]")
    side.set_ylabel("y [m]")
    side.set_xlim(0, args.z_max)
    side.set_ylim(y_max, args.y_min)
    side.grid(alpha=0.2)
    cbar_side = fig.colorbar(scatter_side, ax=side, ticks=range(rows), pad=0.01)
    cbar_side.set_label("fila asignada")

    depth_cmap = plt.get_cmap("turbo").copy()
    depth_cmap.set_bad("black")
    finite_minima = result["minimum"][np.isfinite(result["minimum"])]
    matrix_max = max(args.z_max, float(np.max(finite_minima))) if finite_minima.size else args.z_max
    image = matrix.imshow(
        np.ma.masked_invalid(result["minimum"]), cmap=depth_cmap,
        vmin=args.z_min, vmax=matrix_max, aspect="auto", interpolation="nearest",
    )
    for r in range(rows):
        for c in range(cols):
            value = result["minimum"][r, c]
            label = "--" if not math.isfinite(value) else f"{value:.2f}"
            color = "white" if not math.isfinite(value) or value > 2.7 else "black"
            matrix.text(c, r, label, ha="center", va="center", fontsize=8, color=color)
    matrix.set_title("DepthGrid resultante: distancia mínima [m]")
    matrix.set_xlabel("columna")
    matrix.set_ylabel("fila")
    matrix.set_xticks(range(cols))
    matrix.set_yticks(range(rows))
    fig.colorbar(image, ax=matrix, label="Distancia [m]", pad=0.01)

    status = "Verificación celda por celda: coincide con el mensaje publicado"
    if not result["verified"]:
        status = "ADVERTENCIA: la reproducción no coincide con el mensaje publicado"
    fig.text(0.5, 0.005, status, ha="center", fontsize=10,
             color="black" if result["verified"] else "red")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--cloud-topic", default="/depth_obstacle_filter/obstacle_cloud")
    parser.add_argument("--grid-topic", default="/perception/depth_grid")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--aperture-topic", default="/perception/depth_grid/aperture_state_deg")
    parser.add_argument("--height-topic", default="/depth_obstacle_filter/camera_height")
    parser.add_argument("--cloud-index", type=int, default=30)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--sensor-aperture-deg", type=float, default=70.0)
    parser.add_argument("--active-aperture-deg", type=float, default=45.0)
    parser.add_argument("--resolution-deg", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-0.20)
    parser.add_argument("--y-default-max", type=float, default=2.20)
    parser.add_argument("--y-margin", type=float, default=0.10)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "salida" / "codificacion_angular.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    points, grid, depth, aperture, height = read_sample(args)
    aperture_deg = args.active_aperture_deg if aperture is None else aperture
    y_max = args.y_default_max if height is None else height + args.y_margin
    result = reproduce(points, grid, args, aperture_deg, y_max)
    plot(points, grid, depth, result, args, aperture_deg, y_max)
    print(f"Figura escrita en: {args.output}")
    print(f"Puntos de entrada: {points.shape[0]}; puntos asignados: {np.count_nonzero(result['kept'])}")
    print(f"Apertura activa: ±{aperture_deg:.1f}°; y=[{args.y_min:.2f}, {y_max:.2f}) m")
    print(f"Verificación con DepthGrid: {'OK' if result['verified'] else 'FALLÓ'}")
    if not result["verified"]:
        published_count = np.array([cell.count for cell in grid.cells]).reshape(grid.rows, grid.cols)
        print(
            f"Conteos reproducido/publicado: {int(result['count'].sum())}/"
            f"{int(published_count.sum())}"
        )
    return 0 if result["verified"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
