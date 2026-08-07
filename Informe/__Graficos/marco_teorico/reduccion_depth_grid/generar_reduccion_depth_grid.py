#!/usr/bin/env python3
"""Genera una figura reproducible de Image 16UC1 -> ROIs -> DepthGrid.

Lee una imagen de profundidad y la DepthGrid temporalmente más cercana desde un
rosbag2. No modifica mensajes ni sintetiza mediciones.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def ros_imports():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "No se encontraron las bibliotecas de ROS 2. Ejecute primero: "
            "source /opt/ros/humble/setup.bash && "
            "source /home/tomasdea/Tesis/install/setup.bash"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def open_reader(bag: Path, storage_id: str):
    rosbag2_py, _, _ = ros_imports()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def topic_types(reader) -> dict[str, str]:
    return {entry.name: entry.type for entry in reader.get_all_topics_and_types()}


def read_selected_grid(
    bag: Path, storage_id: str, topic: str, index: int
):
    _, deserialize_message, get_message = ros_imports()
    reader = open_reader(bag, storage_id)
    types = topic_types(reader)
    if topic not in types:
        raise RuntimeError(f"El rosbag no contiene el tópico {topic!r}.")
    if types[topic] != "custom_interfaces/msg/DepthGrid":
        raise RuntimeError(
            f"{topic!r} tiene tipo {types[topic]!r}, no custom_interfaces/msg/DepthGrid."
        )
    message_type = get_message(types[topic])
    current = 0
    while reader.has_next():
        read_topic, raw, timestamp = reader.read_next()
        if read_topic != topic:
            continue
        if current == index:
            return timestamp, deserialize_message(raw, message_type)
        current += 1
    raise RuntimeError(
        f"El tópico {topic!r} contiene {current} mensajes; no existe el índice {index}."
    )


def read_nearest_image(
    bag: Path, storage_id: str, topic: str, target_timestamp: int
):
    _, deserialize_message, get_message = ros_imports()
    reader = open_reader(bag, storage_id)
    types = topic_types(reader)
    if topic not in types:
        raise RuntimeError(f"El rosbag no contiene el tópico {topic!r}.")
    if types[topic] != "sensor_msgs/msg/Image":
        raise RuntimeError(
            f"{topic!r} tiene tipo {types[topic]!r}, no sensor_msgs/msg/Image."
        )
    best = None
    while reader.has_next():
        read_topic, raw, timestamp = reader.read_next()
        if read_topic != topic:
            continue
        distance = abs(timestamp - target_timestamp)
        if best is None or distance < best[0]:
            best = (distance, timestamp, bytes(raw))
    if best is None:
        raise RuntimeError(f"El tópico {topic!r} no contiene mensajes.")
    distance, timestamp, raw = best
    return distance, timestamp, deserialize_message(raw, get_message(types[topic]))


def decode_depth(message) -> np.ndarray:
    encoding = message.encoding.upper()
    if encoding in {"16UC1", "MONO16"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        elements_per_row = message.step // dtype.itemsize
        raw = np.frombuffer(bytes(message.data), dtype=dtype)
        expected = int(message.height) * elements_per_row
        if raw.size < expected:
            raise RuntimeError("La imagen 16UC1 tiene menos datos que los indicados por step.")
        depth = raw[:expected].reshape(int(message.height), elements_per_row)
        return depth[:, : int(message.width)].astype(np.float32) * 1e-3
    if encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        elements_per_row = message.step // dtype.itemsize
        raw = np.frombuffer(bytes(message.data), dtype=dtype)
        expected = int(message.height) * elements_per_row
        if raw.size < expected:
            raise RuntimeError("La imagen 32FC1 tiene menos datos que los indicados por step.")
        depth = raw[:expected].reshape(int(message.height), elements_per_row)
        return depth[:, : int(message.width)].astype(np.float32)
    raise RuntimeError(
        f"Codificación {message.encoding!r} no soportada; se esperaba 16UC1 o 32FC1."
    )


def grid_array(grid, field: str) -> np.ndarray:
    rows, cols = int(grid.rows), int(grid.cols)
    if rows <= 0 or cols <= 0 or len(grid.cells) != rows * cols:
        raise RuntimeError(
            f"DepthGrid inválida: rows={rows}, cols={cols}, cells={len(grid.cells)}."
        )
    values = np.array([float(getattr(cell, field)) for cell in grid.cells])
    return values.reshape(rows, cols)


def finite_stats(values: np.ndarray) -> tuple[int, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, math.nan, math.nan, math.nan
    return int(finite.size), float(np.min(finite)), float(np.mean(finite)), float(np.max(finite))


def close_or_nan(a: float, b: float, tolerance: float = 2e-3) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tolerance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera la figura imagen de profundidad -> ROI -> DepthGrid desde rosbag2."
    )
    parser.add_argument("bag", type=Path, help="Directorio del rosbag2.")
    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/depth/image_rect_raw",
        help="Tópico sensor_msgs/Image de profundidad.",
    )
    parser.add_argument(
        "--grid-topic",
        default="/perception/depth_grid",
        help="Tópico custom_interfaces/msg/DepthGrid.",
    )
    parser.add_argument(
        "--grid-index", type=int, default=0,
        help="Índice, desde cero, del mensaje DepthGrid que se representará.",
    )
    parser.add_argument("--roi-row", type=int, default=2, help="Fila de la ROI ampliada.")
    parser.add_argument("--roi-col", type=int, default=5, help="Columna de la ROI ampliada.")
    parser.add_argument(
        "--stat", choices=("min_m", "mean_m", "max_m"), default="min_m",
        help="Estadística mostrada en la grilla 5x10; no altera los datos del rosbag.",
    )
    parser.add_argument("--z-min", type=float, default=0.25, help="Límite válido inferior [m].")
    parser.add_argument("--z-max", type=float, default=5.0, help="Límite válido superior [m].")
    parser.add_argument(
        "--max-sync-ms", type=float, default=100.0,
        help="Máxima diferencia admisible entre tiempos de registro [ms].",
    )
    parser.add_argument("--storage-id", default="sqlite3", help="sqlite3 o mcap.")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "salida" / "reduccion_depth_grid.png",
        help="Archivo PNG de salida.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid_index < 0:
        raise RuntimeError("--grid-index debe ser mayor o igual que cero.")
    if not args.bag.exists():
        raise RuntimeError(f"No existe el rosbag: {args.bag}")
    if not 0.0 <= args.z_min < args.z_max:
        raise RuntimeError("Se requiere 0 <= --z-min < --z-max.")

    grid_time, grid = read_selected_grid(
        args.bag, args.storage_id, args.grid_topic, args.grid_index
    )
    delta_ns, image_time, image_message = read_nearest_image(
        args.bag, args.storage_id, args.depth_topic, grid_time
    )
    delta_ms = delta_ns / 1e6
    if delta_ms > args.max_sync_ms:
        raise RuntimeError(
            f"La imagen más cercana está a {delta_ms:.1f} ms de DepthGrid, por encima de "
            f"--max-sync-ms={args.max_sync_ms:.1f}. Elija otro --grid-index o aumente la "
            "tolerancia justificadamente."
        )

    depth = decode_depth(image_message)
    rows, cols = int(grid.rows), int(grid.cols)
    if not (0 <= args.roi_row < rows and 0 <= args.roi_col < cols):
        raise RuntimeError(
            f"ROI ({args.roi_row}, {args.roi_col}) fuera de una grilla {rows}x{cols}."
        )
    height, width = depth.shape
    x_edges = np.arange(cols + 1, dtype=int) * width // cols
    y_edges = np.arange(rows + 1, dtype=int) * height // rows
    x0, x1 = x_edges[args.roi_col], x_edges[args.roi_col + 1]
    y0, y1 = y_edges[args.roi_row], y_edges[args.roi_row + 1]
    roi = depth[y0:y1, x0:x1]
    valid_roi = roi[(roi >= args.z_min) & (roi <= args.z_max) & np.isfinite(roi)]
    computed = finite_stats(valid_roi)
    cell = grid.cells[args.roi_row * cols + args.roi_col]
    recorded = (int(cell.count), float(cell.min_m), float(cell.mean_m), float(cell.max_m))
    consistent = (
        computed[0] == recorded[0]
        and close_or_nan(computed[1], recorded[1])
        and close_or_nan(computed[2], recorded[2])
        and close_or_nan(computed[3], recorded[3])
    )
    if not consistent:
        print(
            "ADVERTENCIA: las estadísticas recalculadas no coinciden con DepthGrid. "
            "Revise el emparejamiento temporal y los límites --z-min/--z-max.\n"
            f"  recalculadas={computed}\n  publicadas={recorded}",
            file=sys.stderr,
        )

    field_labels = {"min_m": "mínimo", "mean_m": "media", "max_m": "máximo"}
    grid_values = grid_array(grid, args.stat)
    masked_depth = np.ma.masked_where(
        (~np.isfinite(depth)) | (depth < args.z_min) | (depth > args.z_max), depth
    )
    masked_roi = np.ma.masked_where(
        (~np.isfinite(roi)) | (roi < args.z_min) | (roi > args.z_max), roi
    )
    masked_grid = np.ma.masked_invalid(grid_values)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    image_ax = fig.add_subplot(layout[:, 0])
    roi_ax = fig.add_subplot(layout[0, 1])
    grid_ax = fig.add_subplot(layout[1, 1])

    image_plot = image_ax.imshow(
        masked_depth, cmap=cmap, vmin=args.z_min, vmax=args.z_max, interpolation="nearest"
    )
    for edge in x_edges:
        image_ax.axvline(edge - 0.5, color="white", lw=0.7, alpha=0.85)
    for edge in y_edges:
        image_ax.axhline(edge - 0.5, color="white", lw=0.7, alpha=0.85)
    image_ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
            fill=False, edgecolor="#ff2d55", lw=3,
        )
    )
    image_ax.set_title(f"Imagen de profundidad {width}x{height} con ROIs {rows}x{cols}")
    image_ax.set_xlabel("u [píxeles]")
    image_ax.set_ylabel("v [píxeles]")
    fig.colorbar(image_plot, ax=image_ax, label="Profundidad [m]", shrink=0.82)

    roi_ax.imshow(
        masked_roi, cmap=cmap, vmin=args.z_min, vmax=args.z_max, interpolation="nearest"
    )
    roi_ax.set_title(f"Ampliación de ROI ({args.roi_row}, {args.roi_col})")
    roi_ax.set_xlabel(f"u local: 0...{max(0, x1-x0-1)}")
    roi_ax.set_ylabel(f"v local: 0...{max(0, y1-y0-1)}")
    status = "coinciden" if consistent else "REVISAR SINCRONIZACIÓN"
    roi_ax.text(
        1.02, 0.5,
        "DepthGrid publicada\n"
        f"válidos: {recorded[0]}\n"
        f"mínimo: {recorded[1]:.3f} m\n"
        f"media: {recorded[2]:.3f} m\n"
        f"máximo: {recorded[3]:.3f} m\n\n"
        f"Verificación: {status}",
        transform=roi_ax.transAxes, va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"),
    )

    grid_plot = grid_ax.imshow(
        masked_grid, cmap=cmap, vmin=args.z_min, vmax=args.z_max,
        interpolation="nearest", aspect="auto"
    )
    for row in range(rows):
        for col in range(cols):
            value = grid_values[row, col]
            label = "--" if not math.isfinite(value) else f"{value:.2f}"
            color = "white" if not math.isfinite(value) or value > (args.z_min + args.z_max) / 2 else "black"
            grid_ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color)
    grid_ax.add_patch(
        Rectangle(
            (args.roi_col - 0.5, args.roi_row - 0.5), 1, 1,
            fill=False, edgecolor="#ff2d55", lw=3,
        )
    )
    grid_ax.set_title(f"DepthGrid {rows}x{cols}: {field_labels[args.stat]} por celda")
    grid_ax.set_xlabel("columna")
    grid_ax.set_ylabel("fila")
    grid_ax.set_xticks(range(cols))
    grid_ax.set_yticks(range(rows))
    fig.colorbar(grid_plot, ax=grid_ax, label="Distancia [m]", shrink=0.82)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura escrita en: {args.output}")
    print(f"Diferencia temporal de registro: {delta_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
