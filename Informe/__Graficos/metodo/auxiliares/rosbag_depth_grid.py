"""Lectura de profundidad y verificación de DepthGrid para las figuras del informe."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np



def ros_imports():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "No se encontraron las bibliotecas de ROS 2. Ejecute: "
            "source /opt/ros/humble/setup.bash && "
            "source /home/tomasdea/Tesis/develop/nav_mapper/install/setup.bash"
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
    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def read_grid(bag: Path, storage_id: str, topic: str, index: int):
    _, deserialize_message, get_message = ros_imports()
    reader = open_reader(bag, storage_id)
    types = topic_types(reader)
    expected = "custom_interfaces/msg/DepthGrid"
    if topic not in types:
        raise RuntimeError(f"El rosbag no contiene {topic!r}.")
    if types[topic] != expected:
        raise RuntimeError(f"{topic!r} tiene tipo {types[topic]!r}, no {expected!r}.")
    message_type = get_message(expected)
    current = 0
    while reader.has_next():
        read_topic, raw, timestamp = reader.read_next()
        if read_topic != topic:
            continue
        if current == index:
            return timestamp, deserialize_message(raw, message_type)
        current += 1
    raise RuntimeError(f"{topic!r} contiene {current} mensajes; no existe el índice {index}.")


def read_nearest_image(bag: Path, storage_id: str, topic: str, target_ns: int):
    _, deserialize_message, get_message = ros_imports()
    reader = open_reader(bag, storage_id)
    types = topic_types(reader)
    expected = "sensor_msgs/msg/Image"
    if topic not in types:
        raise RuntimeError(f"El rosbag no contiene {topic!r}.")
    if types[topic] != expected:
        raise RuntimeError(f"{topic!r} tiene tipo {types[topic]!r}, no {expected!r}.")
    message_type = get_message(expected)
    best = None
    while reader.has_next():
        read_topic, raw, timestamp = reader.read_next()
        if read_topic != topic:
            continue
        delta = abs(timestamp - target_ns)
        if best is None or delta < best[0]:
            best = (delta, timestamp, bytes(raw))
    if best is None:
        raise RuntimeError(f"{topic!r} no contiene imágenes.")
    delta, timestamp, raw = best
    return delta, timestamp, deserialize_message(raw, message_type)


def decode_depth(message) -> np.ndarray:
    encoding = message.encoding.upper()
    if encoding in {"16UC1", "MONO16"}:
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = 1e-3
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise RuntimeError(f"Codificación no soportada: {message.encoding!r}.")
    per_row = int(message.step) // dtype.itemsize
    raw = np.frombuffer(bytes(message.data), dtype=dtype)
    expected = int(message.height) * per_row
    if raw.size < expected:
        raise RuntimeError("La imagen contiene menos datos que los indicados por height y step.")
    image = raw[:expected].reshape(int(message.height), per_row)
    return image[:, : int(message.width)].astype(np.float32) * scale


def roi_bounds(row: int, col: int, x_edges: np.ndarray, y_edges: np.ndarray):
    return x_edges[col], x_edges[col + 1], y_edges[row], y_edges[row + 1]


def classify(roi: np.ndarray, z_min: float, z_max: float) -> dict[str, np.ndarray]:
    finite_positive = np.isfinite(roi) & (roi > 0.0)
    return {
        "invalid": ~finite_positive,
        "below": finite_positive & (roi < z_min),
        "valid": finite_positive & (roi >= z_min) & (roi <= z_max),
        "above": finite_positive & (roi > z_max),
    }


def roi_metrics(roi: np.ndarray, z_min: float, z_max: float) -> dict[str, float]:
    masks = classify(roi, z_min, z_max)
    valid = roi[masks["valid"]]
    return {
        "invalid": int(np.count_nonzero(masks["invalid"])),
        "below": int(np.count_nonzero(masks["below"])),
        "valid": int(valid.size),
        "above": int(np.count_nonzero(masks["above"])),
        "min": float(np.min(valid)) if valid.size else math.nan,
        "mean": float(np.mean(valid)) if valid.size else math.nan,
        "max": float(np.max(valid)) if valid.size else math.nan,
    }


def same_or_nan(a: float, b: float, tolerance: float = 2e-3) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tolerance


def validate_cell(grid, row, col, metrics):
    cell = grid.cells[row * int(grid.cols) + col]
    expected = (metrics["valid"], metrics["min"], metrics["mean"], metrics["max"])
    recorded = (int(cell.count), float(cell.min_m), float(cell.mean_m), float(cell.max_m))
    ok = expected[0] == recorded[0] and all(
        same_or_nan(a, b) for a, b in zip(expected[1:], recorded[1:])
    )
    return ok, recorded
