"""Lectura de escenas R01 para las figuras de evidencia libre y ocupada."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np


BAG = Path(
    "/home/tomasdea/Tesis/bags/experimentos_tesis/derivados/"
    "R01_2_v2_mapa_completo"
)
RAW_BAG = Path(
    "/home/tomasdea/Tesis/bags/experimentos_tesis/"
    "R01_estatica_multirango_calibracion_2"
)
PUBLISHED_PHASE = 9  # configuración conceptual E1-G
OFFSET_S = 8.0
TOPICS = (
    "/depth_obstacle_filter/debug/ground_cloud",
    "/depth_obstacle_filter/debug/ceiling_cloud",
    "/depth_obstacle_filter/obstacle_cloud",
    "/local_mapper/occupancy_grid",
    "/nav_odom",
)


def ros_modules():
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    return rosbag2_py, deserialize_message, get_message


def nearest_raw(connection, topic_id: int, target: int) -> bytes:
    candidates = []
    for operator, order in (("<=", "DESC"), (">=", "ASC")):
        row = connection.execute(
            f"SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp{operator}? "
            f"ORDER BY timestamp {order} LIMIT 1",
            (topic_id, target),
        ).fetchone()
        if row:
            candidates.append(row)
    return bytes(min(candidates, key=lambda row: abs(row[0] - target))[1])


def records_at_target(published_phase=PUBLISHED_PHASE, offset_s=OFFSET_S, extra_topics=()):
    _, deserialize, get_message = ros_modules()
    database = next(BAG.glob("*.db3"))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    topics = {
        name: (topic_id, message_type)
        for topic_id, name, message_type in connection.execute(
            "SELECT id,name,type FROM topics"
        )
    }
    phase_id, phase_name = topics["/experiment/phase"]
    phase_type = get_message(phase_name)
    phases = [
        (deserialize(bytes(raw), phase_type).data, timestamp)
        for timestamp, raw in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (phase_id,),
        )
    ]
    phase_start = next(timestamp for value, timestamp in phases if value == published_phase)
    best_target = phase_start + int(offset_s * 1e9)

    result = {}
    for topic in (*TOPICS, *extra_topics):
        topic_id, message_name = topics[topic]
        result[topic] = deserialize(
            nearest_raw(connection, topic_id, best_target),
            get_message(message_name),
        )
    connection.close()
    return result


def raw_depth_at_phase(published_phase=PUBLISHED_PHASE, offset_s=OFFSET_S):
    """Obtiene el frame crudo que originó la configuración procesada."""
    _, deserialize, get_message = ros_modules()
    database = next(RAW_BAG.glob("*.db3"))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    topics = {
        name: (topic_id, message_type)
        for topic_id, name, message_type in connection.execute(
            "SELECT id,name,type FROM topics"
        )
    }
    phase_id, phase_name = topics["/experiment/phase"]
    phase_type = get_message(phase_name)
    phases = [
        (deserialize(bytes(raw), phase_type).data, timestamp)
        for timestamp, raw in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (phase_id,),
        )
    ]
    phase_start = next(timestamp for value, timestamp in phases if value == published_phase)
    target = phase_start + int(offset_s * 1e9)
    topic_id, message_name = topics["/camera/camera/depth/image_rect_raw"]
    message = deserialize(nearest_raw(connection, topic_id, target), get_message(message_name))
    connection.close()
    row = np.frombuffer(bytes(message.data), dtype=np.uint16).reshape(message.height, message.step//2)
    depth = row[:, :message.width].astype(np.float32) * 1e-3
    depth[depth <= 0] = np.nan
    return depth


def cloud_array(msg) -> np.ndarray:
    offsets = {field.name: field.offset for field in msg.fields}
    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [endian + "f4"] * 3,
            "offsets": [offsets[name] for name in ("x", "y", "z")],
            "itemsize": msg.point_step,
        }
    )
    data = np.frombuffer(bytes(msg.data), dtype=dtype)
    points = np.column_stack((data["x"], data["y"], data["z"]))
    return points[np.isfinite(points).all(axis=1)]
