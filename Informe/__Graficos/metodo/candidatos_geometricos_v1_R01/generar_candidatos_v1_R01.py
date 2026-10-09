#!/usr/bin/env python3
"""Visualiza la preparación geométrica de v1 sobre la fase lógica 4 de R01."""
from pathlib import Path
import sqlite3
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo

TOOLS = Path("/home/tomasdea/Tesis/tools/experimentos")
sys.path.insert(0, str(TOOLS))
from generar_contacto_fases import decode_depth, nearest_image, phase_messages  # noqa: E402

BAG = Path("/home/tomasdea/Tesis/bags/experimentos_tesis/R01_estatica_multirango_calibracion_2")
OUT = Path(__file__).resolve().parent / "salida" / "candidatos_geometricos_v1_R01_fase4.png"
PUBLISHED_PHASE = 5
OFFSET_S = 8.0
STRIDE = 2
Z_MIN, Z_MAX = 0.25, 5.0


def nearest_camera_info(connection, topic, timestamp):
    topic_id = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()[0]
    rows = []
    for operator, order in (("<=", "DESC"), (">=", "ASC")):
        row = connection.execute(
            f"SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp{operator}? "
            f"ORDER BY timestamp {order} LIMIT 1", (topic_id, timestamp)
        ).fetchone()
        if row: rows.append(row)
    _, raw = min(rows, key=lambda row: abs(row[0] - timestamp))
    return deserialize_message(bytes(raw), CameraInfo)


def main():
    database = next(BAG.glob("*.db3"))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    phases = dict(phase_messages(connection))
    target = phases[PUBLISHED_PHASE] + round(OFFSET_S * 1e9)
    depth = decode_depth(nearest_image(
        connection, "/camera/camera/depth/image_rect_raw", target
    )).astype(np.float32)
    info = nearest_camera_info(
        connection, "/camera/camera/depth/camera_info", target
    )
    connection.close()

    height, width = depth.shape
    vv, uu = np.mgrid[0:height:STRIDE, 0:width:STRIDE]
    sampled = depth[::STRIDE, ::STRIDE]
    finite_positive = np.isfinite(sampled) & (sampled > 0)
    below = finite_positive & (sampled < Z_MIN)
    inside = finite_positive & (sampled >= Z_MIN) & (sampled < Z_MAX)
    above = finite_positive & (sampled >= Z_MAX)
    invalid = ~finite_positive

    fx, fy, cx, cy = float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])
    z = sampled[inside]
    x = (uu[inside] - cx) * z / fx
    y = (vv[inside] - cy) * z / fy

    # El nodo retiene una de cada 16 mediciones beyond-range como extremo libre.
    above_flat = np.flatnonzero(above.ravel())
    retained_above = above_flat[::16]
    categories = np.zeros(sampled.shape, dtype=np.uint8)
    categories[below] = 1
    categories[inside] = 2
    categories[above] = 3
    cmap = ListedColormap(["#d9d9d9", "#d81b60", "#2ca25f", "#756bb1"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flat
    shown = np.ma.masked_where(~finite_positive, sampled)
    p0 = ax0.imshow(shown, cmap="turbo", vmin=0, vmax=Z_MAX, aspect="auto")
    ax0.set_title(f"1. Profundidad de entrada: {width}×{height} = {depth.size:,} píxeles", weight="bold")
    ax0.set(xlabel="u [píxeles]", ylabel="v [píxeles]")
    fig.colorbar(p0, ax=ax0, label="profundidad axial Z [m]")

    ax1.imshow(categories, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", aspect="auto")
    ax1.set_title(
        f"2. Muestreo stride={STRIDE}: {sampled.size:,} candidatos ({100*sampled.size/depth.size:.0f}% de la imagen)",
        weight="bold",
    )
    ax1.set(xlabel="u/stride", ylabel="v/stride")
    labels = [
        ("inválidos", invalid, "#777777"), (f"Z<{Z_MIN:g} m", below, "#d81b60"),
        (f"{Z_MIN:g}≤Z<{Z_MAX:g} m", inside, "#2ca25f"), (f"Z≥{Z_MAX:g} m", above, "#756bb1")
    ]
    ax1.legend(
        [plt.Line2D([0],[0], marker="s", color="none", markerfacecolor=c, markersize=10) for _,_,c in labels],
        [f"{name}: {np.count_nonzero(mask):,}" for name,mask,_ in labels],
        loc="lower right", fontsize=8, framealpha=.92,
    )

    use = slice(None, None, max(1, len(z)//45000))
    p2 = ax2.scatter(x[use], y[use], c=z[use], cmap="turbo", vmin=Z_MIN, vmax=Z_MAX,
                     s=2, alpha=.75, linewidths=0)
    ax2.invert_yaxis(); ax2.set_aspect("equal", adjustable="box")
    ax2.set_title(f"3. Retroproyección métrica: {len(z):,} puntos dentro del rango", weight="bold")
    ax2.set(xlabel="X en marco de cámara [m]", ylabel="Y en marco de cámara [m]")
    fig.colorbar(p2, ax=ax2, label="Z [m]")

    ax3.axis("off")
    ax3.set_title("4. Diferencia metodológica después del filtro compartido", weight="bold")
    box = dict(boxstyle="round,pad=.6", ec="#315a8a", lw=1.3)
    ax3.text(.5, .88, "Muestras con 0,25 m < Z < 5 m\n(marco óptico de la cámara)",
             ha="center", va="center", fontsize=11, weight="bold",
             bbox={**box, "fc": "#e8f1fb"})
    ax3.annotate("", xy=(.25, .68), xytext=(.43, .80), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    ax3.annotate("", xy=(.75, .68), xytext=(.57, .80), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    ax3.text(.24, .57, "v0\nregiones rectangulares\nde la imagen\n↓\nmínimo axial Z",
             ha="center", va="center", fontsize=10,
             bbox={**box, "fc": "#fff2cc", "ec": "#b8860b"})
    ax3.text(.76, .57, "v1\nretroproyección 3D\n↓\nalineación gravitacional\n↓\nclasificación geométrica\n↓\nmínimo radial √(X²+Z²)",
             ha="center", va="center", fontsize=10,
             bbox={**box, "fc": "#dff2df", "ec": "#3c8c4a"})
    ax3.text(.5, .12,
             f"Z ≥ 5 m: no son obstáculos; v1 conserva {len(retained_above):,} extremos libres "
             "submuestreados para v2",
             transform=ax3.transAxes, ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#eee7f7", ec="#756bb1"))
    fig.suptitle("Rango compartido y cambio de representación entre v0 y v1",
                 fontsize=17, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)
    print({"pixels": depth.size, "sampled": sampled.size,
           "inside": len(z), "free_endpoints": len(retained_above)})


if __name__ == "__main__": main()
