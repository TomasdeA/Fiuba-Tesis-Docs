#!/usr/bin/env python3
"""Alternativa Matplotlib para ilustrar alineación gravitacional con R01 fase 8."""
from pathlib import Path
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Imu

TOOLS = Path("/home/tomasdea/Tesis/tools/experimentos")
sys.path.insert(0, str(TOOLS))
from generar_contacto_fases import decode_depth, nearest_image, phase_messages  # noqa: E402

BAG = Path("/home/tomasdea/Tesis/bags/experimentos_tesis/R01_estatica_multirango_calibracion_2")
OUT = Path(__file__).resolve().parent / "salida" / "alineacion_gravedad_R01_fase8_planos.png"
OUT_GROUND_BAND = Path(__file__).resolve().parents[2] / "resultados" / \
    "clasificacion_suelo_distancia/salida/clasificacion_suelo_tolerancia_adaptativa.png"
PUBLISHED_PHASE = 9  # fase lógica 8: superficie superior sin obstáculo debajo
OFFSET_S = 8.0
STRIDE = 3


def nearest_typed(connection, topic, timestamp, message_type):
    topic_id = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()[0]
    rows = []
    for operator, order in (("<=", "DESC"), (">=", "ASC")):
        row = connection.execute(
            f"SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp{operator}? "
            f"ORDER BY timestamp {order} LIMIT 1", (topic_id, timestamp)
        ).fetchone()
        if row: rows.append(row)
    _, raw = min(rows, key=lambda row: abs(row[0] - timestamp))
    return deserialize_message(bytes(raw), message_type)


def average_accel(connection, timestamp, half_window_s=.5):
    topic_id = connection.execute(
        "SELECT id FROM topics WHERE name='/camera/camera/accel/sample'"
    ).fetchone()[0]
    radius = round(half_window_s * 1e9)
    rows = connection.execute(
        "SELECT data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ?",
        (topic_id, timestamp-radius, timestamp+radius),
    )
    values = []
    for (raw,) in rows:
        a = deserialize_message(bytes(raw), Imu).linear_acceleration
        values.append((a.x, a.y, a.z))
    return np.median(np.asarray(values), axis=0), len(values)


def rotation_to_down(accel):
    a = accel / np.linalg.norm(accel)
    b = np.array([0.0, -1.0, 0.0])
    dot = float(np.dot(a, b))
    if dot < -0.999:
        return np.diag([1.0, -1.0, -1.0])
    q = np.array([1.0 + dot, *(np.cross(a, b))])
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def floor_plane(points):
    """Ajuste robusto y=a*x+b*z+c sobre la franja inferior de la nube alineada."""
    candidates = (points[:, 1] > 1.55) & (points[:, 2] > .55) & (points[:, 2] < 4.8)
    ids = np.flatnonzero(candidates)
    rng = np.random.default_rng(7)
    best = np.zeros(len(points), dtype=bool)
    design = np.column_stack((points[:, 0], points[:, 2], np.ones(len(points))))
    for _ in range(700):
        sample = rng.choice(ids, 3, replace=False)
        matrix = design[sample]
        if abs(np.linalg.det(matrix)) < 1e-5:
            continue
        model = np.linalg.solve(matrix, points[sample, 1])
        residual = np.abs(points[:, 1] - design @ model)
        inliers = candidates & (residual < .045)
        if inliers.sum() > best.sum():
            best = inliers
    model, *_ = np.linalg.lstsq(design[best], points[best, 1], rcond=None)
    residual = np.abs(points[:, 1] - design @ model)
    inliers = candidates & (residual < .055)
    model, *_ = np.linalg.lstsq(design[inliers], points[inliers, 1], rcond=None)
    return model, inliers


def adaptive_tolerance(points):
    """Reproduce el diagnóstico conceptual del estimador: 3σNN, limitado a 1--10 cm."""
    rng = np.random.default_rng(42)
    sample = points[rng.choice(len(points), min(100, len(points)), replace=False)]
    delta = sample[:, None, :] - sample[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    sigma_nn = np.min(distances, axis=1).std()
    return float(np.clip(3.0 * sigma_nn, .01, .10)), float(sigma_nn)


def scatter_projection(axis, points, floor, horizontal, vertical, title, limits):
    step = max(1, len(points) // 42000)
    other = np.flatnonzero(~floor)[::step]
    ground = np.flatnonzero(floor)[::max(1, step // 2)]
    axis.scatter(points[other, horizontal], points[other, vertical], s=1.2,
                 color="#aeb6bf", alpha=.24, linewidths=0, rasterized=True,
                 label="resto de la escena")
    axis.scatter(points[ground, horizontal], points[ground, vertical], s=3.0,
                 color="#d81b60", alpha=.9, linewidths=0, rasterized=True,
                 label="inliers del suelo")
    axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1]); axis.invert_yaxis()
    axis.set_title(title, weight="bold", fontsize=11)
    axis.grid(alpha=.18); axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="lower right", fontsize=8, markerscale=2.5)


def draw_camera_frame(axis, rotation, horizontal, vertical, scale=.62, show_x=False):
    """Marca la cámara, su eje óptico Z y, en vista frontal, el eje X."""
    axes = [(2, "Z", "#1565c0")]
    if show_x:
        axes.append((0, "X", "#d62728"))
    for index, name, color in axes:
        basis = rotation[:, index]
        dx, dy = scale * basis[horizontal], scale * basis[vertical]
        if np.hypot(dx, dy) < .07:
            axis.scatter([0], [0], s=52, facecolors="none", edgecolors=color,
                         linewidths=1.8, zorder=8)
            axis.text(.07, -.08, f"{name} ⊙", color=color, weight="bold", fontsize=9, zorder=9)
        else:
            axis.annotate("", xy=(dx, dy), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4), zorder=8)
            axis.text(dx * 1.08, dy * 1.08, name, color=color, weight="bold", fontsize=10,
                      ha="center", va="center", zorder=9)
    axis.scatter([0], [0], s=18, color="black", zorder=10)


def plane_sections(plane, rotation):
    """Secciones centrales del plano, expresadas en marco crudo y alineado."""
    a, b, c = plane
    z = np.linspace(.55, 5.0, 180)
    aligned_side = np.column_stack((np.zeros_like(z), b*z+c, z))
    x = np.linspace(-3.0, 3.0, 180)
    z_ref = 2.5
    aligned_front = np.column_stack((x, a*x+b*z_ref+c, np.full_like(x, z_ref)))
    # aligned = raw @ rotation.T  =>  raw = aligned @ rotation
    return aligned_side @ rotation, aligned_side, aligned_front @ rotation, aligned_front


def draw_plane(axis, points, horizontal, vertical):
    axis.plot(points[:, horizontal], points[:, vertical], color="#111111", ls="--", lw=2.2,
              label="sección del plano estimado", zorder=7)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="lower right", fontsize=8, markerscale=2.5)


def main():
    database = next(BAG.glob("*.db3"))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    marker = dict(phase_messages(connection))[PUBLISHED_PHASE]
    target = marker + round(OFFSET_S*1e9)
    depth = decode_depth(nearest_image(
        connection, "/camera/camera/depth/image_rect_raw", target
    )).astype(np.float32)
    info = nearest_typed(connection, "/camera/camera/depth/camera_info", target, CameraInfo)
    accel, accel_count = average_accel(connection, target)
    connection.close()

    vv, uu = np.mgrid[0:depth.shape[0]:STRIDE, 0:depth.shape[1]:STRIDE]
    z = depth[::STRIDE, ::STRIDE]
    valid = np.isfinite(z) & (z >= .25) & (z <= 5.0)
    z = z[valid]
    x = (uu[valid]-info.k[2])*z/info.k[0]
    y = (vv[valid]-info.k[5])*z/info.k[4]
    raw = np.column_stack((x,y,z))
    rotation = rotation_to_down(accel)
    aligned = raw @ rotation.T
    plane, floor = floor_plane(aligned)
    tolerance, sigma_nn = adaptive_tolerance(aligned)
    # Los mismos índices se muestran antes y después: no se reclasifica la nube cruda.
    # Inclinaciones derivadas de la normal 3D del mismo plano. Esto evita que
    # correlaciones entre X y Z sesguen regresiones bidimensionales separadas.
    normal_aligned = np.array([-plane[0], 1.0, -plane[1]])
    normal_aligned /= np.linalg.norm(normal_aligned)
    normal_raw = rotation.T @ normal_aligned
    raw_roll = -normal_raw[0] / normal_raw[1]
    raw_pitch = -normal_raw[2] / normal_raw[1]
    aligned_roll = plane[0]
    aligned_pitch = plane[1]
    raw_side_plane, aligned_side_plane, raw_front_plane, aligned_front_plane = \
        plane_sections(plane, rotation)

    fig = plt.figure(figsize=(15,10), constrained_layout=True)
    gs = fig.add_gridspec(2,3)
    ax_depth = fig.add_subplot(gs[0,0])
    shown_depth = np.ma.masked_where((depth<=0)|(~np.isfinite(depth)), depth)
    p0 = ax_depth.imshow(shown_depth, cmap="turbo", vmin=.25, vmax=5, aspect="auto")
    ax_depth.set_title("Fotograma de profundidad — superficie elevada", weight="bold")
    ax_depth.set(xlabel="u [píxeles]", ylabel="v [píxeles]")
    fig.colorbar(p0, ax=ax_depth, label="profundidad axial [m]")

    ax_raw_side = fig.add_subplot(gs[0,1])
    scatter_projection(ax_raw_side, raw, floor, 2, 1,
                       "Vista lateral antes (pitch)", ((-.35,5.2),(-2,1.5)))
    draw_plane(ax_raw_side, raw_side_plane, 2, 1)
    draw_camera_frame(ax_raw_side, np.eye(3), 2, 1)
    ax_raw_side.set(xlabel="Z: distancia frontal [m]", ylabel="Y óptico [m]")

    ax_aligned_side = fig.add_subplot(gs[0,2])
    scatter_projection(ax_aligned_side, aligned, floor, 2, 1,
                       "Vista lateral alineada", ((-.35,5.2),(-1,2.4)))
    draw_plane(ax_aligned_side, aligned_side_plane, 2, 1)
    draw_camera_frame(ax_aligned_side, rotation, 2, 1)
    ax_aligned_side.set(xlabel="Z: distancia frontal [m]", ylabel="Y gravitacional [m]")

    ax_raw_front = fig.add_subplot(gs[1,1])
    scatter_projection(ax_raw_front, raw, floor, 0, 1,
                       "Vista frontal antes (roll)", ((-4,4),(-2,1.5)))
    draw_plane(ax_raw_front, raw_front_plane, 0, 1)
    draw_camera_frame(ax_raw_front, np.eye(3), 0, 1, show_x=True)
    ax_raw_front.set(xlabel="X: lateral [m]", ylabel="Y óptico [m]")

    ax_aligned_front = fig.add_subplot(gs[1,2])
    scatter_projection(ax_aligned_front, aligned, floor, 0, 1,
                       "Vista frontal alineada", ((-4,4),(-1,2.4)))
    draw_plane(ax_aligned_front, aligned_front_plane, 0, 1)
    draw_camera_frame(ax_aligned_front, rotation, 0, 1, show_x=True)
    ax_aligned_front.set(xlabel="X: lateral [m]", ylabel="Y gravitacional [m]")

    ax_summary = fig.add_subplot(gs[1,0]); ax_summary.axis("off")
    pitch_raw = np.degrees(np.arctan(raw_pitch)); pitch_aligned = np.degrees(np.arctan(aligned_pitch))
    roll_raw = np.degrees(np.arctan(raw_roll)); roll_aligned = np.degrees(np.arctan(aligned_roll))
    # Roll y pitch interpretativos en el marco óptico, según la ecuación del
    # Marco teórico. GravityAligner aplica luego una única rotación mínima 3D.
    gravity_down = -accel / np.linalg.norm(accel)
    correction_roll = np.degrees(np.arctan2(gravity_down[0], gravity_down[1]))
    correction_pitch = np.degrees(np.arctan2(
        gravity_down[2], np.hypot(gravity_down[0], gravity_down[1])))
    floor_tilt = np.degrees(np.arctan(np.hypot(plane[0], plane[1])))
    ax_summary.set_title("Propiedad obtenida", weight="bold", pad=18)
    ax_summary.text(.5,.84,"Aceleración cuasiestática medida",ha="center",fontsize=12,weight="bold")
    ax_summary.text(.5,.74,f"a = ({accel[0]:+.2f}, {accel[1]:+.2f}, {accel[2]:+.2f}) m/s²\n"
                              f"mediana de {accel_count} muestras",ha="center",fontsize=11)
    ax_summary.annotate("compensación de roll y pitch\nhasta la vertical canónica",xy=(.5,.44),xytext=(.5,.58),
                        ha="center",arrowprops=dict(arrowstyle="-|>",lw=2),fontsize=11)
    ax_summary.text(.5,.36,f"corrección gravitacional\npitch: {correction_pitch:+.1f}° · roll: {correction_roll:+.1f}°",
                    ha="center",fontsize=14,weight="bold",
                    bbox=dict(boxstyle="round,pad=.5",fc="#e8f5e9",ec="#2e7d32",lw=1.5))
    ax_summary.text(.5,.16,"Plano medido después de alinear",ha="center",fontsize=12,weight="bold")
    ax_summary.text(.5,.07,f"pendiente frontal: {pitch_aligned:+.1f}° · lateral: {roll_aligned:+.1f}°\n"
                              f"inclinación total: {floor_tilt:.1f}° · {floor.sum():,} inliers",
                    ha="center",fontsize=11)
    fig.suptitle("Alineación gravitacional — configuración con superficie elevada",
                 fontsize=18, weight="bold")
    OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=220,bbox_inches="tight")
    plt.close(fig)

    # Figura específica de la regla usada en stage7_classify. Una proyección
    # frontal X-Y pierde Z y no permite dibujar una única sección del plano
    # válida para todos los puntos. Por eso la decisión se muestra aparte en
    # función de la distancia firmada 3D que evalúa realmente el código.
    norm = np.sqrt(1.0 + plane[0]**2 + plane[1]**2)
    # El C++ fuerza ny<0: d<0 queda del lado inferior del plano (+Y).
    signed_distance = -(aligned[:, 1] - (plane[0]*aligned[:, 0] +
                        plane[1]*aligned[:, 2] + plane[2])) / norm
    classified_ground = signed_distance < tolerance
    near_rejected = ((signed_distance >= tolerance) &
                     (signed_distance < 2.0*tolerance))
    near_z = aligned[near_rejected, 2]
    near_radial = np.hypot(aligned[near_rejected, 0], aligned[near_rejected, 2])
    fig_band, (ax_front, ax_dist) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    step = max(1, len(aligned)//70000)
    other = np.flatnonzero(~classified_ground)[::step]
    ground = np.flatnonzero(classified_ground)[::step]
    for axis in (ax_front, ax_dist):
        # Dibujar primero el suelo y después los puntos rechazados. En la
        # proyección X-Y varios puntos con distinto Z coinciden visualmente;
        # este orden evita que el suelo oculte los casos no clasificados.
        axis.scatter(aligned[ground,0],
                     aligned[ground,1] if axis is ax_front else signed_distance[ground],
                     s=2.0, color="#d81b60", alpha=.42, linewidths=0,
                     rasterized=True, label="clasificados como suelo")
        axis.scatter(aligned[other,0],
                     aligned[other,1] if axis is ax_front else signed_distance[other],
                     s=3.0, color="#174a7e", alpha=.72, linewidths=0,
                     rasterized=True, label="no clasificados como suelo", zorder=4)

    draw_camera_frame(ax_front, rotation, 0, 1, show_x=True)
    ax_front.set(xlim=(-4,4), ylim=(-1,2.4), xlabel="X: lateral [m]",
                 ylabel="Y gravitacional [m]")
    ax_front.invert_yaxis(); ax_front.grid(alpha=.2)
    ax_front.set_aspect("equal", adjustable="box")
    ax_front.set_title("Vista frontal alineada\n(proyección: se pierde la coordenada Z)",
                       weight="bold")
    ax_front.legend(loc="upper right", fontsize=9, markerscale=3)

    ax_dist.axhspan(-tolerance, tolerance, color="#f9a3c3", alpha=.50,
                    label=r"corredor simétrico $\pm\varepsilon$")
    ax_dist.axhline(0, color="#111111", ls="--", lw=2.0, label="plano estimado")
    ax_dist.axhline(tolerance, color="#7b1fa2", lw=2.2,
                    label=r"frontera efectiva $d=+\varepsilon$")
    ax_dist.axhspan(-.45, -tolerance, color="#f8d7e3", alpha=.18)
    ax_dist.set(xlim=(-4,4), ylim=(-.45,.45), xlabel="X: lateral [m]",
                ylabel=r"distancia firmada al plano $d$ [m]")
    ax_dist.grid(alpha=.2)
    ax_dist.set_title("Decisión en la magnitud evaluada por el código",
                      weight="bold")
    ax_dist.text(.02, .04,
            rf"$\varepsilon={100*tolerance:.1f}$ cm para esta observación "
            rf"($3\sigma_{{NN}}$, acotado entre 1 y 10 cm)" "\n"
            r"Regla implementada: suelo si $d_{\mathrm{plano}}<0$ o "
            r"$|d_{\mathrm{plano}}|<\varepsilon$",
            transform=ax_dist.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#777777", alpha=.94))
    if len(near_radial):
        qz = np.percentile(near_z, [25, 50, 75])
        qr = np.percentile(near_radial, [25, 50, 75])
        far_fraction = 100.0*np.mean(near_radial >= 3.0)
        ax_dist.text(.98, .43,
                     "Puntos apenas fuera de la tolerancia\n"
                     rf"($\varepsilon\leq d<2\varepsilon$; n={len(near_radial):,})" "\n"
                     rf"Z mediano: {qz[1]:.2f} m (Q1–Q3: {qz[0]:.2f}–{qz[2]:.2f} m)" "\n"
                     rf"distancia radial mediana: {qr[1]:.2f} m "
                     rf"(Q1–Q3: {qr[0]:.2f}–{qr[2]:.2f} m)" "\n"
                     rf"{far_fraction:.1f}% a 3 m o más: intensidad háptica 0%",
                     transform=ax_dist.transAxes, ha="right", va="center", fontsize=9.3,
                     bbox=dict(boxstyle="round", fc="#eaf2f8", ec="#174a7e", alpha=.96))
    ax_dist.legend(loc="upper right", fontsize=8.5, markerscale=3)
    fig_band.suptitle("Clasificación del suelo después de alinear la nube con la gravedad",
                      fontsize=15, weight="bold")
    fig_band.savefig(OUT_GROUND_BAND, dpi=220, bbox_inches="tight")
    plt.close(fig_band)
    print(OUT)
    print(OUT_GROUND_BAND)
    print(f"accel={accel}, pitch_correction={correction_pitch:.2f}, "
          f"roll_correction={correction_roll:.2f}, floor={floor.sum()}, "
          f"pitch={pitch_aligned:.2f}, roll={roll_aligned:.2f}, tilt={floor_tilt:.2f}, plane={plane}")
    print(f"sigma_nn={sigma_nn:.5f}, adaptive_tolerance={tolerance:.5f}, "
          f"classified_ground={classified_ground.sum()}")
    if len(near_radial):
        print("near_rejected", len(near_radial),
              "z_quantiles", np.percentile(near_z, [0,25,50,75,100]),
              "radial_quantiles", np.percentile(near_radial, [0,25,50,75,100]),
              "radial_ge_3", np.mean(near_radial >= 3.0))


if __name__ == "__main__": main()
