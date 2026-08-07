#!/usr/bin/env python3
"""Genera el esquema de geometría estéreo y disparidad."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "salida/estereo_disparidad.png"


def main():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    left = np.array([-1.0, 0.0]); right = np.array([1.0, 0.0]); point = np.array([0.35, 3.0])
    image_y = 0.48
    left_hit = left + (image_y / point[1]) * (point - left)
    right_hit = right + (image_y / point[1]) * (point - right)
    ax.plot([left[0], point[0]], [left[1], point[1]], "C0")
    ax.plot([right[0], point[0]], [right[1], point[1]], "C1")
    ax.plot([left[0]-.42, left[0]+.42], [image_y, image_y], color="C0", lw=2)
    ax.plot([right[0]-.42, right[0]+.42], [image_y, image_y], color="C1", lw=2)
    ax.scatter(*left_hit, color="C0", s=28, zorder=3); ax.scatter(*right_hit, color="C1", s=28, zorder=3)
    ax.scatter(*left, color="C0"); ax.scatter(*right, color="C1"); ax.scatter(*point, color="k")
    ax.annotate("cámara IR izquierda", left, xytext=(-2.0, -.45))
    ax.annotate("cámara IR derecha", right, xytext=(.65, -.45))
    ax.annotate("punto observado", point, xytext=(.55, 2.8))
    ax.annotate("", right, left, arrowprops=dict(arrowstyle="<->")); ax.text(0, -.12, "$B$", ha="center")
    ax.annotate("", (left[0]-.26, image_y), (left[0]-.26, 0), arrowprops=dict(arrowstyle="<->", color="0.25"))
    ax.text(left[0]-.36, image_y/2, "$f$", ha="right", va="center")
    ax.text(left_hit[0]+.04, image_y+.07, "$u_L$", color="C0")
    ax.text(right_hit[0]+.04, image_y+.07, "$u_R$", color="C1")
    ax.text(0, image_y+.09, "planos de imagen", ha="center", color="0.3", fontsize=9)
    ax.plot([point[0], point[0]], [0, point[1]], "k--", alpha=.5); ax.text(point[0]+.08, 1.5, "$Z$")
    ax.text(-1.65, 3.24, r"$d=u_L-u_R,\qquad Z=\dfrac{fB}{d}$")
    ax.set(xlim=(-2.2, 2.2), ylim=(-.6, 3.6), aspect="equal"); ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
