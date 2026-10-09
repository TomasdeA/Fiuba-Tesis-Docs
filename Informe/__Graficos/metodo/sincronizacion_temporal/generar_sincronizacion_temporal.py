#!/usr/bin/env python3
"""Genera un esquema simple de la asociación entre observación y odometría."""

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "salida/sincronizacion_temporal.png"


def main():
    fig, ax = plt.subplots(figsize=(12.5, 2.8), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = .42
    ax.annotate("", xy=(.92, y), xytext=(.08, y), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.8))
    ax.text(.935, y, "$t$", ha="left", va="center", fontsize=13,
            color="#34495e", transform=ax.transAxes)

    for i, x in enumerate([.16, .34, .49, .67, .84], start=1):
        chosen = x in (.49, .67)
        ax.scatter([x], [y], s=150 if chosen else 75,
                   color="#2e7d32" if chosen else "#2471a3",
                   edgecolor="white", linewidth=1.3, zorder=4,
                   transform=ax.transAxes)
        ax.text(x, y+.075, rf"pose $o_{i}$", ha="center", fontsize=10,
                color="#1f4e79", transform=ax.transAxes)

    ax.scatter([.55], [y], marker="D", s=115, color="#c62828",
               edgecolor="white", linewidth=1.3, zorder=5,
               transform=ax.transAxes)
    ax.text(.55, y-.09, "instante de la observación",
            ha="center", va="top", fontsize=10, color="#922b21",
            transform=ax.transAxes)
    ax.annotate("Pose en el instante de la observación:\ninterpolación entre $o_3$ y $o_4$",
                xy=(.55, y+.012), xytext=(.55, .87), xycoords=ax.transAxes,
                ha="center", fontsize=12, weight="bold", color="#21652a",
                arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.8))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
