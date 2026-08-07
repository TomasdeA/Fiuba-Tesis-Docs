#!/usr/bin/env python3
"""Genera el esquema de asociación temporal entre nubes y odometría."""

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "salida/sincronizacion_temporal.png"


def main():
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    cases = [
        (axes[0], [1.0, 2.3, 3.7, 5.2], 4.3, 3.7, "Operación normal: última muestra no posterior"),
        (axes[1], [4.8, 5.6, 6.5], 4.3, 4.8, "Arranque: primera muestra disponible, aunque sea posterior"),
    ]
    for ax, odom_times, observation, selected, title in cases:
        ax.axhline(0, color="0.2", lw=1)
        for x in odom_times:
            ax.plot(x, 0, "o", color="C0"); ax.text(x, .12, "odom", ha="center", fontsize=8)
        ax.plot(observation, 0, "s", color="C3", ms=7)
        ax.text(observation, -.25, "obstáculos + libres\n(sello común)", ha="center", va="top", fontsize=8)
        ax.annotate("muestra elegida", (selected, .02), (selected, .62), ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
        ax.set_title(title, fontsize=10); ax.set_ylim(-.65, .9); ax.set_yticks([])
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    axes[1].set_xlim(.5, 7.0); axes[1].set_xlabel("tiempo")
    fig.text(.98, .5, "Sin interpolación ni umbral máximo de antigüedad", rotation=90, ha="right", va="center", fontsize=8, color="#8a3b12")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
