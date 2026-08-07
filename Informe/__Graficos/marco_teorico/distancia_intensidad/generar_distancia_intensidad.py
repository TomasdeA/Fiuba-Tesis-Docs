#!/usr/bin/env python3
"""Genera la curva de conversión de distancia a intensidad lógica."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "salida/distancia_intensidad.png"


def main():
    zmin, zmax, sensor_max = .25, 3.0, 5.0
    z = np.linspace(0, 5.2, 600)
    value = 100 * np.clip((zmax-z)/(zmax-zmin), 0, 1)
    example_z = 1.5; example_i = 100 * (zmax-example_z)/(zmax-zmin)
    fig, ax = plt.subplots(figsize=(8.5, 4.4)); ax.plot(z, value, lw=2)
    ax.axvspan(0, zmin, color="0.8", alpha=.25, label="fuera del rango perceptivo")
    ax.axvspan(zmax, sensor_max, color="C2", alpha=.10, label="medición válida, intensidad nula")
    ax.axvline(zmin, ls="--", color="gray"); ax.axvline(zmax, ls="--", color="gray")
    ax.axvline(sensor_max, ls=":", color="C2"); ax.scatter([example_z], [example_i], color="C3", zorder=3)
    label = f"Ejemplo: $d={example_z:.1f}$ m, $I={example_i:.1f}$".replace(".", ",")
    ax.annotate(label, (example_z, example_i), xytext=(1.85, 69), arrowprops=dict(arrowstyle="->", color="C3"), color="C3")
    ax.annotate(r"$z_{\min}^{\mathrm{háptico}}=0{,}25$ m", (zmin, 4), xytext=(.48, 18), arrowprops=dict(arrowstyle="->"))
    ax.annotate(r"$z_{\max}^{\mathrm{háptico}}=3{,}0$ m", (zmax, 4), xytext=(3.18, 18), ha="left", arrowprops=dict(arrowstyle="->"))
    ax.text(sensor_max-.04, 7, "límite perceptivo: 5 m", rotation=90, ha="right", va="bottom", color="C2")
    ax.set(xlabel="Distancia mínima válida de la celda [m]", ylabel="Intensidad lógica", xlim=(0,5.2), ylim=(-3,105))
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
