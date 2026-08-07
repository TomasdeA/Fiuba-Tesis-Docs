#!/usr/bin/env python3
"""Genera el ejemplo de acumulación log-odds y publicación OccupancyGrid."""

from pathlib import Path
import math
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "salida/ejemplo_log_odds.png"


def main():
    labels = ["sin observaciones", "+ ocupada", "+ libre", "+ libre", "+ libre"]
    values = [0, 2.197, 2.197-.847, 2.197-2*.847, 2.197-3*.847]
    probs = [1/(1+math.exp(-value)) for value in values]
    fig, ax = plt.subplots(figsize=(8, 4.4)); ax.plot(range(5), values, "o-")
    ax.axhline(0, color="gray", ls="--"); ax.set_xticks(range(5), labels); ax.set_ylabel("Log-odds de ocupación $l$")
    nav_values = [-1] + [int(probability*100+.5) for probability in probs[1:]]
    for index, (value, probability, message) in enumerate(zip(values, probs, nav_values)):
        published = f"OccupancyGrid.data[i]={message}" + (" (desconocida)" if message == -1 else "")
        ax.text(index, value+.15, f"P={probability:.2f}\n{published}", ha="center", fontsize=8)
    ax.grid(alpha=.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"Gráfico escrito en {OUT}")


if __name__ == "__main__":
    main()
