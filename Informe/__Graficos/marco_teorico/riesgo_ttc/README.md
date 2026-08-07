# Esquema de riesgo y TTC

El script genera una vista superior conceptual de la geometría implementada por
`spatial_awareness`. Lee los umbrales y el tamaño de celda vigentes desde:

```text
develop/nav_mapper/src/spatial_awareness/config/params.yaml
develop/nav_mapper/src/local_mapper/config/params.yaml
```

La pose, la velocidad y el obstáculo son un ejemplo geométrico explícito para
mostrar la proyección de la velocidad, la distancia y el TTC; no representan
resultados experimentales.

El esquema adopta el criterio previsto de medir `d` desde el contorno corporal
de radio 0,20 m hasta el centro de la celda ocupada. Debe incorporarse el mismo
criterio en `RiskEvaluator` antes de presentar la figura como representación de
la implementación vigente.

Desde `docs/Informe`:

```bash
MPLCONFIGDIR=/tmp/matplotlib-riesgo \
python3 __Graficos/marco_teorico/riesgo_ttc/generar_riesgo_ttc.py
```

La salida se escribe en:

```text
__Graficos/marco_teorico/riesgo_ttc/salida/riesgo_ttc.png
```

La validación experimental debe realizarse por separado en RViz mediante
`/local_mapper/occupancy_grid`, `/spatial_awareness/debug/markers`, `odom` y
`nav_odom`.
