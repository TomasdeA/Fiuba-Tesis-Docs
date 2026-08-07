# Codificación angular de la nube filtrada

Esta carpeta reproduce con datos reales la asignación implementada por
`obstacle_grid_encoder`. La figura contiene:

- la imagen de profundidad que origina la nube;
- una vista superior de la nube coloreada por columna angular;
- una vista lateral coloreada por fila métrica;
- la `DepthGrid` 5x10 obtenida en el mismo instante.

El script aplica los mismos filtros, la grilla angular intermedia, la reducción a diez
columnas, las cinco bandas verticales y la distancia radial `sqrt(x²+z²)`. Finalmente compara
`count`, `min_m`, `mean_m` y `max_m` de todas las celdas con el mensaje publicado.

## Regenerar la figura incluida

```bash
source /opt/ros/humble/setup.bash
source /home/tomasdea/Tesis/install/setup.bash
cd /home/tomasdea/Tesis/docs/Informe

python3 __Graficos/marco_teorico/codificacion_angular/generar_codificacion_angular.py \
  __Graficos/marco_teorico/codificacion_angular/rosbag \
  --cloud-index 30
```

La salida se escribe en `salida/codificacion_angular.png`. Solo debe utilizarse si el comando
termina con `Verificación con DepthGrid: OK`.

## Obtener una nueva captura

Dentro del contenedor:

```bash
nav-start-v1 use_hw:=false realsense_depth_profile:=1280x720x15
```

En otra terminal del contenedor:

```bash
ros2 bag record \
  -o /tmp/codificacion_angular \
  /camera/camera/depth/image_rect_raw \
  /depth_obstacle_filter/obstacle_cloud \
  /perception/depth_grid \
  /perception/depth_grid/aperture_state_deg \
  /depth_obstacle_filter/camera_height
```

Si no se publica una nueva altura válida durante la captura, el script usa el límite vertical
predeterminado de 2,20 m, tal como hace `obstacle_grid_encoder`. Los demás valores
predeterminados corresponden a `obstacle_grid_encoder.yaml` y pueden modificarse mediante la
ayuda del comando:

```bash
python3 __Graficos/marco_teorico/codificacion_angular/generar_codificacion_angular.py --help
```
