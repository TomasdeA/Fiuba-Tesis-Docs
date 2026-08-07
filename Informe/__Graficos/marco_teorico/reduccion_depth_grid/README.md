# Figura de reducción de imagen de profundidad a `DepthGrid`

Esta carpeta contiene una herramienta para generar, a partir de datos reales grabados con
ROS 2, la figura solicitada en el Marco Teórico. La salida muestra tres representaciones del
mismo instante:

1. la imagen de profundidad con la división en regiones de interés de 5 filas por 10 columnas;
2. una ROI ampliada con las estadísticas publicadas para esa celda;
3. la `DepthGrid` resultante, expresada en metros y sin aplicar el mapeo háptico.

El script no utiliza `output_viewer`, porque esa herramienta representa `HapticGrid` después
de convertir distancia en intensidad. Tampoco utiliza RViz ni una nube de puntos, ya que la
ruta v0 reduce directamente la imagen de profundidad.

## Captura incluida en el informe

La subcarpeta `rosbag_1280x720/` contiene la captura real de alta resolución utilizada en la
Figura 3.2 del informe. La
figura puede regenerarse exactamente, desde la carpeta `Informe`, mediante:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  __Graficos/marco_teorico/reduccion_depth_grid/rosbag_1280x720 \
  --grid-index 30 \
  --roi-row 2 \
  --roi-col 5 \
  --output __Graficos/marco_teorico/reduccion_depth_grid/salida/reduccion_depth_grid.png
```

La imagen de entrada tiene una resolución de 1280x720 píxeles. La imagen y la grilla
seleccionadas presentan una diferencia temporal de registro de 8,12 ms.
El script verificó que `count`, `min_m`, `mean_m` y `max_m` de la ROI coincidieran con los
valores recalculados desde la imagen.

## 1. Preparar el entorno

Abrir una terminal y cargar ROS 2 y el workspace compilado:

```bash
source /opt/ros/humble/setup.bash
source /home/tomasdea/Tesis/install/setup.bash
```

El script requiere los paquetes Python de ROS 2, NumPy y Matplotlib. Puede verificarse con:

```bash
python3 -c "import rosbag2_py, rclpy, numpy, matplotlib, custom_interfaces"
```

## 2. Ejecutar la ruta v0

En una terminal con el entorno cargado, iniciar el sistema en modo `raw`:

```bash
ros2 launch nav_bringup nav.launch.py pipeline_mode:=raw
```

Antes de grabar, comprobar que existen los dos tópicos necesarios:

```bash
ros2 topic info /camera/camera/depth/image_rect_raw
ros2 topic info /perception/depth_grid
```

Los tipos esperados son, respectivamente:

```text
sensor_msgs/msg/Image
custom_interfaces/msg/DepthGrid
```

## 3. Grabar el rosbag

Desde otra terminal con el mismo entorno cargado:

```bash
mkdir -p /home/tomasdea/Tesis/rosbags
ros2 bag record \
  -o /home/tomasdea/Tesis/rosbags/reduccion_depth_grid \
  /camera/camera/depth/image_rect_raw \
  /perception/depth_grid
```

Mantener una escena representativa durante algunos segundos y finalizar la grabación con
`Ctrl+C`. Conviene incluir un obstáculo que ocupe solo parte de alguna ROI y evitar mover la
cámara bruscamente.

Verificar el contenido:

```bash
ros2 bag info /home/tomasdea/Tesis/rosbags/reduccion_depth_grid
```

Si el directorio de salida ya existe, elegir otro nombre para no mezclar grabaciones.

### Reproducir la grabación para inspeccionarla

El script de la sección siguiente lee el rosbag directamente, por lo que no hace falta
reproducirlo para generar la figura. Si se quiere inspeccionar la grabación con herramientas
ROS 2, detener primero la publicación de la cámara en vivo y ejecutar:

```bash
ros2 bag play /home/tomasdea/Tesis/rosbags/reduccion_depth_grid --clock
```

Mientras se reproduce, pueden verificarse los mensajes con:

```bash
ros2 topic echo /perception/depth_grid --once
```

No se debe ejecutar al mismo tiempo otra instancia de `depth_to_matrix` alimentada por una
cámara en vivo, porque se mezclarían mensajes de dos escenas sobre el mismo tópico.

## 4. Generar la figura

Desde la carpeta del informe:

```bash
cd /home/tomasdea/Tesis/docs/Informe
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  /home/tomasdea/Tesis/rosbags/reduccion_depth_grid \
  --grid-index 0 \
  --roi-row 2 \
  --roi-col 5
```

La salida predeterminada se escribe en:

```text
__Graficos/marco_teorico/reduccion_depth_grid/salida/reduccion_depth_grid.png
```

`--grid-index` selecciona, desde cero, qué mensaje `DepthGrid` se utiliza. Para probar otros
instantes, repetir el comando con valores como `30`, `60` o `100`. La ROI resaltada se elige
con `--roi-row` y `--roi-col`; para una grilla 5x10, sus rangos son 0--4 y 0--9.

La matriz muestra `min_m` de manera predeterminada. Esto puede cambiarse sin modificar el
rosbag:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  /home/tomasdea/Tesis/rosbags/reduccion_depth_grid \
  --grid-index 30 \
  --roi-row 2 \
  --roi-col 5 \
  --stat mean_m \
  --output __Graficos/marco_teorico/reduccion_depth_grid/salida/reduccion_media.png
```

Los valores posibles para `--stat` son `min_m`, `mean_m` y `max_m`. Para la figura conceptual
del Marco Teórico alcanza con utilizar uno; no es necesario presentar allí una comparación.

### Comparación utilizada en el capítulo de Método

La comparación entre los tres estadísticos se genera a partir del mismo mensaje mediante:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_comparacion_estadisticos.py \
  __Graficos/marco_teorico/reduccion_depth_grid/rosbag_1280x720 \
  --grid-index 30 \
  --roi-row 2 \
  --roi-col 5
```

La salida se escribe en `salida/comparacion_estadisticos.png`. Las tres matrices utilizan la
misma escala de color; por lo tanto, las diferencias visuales provienen del estadístico y no
de una normalización independiente de cada panel.

## 5. Límites de profundidad y sincronización

Los valores predeterminados reproducen la configuración usual de v0:

```text
--z-min 0.25
--z-max 5.0
```

Si la ejecución utilizó otros parámetros, deben pasarse también al script. Por ejemplo:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  /home/tomasdea/Tesis/rosbags/reduccion_depth_grid \
  --z-min 0.30 \
  --z-max 3.50
```

Actualmente `depth_to_matrix` no copia el `header` de la imagen al mensaje `DepthGrid`. Por
este motivo, el script empareja ambos mensajes mediante sus tiempos de registro en el rosbag.
La diferencia máxima permitida es 100 ms y puede ajustarse con `--max-sync-ms`. No conviene
aumentarla sin revisar visualmente la escena, porque podrían combinarse instantes diferentes.

El script recalcula las estadísticas de la ROI con los límites indicados y las compara con las
publicadas en `DepthGrid`. Si no coinciden, imprime una advertencia y la figura marca
`REVISAR SINCRONIZACIÓN`. En ese caso se debe probar otro `--grid-index`, verificar los límites
o revisar los tópicos grabados; no debe utilizarse esa figura en el informe.

## 6. Rosbags MCAP o tópicos remapeados

Para un rosbag MCAP:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  /ruta/al/rosbag \
  --storage-id mcap
```

Si los tópicos tienen otros nombres:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py \
  /ruta/al/rosbag \
  --depth-topic /otro/topico/depth \
  --grid-topic /otro/topico/depth_grid
```

La ayuda completa está disponible con:

```bash
python3 __Graficos/marco_teorico/reduccion_depth_grid/generar_reduccion_depth_grid.py --help
```
