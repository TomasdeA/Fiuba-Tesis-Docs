# Documentación de tesis

Repositorio con el material escrito de la tesis **"Navegación asistida para ciegos mediante mapeo del entorno"** de Ingeniería Electrónica, Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA).

Este repo reúne el informe final, la presentación, bibliografía, figuras, tablas, notas de desarrollo y documentos auxiliares usados durante la elaboración del trabajo.

## Documentos principales

- **Informe de tesis:** [Informe/main.pdf](Informe/main.pdf)
- **Presentación / plan de trabajo:** [Presentacion/main.pdf](Presentacion/main.pdf)
- **Acta de acuerdo:** [Presentacion/Acta-de-acuerdo.pdf](Presentacion/Acta-de-acuerdo.pdf)

## Estructura del repositorio

```text
.
├── Informe/        # Informe final en LaTeX y PDF generado
├── Presentacion/   # Presentación, plan y documentos administrativos
├── README.md       # Descripción general del repositorio
```

Dentro de `Informe/`:

```text
Informe/
├── main.tex             # Documento principal
├── main.pdf             # PDF generado
├── Desarrollo/          # Capítulos y secciones del informe
├── Notas/               # Notas técnicas y borradores de desarrollo
├── _Imagenes/           # Figuras e imágenes
├── _Tablas/             # Tablas
├── __Graficos/          # Diagramas fuente
├── __Header/            # Paquetes, comandos y carátula
└── bibliografia.bib     # Bibliografía
```

## Compilación

Los documentos están escritos en LaTeX. Para regenerar el informe:

```bash
cd Informe
latexmk -pdf main.tex
```

Para regenerar la presentación:

```bash
cd Presentacion
latexmk -pdf main.tex
```

Si no se usa `latexmk`, también puede compilarse con una cadena LaTeX/Biber equivalente, ya que los documentos usan bibliografía en `bibliografia.bib`.

## Relación con el desarrollo

La implementación del sistema se mantiene en el repositorio `nav_mapper`. Este repo documenta el problema, el estado del arte, el diseño de la solución, la arquitectura ROS 2, los experimentos, resultados, discusión y conclusiones.

## Contenido del informe

El informe principal incluye:

- Introducción y motivación.
- Estado del arte.
- Marco teórico.
- Arquitectura del sistema.
- Diseño del método y pipeline de percepción.
- Implementación en ROS 2.
- Metodología experimental.
- Resultados.
- Discusión, conclusiones y trabajo futuro.
- Anexos y trazabilidad.

## Notas de trabajo

Los archivos en `Informe/Notas/` contienen análisis, borradores y material auxiliar. Pueden estar menos pulidos que el informe principal, pero sirven como registro de decisiones técnicas y exploración durante el desarrollo.
