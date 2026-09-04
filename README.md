---
title: Sentinel Gait
emoji: 🚶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PhysioSentinel Gait Online v0.10.9

Versión centrada en mejorar la coherencia biomecánica de las métricas temporales 2D.

Principales cambios: exclusión automática de giro/transiciones, CV y asimetría derivados de contactos
I/D validados, doble apoyo restringido a ciclos válidos y controles internos que suprimen resultados
incoherentes en lugar de mostrarlos como fiables.

Las métricas markerless 2D son experimentales y deben integrarse con la observación clínica.


## v0.10.9 · Modo multipersona
Selección manual del paciente + seguimiento de identidad bloqueado + rechazo de frames ambiguos. Pensado para marcha con acompañante, supervisión estrecha o asistencia física.


## v0.10.9 · CV robusto por lado
CV izquierdo y derecho calculados por separado, rechazo robusto de ciclos atípicos, CV global ponderado y tamaño muestral explícito.

## v0.10.9 · Coherencia temporal fuerte
Cadencia desde una línea temporal anatómica L-R, control independiente por ciclos
ipsilaterales y cierre físico apoyo/doble apoyo. Las métricas temporalmente
incompatibles se suprimen en lugar de publicarse con falsa precisión.

## v0.10.9 · Metadatos editables
Paciente/código, nombre del registro, edad, sexo y fecha del registro pueden modificarse
desde un formulario sin reprocesar el vídeo ni perder los resultados biomecánicos.

## v0.10.9 · Exportación completa
Nueva pestaña **Exportar / Descargar** con CSV, JSON, informes, gráficos PNG,
datos fuente y vídeos anotados. El paquete ZIP se crea bajo demanda en memoria,
sin guardar contenido pesado en Supabase.

## v0.10.9 · Analizador interactivo del ciclo
Nueva pestaña con vídeo/tracking sincronizado con curvas 0–100% y bandas de fases
por extremidad. Permite revisar y corregir IC/TO por ciclo sin repetir Pose2Sim.

## v0.10.9 · Ciclo robusto
La pestaña de ciclo conserva en `session_state` el segmento exacto usado para
los resultados, FPS y resúmenes IC/TO antes de eliminar archivos temporales.
También reutiliza el vídeo anotado almacenado en memoria para sincronizar
vídeo + tracking + fases incluso después de la limpieza.

## v0.10.9 · Corrección pandas
Se corrige la evaluación booleana ambigua de DataFrames en la pestaña
**9 · Ciclo de marcha**. Los snapshots se seleccionan ahora mediante comprobaciones
explícitas y seguras.

## v0.10.9 · Sujeto bloqueado + panel compacto
La pestaña de ciclo dibuja únicamente el sujeto manualmente seleccionado sobre
el vídeo limpio conservado en memoria de sesión. El vídeo se recorta alrededor
del paciente y se muestra junto a la banda de fases en un panel de dos columnas.

## v0.10.9 · Ciclo compacto
Corrige la conversión entre posiciones internas del segmento y frames reales de
vídeo, recupera las curvas cinemáticas 0–100% y concentra vídeo, fases y curva
en un panel compacto.

## v0.10.9 · Cadencia y ciclo
Corrige el error de fase en la pestaña 9 y desacopla la estimación de cadencia
del control físico de doble apoyo. La cadencia puede publicarse con una etiqueta
de calidad aunque el cierre global de apoyo falle.

## v0.10.9 · Ritmo cinemático + scrubber
Cadencia, CV y asimetría pueden estimarse desde alternancia distal cuando las
máscaras IC/TO fallan. La pestaña de ciclo incorpora una barra única bajo el
vídeo que sincroniza frame, fases y curvas.

## v0.10.9 · Timeline clínico
La pestaña de ciclo usa ahora una barra temporal en segundos bajo el vídeo y un
mapa de fases izquierda/derecha con el mismo eje temporal. El cursor, vídeo,
fases y curvas se actualizan con un único control.

## v0.10.9 · Reproducción y alineación
Mapa de fases y cinemática comparten ahora el eje temporal en segundos. El
reproductor permite revisar el vídeo con fases sobreimpresas a 0.25×, 0.5×,
1×, 1.5× y 2×. La vista 0–100% queda como análisis secundario.

## v0.10.9
QC temporal IC/TO y fallback distal bilateral robusto.
