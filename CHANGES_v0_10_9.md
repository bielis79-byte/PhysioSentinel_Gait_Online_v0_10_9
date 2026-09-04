# PhysioSentinel Gait Online v0.10.9

## Corrección IC/TO
La auditoría mostró fases visualmente incorrectas pese a una sincronización gráfica correcta.

La v0.10.9:
- mantiene el detector original como primera opción;
- aplica controles amplios de coherencia temporal;
- cuando el original falla, intenta una cadena distal robusta;
- usa tobillo + talón + antepié;
- impone alternancia bilateral L-R-L-R;
- filtra intervalos anómalos mediante mediana/MAD;
- reconstruye ciclos ipsilaterales;
- limita TO a una ventana amplia de 45–80% del ciclo.

Se exportan `support_event_source` y `support_event_quality`.

Los eventos siguen siendo estimaciones cinemáticas 2D y deben validarse externamente frente a mocap/plataforma de fuerza.
