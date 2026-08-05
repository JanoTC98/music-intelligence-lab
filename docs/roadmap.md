# Roadmap

## Completado

- Ingesta, auditoría y validación de datos.
- Limpieza, catálogo e identidad (consolidación por `track_id` y huella exacta).
- Análisis exploratorio.
- Recomendador por canción (basado en vecinos y distancias).
- Recomendador por preferencias (distancia ponderada).
- Laboratorio supervisado de clasificación multietiqueta de géneros.
- Experimento multiclase de género acústico dominante.
- Aplicación web multipágina en Streamlit.
- Pipeline reproducible de datos y modelos.
- Repositorio profesional con pruebas, documentación y despliegue.

## En curso

- Documentación detallada por dominio (`docs/spec/`).
- Proceso de empaquetado limpio (`scripts/package_source.py`).

## Pendiente

- Especificaciones adicionales según necesidad del proyecto.

## Opcional

- Persistencia opcional de eventos y feedback en MySQL. Esquemas en `database/` y especificación en `docs/spec/persistence.md`, pero el módulo **no está implementado** ni forma parte del alcance actual. Requeriría MySQL local y no afecta a la aplicación desplegada.