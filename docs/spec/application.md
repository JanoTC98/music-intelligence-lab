# Aplicación Streamlit

## Estructura

- `app/streamlit_app.py` — Router multipágina.
- `app/pages/` — Páginas individuales (home, recomendación por canción, recomendación por preferencias, laboratorio multietiqueta, laboratorio multiclase, auditoría de datos, metodología).
- `app/components/` — Componentes reutilizables (cards, charts, filters, tables, messages, resources).

## Reglas

- Streamlit no reentrena modelos durante una interacción.
- Artefactos ausentes producen un error claro, no entrenamiento silencioso.
- La aplicación no contiene lógica de producción; delega a `src/spotify_intelligence/`.