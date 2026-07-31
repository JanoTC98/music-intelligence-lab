# Spotify Music Intelligence

Proyecto end-to-end de ciencia de datos y machine learning que transforma un catálogo musical en:

1. Un recomendador basado en una canción.
2. Un recomendador basado en preferencias y presets editables.
3. Un laboratorio supervisado de clasificación multietiqueta de géneros.
4. Un experimento multiclase de género acústico dominante.
5. Una aplicación web multipágina en Streamlit.
6. Un pipeline reproducible de datos y modelos.
7. Un sistema opcional de eventos y feedback almacenado en MySQL.
8. Un repositorio profesional con pruebas, documentación, CI y despliegue.

## Estado actual

| Módulo | Estado |
|---|---|
| 0 — Inicialización y entorno | Completado |
| 1 — Ingesta, auditoría y validación | Completado |
| 2 — Limpieza, catálogo e identidad | Pendiente |
| 3 — Análisis exploratorio | Pendiente |
| 4 — Recomendador por canción | Pendiente |
| 5 — Recomendador por preferencias | Pendiente |
| 6 — Clasificador multietiqueta | Pendiente |
| 7 — Clasificador multiclase | Pendiente |
| 8 — Aplicación Streamlit | Pendiente |
| 9 — Persistencia MySQL | Pendiente |

## Requisitos

- Python 3.12.x
- [uv](https://docs.astral.sh/uv/)

## Instalación

```powershell
uv sync
uv run python -c "import spotify_intelligence"
```

## Calidad

```powershell
uv run ruff check .
uv run pytest
```

## Estructura

- `src/spotify_intelligence/` — código importable del proyecto.
- `configs/` — configuración versionada (YAML).
- `data/` — dataset bruto, cuarentena, intermedios y procesados.
- `tests/` — pruebas unitarias e integración.
- `notebooks/` — exploración y experimentos.

## Documentación

La especificación normativa completa está en `AGENTS.md`.
