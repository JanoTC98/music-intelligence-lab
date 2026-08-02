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

## Funcionalidad implementada

### Módulo 1 · Ingesta, auditoría y validación
- Auditoría reproducible del catálogo: validación de columnas, tipos, rangos y anomalías del dataset.
- Reporte de calidad en `reports/data_quality/data_quality_report.json` con hash SHA-256 del archivo bruto.
- Notebook `notebooks/01_data_audit.ipynb`.

### Módulo 2 · Limpieza, catálogo e identidad
- Pipeline reproducible `scripts/prepare_data.py` con manifiesto en `data/processed/prepare_data_manifest.json`.
- Cuarentena de identidad inválida en `data/quarantine/`.
- Catálogo consolidado por `track_id` (`tracks.parquet`) con popularidad min/max/mediana.
- Tablas puente de géneros, artistas y grabaciones (`track_genres`, `track_artists`, `recording_tracks`, `recording_genres`, `genre_catalog`).
- `recording_group_id` exacto (huella normalizada + SHA-256) con 83.881 grupos verificados.
- Catálogo canónico por grabación (`recordings.parquet`) con track representativo.
- Reporte de candidatos a casi duplicados sin fusión automática (`reports/identity/near_duplicate_candidates.csv`).
- Notebook `notebooks/02_identity_analysis.ipynb`.

### Módulo 3 · Análisis exploratorio
- Distribuciones, anomalías y duraciones extremas.
- Correlaciones y redundancia de características.
- Solapamiento y co-ocurrencia de géneros.
- Figuras exportadas a `reports/figures/`.
- Notebook `notebooks/03_exploratory_analysis.ipynb`.

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
