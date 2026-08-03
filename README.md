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

### Módulo 4 · Recomendador por canción
- Servicio no supervisado basado en `recording_group_id` con features acústicas (R1 aprobado: StandardScaler + coseno + brute).
- Artefactos versionados en `models/recommender/v1/` (`scaler.joblib`, `neighbors.joblib`, `catalog_matrix.npy`, `catalog_index.parquet`, `manifest.json`).
- Flujo online con recuperación progresiva, exclusión de la propia grabación, deduplicación de grupos y orden estable.
- Filtros desactivados por defecto (explícito, género, duración, artista, popularidad mínima).
- Explicaciones por característica y diferencia de tempo en BPM.
- Evaluación offline (§14.11) en `reports/metrics/track_recommender_evaluation.json/.csv` (muestra de 200 semillas: 0 autorrecomendaciones, 0 duplicados, 100 % cumplimiento de filtros).
- Comparación R1–R4 en `reports/experiments/track_recommender_r1_r4_comparison.json`.
- Notebook `notebooks/04_track_recommender_experiments.ipynb`.

### Módulo 5 · Recomendador por preferencias
- Presets editables en `configs/presets.yaml` (7 presets con valores y pesos 0–3, §32.6) sin tocar Python.
- Distancia euclídea ponderada sobre variables escaladas (§15.6): el peso 0 ignora la variable y se rechazan los perfiles con todos los pesos en cero.
- Modo manual con pesos por variable (0–3) y modo preset.
- Detección de perfiles fuera de distribución (§15.8) por distancia al centroide del catálogo (percentiles p95/p99 precomputados).
- Diversidad MMR opcional (§15.9) con `lambda = 0,85`; el ranking puro es el predeterminado.
- Artefactos versionados en `models/preferences/v1/`.
- Notebook `notebooks/05_preference_recommender_experiments.ipynb`.

### Módulo 6 · Clasificador multietiqueta de géneros
- Dataset multilabel con una fila por `recording_group_id` y matriz binaria `[n, 114]` (§16.2).
- Características primarias §16.3 (`log_duration`, `key_sin/cos`, `mode`, `time_signature` one-hot fija) sin columnas prohibidas de identidad.
- Split agrupado 70/15/15 congelado en `data/processed/splits.parquet` (hash `7cdb3f…b67`), intersecciones vacías y selección por menor desviación de prevalencia (§16.4).
- Experimentos A (excluye `audio_analysis_incomplete`) y B (imputación con medianas de train + indicador, §16.5).
- Modelos M0 (frecuencia) y M1 (OneVsRest logistic, liblinear C=1.0) entrenados; M2 (ClassifierChain ×3), M3 (Extra Trees 400) y M4 (Random Forest 400) listos para comparar con `scripts/compare_models.py`; M5/XGBoost deshabilitado (§16.6).
- Umbral global 0.10–0.90 optimizando samples F1 solo sobre validación (§16.8); la app muestra Top-5 con aviso de umbral no superado.
- Métricas §16.9 (macro/micro/samples F1, Hamming, precision@3, recall@5, hit@3/5, coverage, LRAP, AP por etiqueta) en `reports/experiments/classifier_multilabel_comparison.json`.
- `scripts/evaluate_final_model.py` evalúa el test congelado únicamente con `--use-test` (§16.4).
- Notebook `notebooks/06_multilabel_classifier.ipynb`.

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
