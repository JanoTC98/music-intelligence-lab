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

**Aplicación desplegada:** [https://music-intelligence-lab.streamlit.app/](https://music-intelligence-lab.streamlit.app/)

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
- Evaluación offline en `reports/metrics/track_recommender_evaluation.json/.csv` (muestra de 200 semillas: 0 autorrecomendaciones, 0 duplicados, 100 % cumplimiento de filtros).
- Variante experimental de **afinidad de género** como toggle desactivado por defecto en la app (botón "Priorizar canciones del mismo género de la semilla"); reordena el pool acústico recuperado para que los candidatos que comparten género con la semilla salgan primero, sin añadir candidatos fuera del pool.
- Análisis de relevancia (`scripts/analyze_recommender_relevance.py`, 2000 semillas) en `reports/experiments/recommender_relevance_analysis.json` — evaluado con las métricas oficiales:

  | Variante | Coherencia@10 media | Coseno medio | Diversidad interna (std) | Artistas únicos por lista |
  |---|---|---:|---:|---:|
  | Baseline | 0,131 | 0,9705 | 0,0065 | 0,948 |
  | Pesos ajustados | 0,131 | 0,9607 | 0,0140 | 0,950 |
  | Afinidad de género | **0,552** | 0,9605 | 0,0131 | 0,906 |

  La afinidad cuadruplica la coherencia de género con −0,01 de similitud, casi sin perder variedad de artistas y duplicando la diversidad interna de puntuaciones; los pesos por característica no mejoran nada. La ganancia es variable por género (disponibilidad media del género en los 100 vecinos acústicos: 0,11); rock/alternativo mejoran menos (rock 0,06 → 0,41).
- Comparación R1–R4 en `reports/experiments/track_recommender_r1_r4_comparison.json`.
- Notebook `notebooks/04_track_recommender_experiments.ipynb`.

### Módulo 5 · Recomendador por preferencias
- Presets editables en `configs/presets.yaml` (7 presets con valores y pesos 0–3) sin tocar Python.
- Distancia euclídea ponderada sobre variables escaladas: el peso 0 ignora la variable y se rechazan los perfiles con todos los pesos en cero.
- Modo manual con pesos por variable (0–3) y modo preset.
- Detección de perfiles fuera de distribución por distancia al centroide del catálogo (percentiles p95/p99 precomputados).
- Diversidad MMR opcional con `lambda = 0,85`; el ranking puro es el predeterminado.
- Artefactos versionados en `models/preferences/v1/`.
- Notebook `notebooks/05_preference_recommender_experiments.ipynb`.

### Módulo 6 · Clasificador multietiqueta de géneros
- Dataset multilabel con una fila por `recording_group_id` y matriz binaria `[n, 114]`.
- Características primarias (`log_duration`, `key_sin/cos`, `mode`, `time_signature` one-hot fija) sin columnas prohibidas de identidad.
- Split agrupado 70/15/15 congelado en `data/processed/splits.parquet` (hash `7cdb3f…b67`), intersecciones vacías y selección por menor desviación de prevalencia.
- Experimentos A (excluye `audio_analysis_incomplete`) y B (imputación con medianas de train + indicador); B no mejora a A (M1: samples F1 0.1395 vs 0.1386).
- Modelos M0 (frecuencia) y M1 (OneVsRest logistic, liblinear C=1.0) entrenados; M2 (ClassifierChain ×3), M3 (Extra Trees 400) y M4 (Random Forest 400) entrenados; M5/XGBoost deshabilitado.
- Umbral global 0.10–0.90 optimizando samples F1 solo sobre validación; la app muestra Top-5 con aviso de umbral no superado.
- Métricas (macro/micro/samples F1, Hamming, precision@3, recall@5, hit@3/5, coverage, LRAP, AP por etiqueta) en `reports/experiments/classifier_multilabel_comparison.json`.
- Comparación final (validación): M0=0.0, M1=0.1386, M2=0.0718, M3=0.2522, M4=0.2867 samples F1. M3/M4 ganan en calidad pero son inviables para la app (22–28 GB y 78–90 s de latencia por consulta).
- Modelo final aprobado por el propietario: **M1 (OneVsRest logistic)** por viabilidad en Streamlit (0,1 MB, ~135 ms). Evaluación única sobre test congelado en `reports/metrics/multilabel_final_test_evaluation.json` (samples F1 test=0.1360, macro F1=0.1377, hit@5=0.4573, Hamming=0.0352), consistente con validación (0.1386). *Nota: una primera versión de este reporte registró 0.0314 por un doble-escalado (`scaler.transform` sobre un test ya escalado); corregido en `scripts/evaluate_final_model.py` y re-evaluado.*
- `scripts/evaluate_final_model.py` evalúa el test congelado únicamente con `--use-test`.
- Notebook `notebooks/06_multilabel_classifier.ipynb`.

### Módulo 7 · Clasificador multiclase de género dominante
- Dataset de grabaciones monoetiqueta: 69.943 grupos; split agrupado congelado train 48.987 / validación 10.496 / test 10.460; 112/114 clases en train (conteos 47–717).
- Modelos C0 (clase frecuente), C1 (logística lbfgs C=1.0), C2 (Extra Trees) y C3 (Random Forest) con `max_depth=12` y `n_estimators=300`.
- Comparación (validación): C0 acc 0.0120, C1 acc 0.2247, C2 acc 0.2527, C3 acc 0.2866 (macro F1 0.0002 / 0.1627 / 0.1735 / 0.2097). Reporte en `reports/experiments/classifier_multiclass_comparison.json`.
- C2/C3 ganan en calidad pero pesan ~650 MB cada uno; el propietario aprobó **C1** como modelo final (~0,5 MB, viable en Streamlit). C2/C3 se eliminaron tras consolidar el reporte.
- Métricas sobre validación en `models/classifier/multiclass/<id_C1>/metrics_validation.json`.
- Evaluación exploratoria sobre filas multigénero: validación Hit@1=0.195, Hit@3=0.393, Recall@5=0.284 (2.086 filas); test Hit@1=0.179, Hit@3=0.366, Recall@5=0.263 (2.123 filas).
- Evaluación única sobre test congelado en `reports/metrics/multiclass_final_test_evaluation.json` (accuracy test=0.2202, macro F1=0.1606), consistente con validación.
- `scripts/evaluate_final_multiclass_model.py` evalúa el test congelado únicamente con `--use-test`.
- Notebook `notebooks/07_multiclass_classifier.ipynb`.

### Módulo 8 · Aplicación Streamlit
- Aplicación multipágina (`streamlit_app.py` como router) con 7 páginas: Inicio, Auditoría y catálogo, Recomendar por canción, Recomendar por preferencias, Laboratorio multietiqueta, Laboratorio de género dominante y Metodología y limitaciones.
- Carga de datos con `st.cache_data` y de modelos/escaladores/índices con `st.cache_resource`.
- La app **no entrena** ante artefactos ausentes: muestra el mensaje "El artefacto requerido no existe. Ejecute el script de construcción correspondiente".
- Buscador desambiguado por canción + artista + álbum en `src/spotify_intelligence/recommenders/catalog.py`, con contador de coincidencias y prioridad a las canciones del artista buscado.
- El recomendador excluye la propia grabación y también otras grabaciones de la misma canción (mismo título + artistas en otro `recording_group_id`) para no recomendar la misma obra como primera opción.
- Carga de artefactos de clasificación en `src/spotify_intelligence/classification/serving.py` (M1_A/M1_B multietiqueta y C1 multiclase) con selector A/B en el laboratorio multietiqueta.
- Resultados descargables en CSV, explicaciones por característica y avisos de limitaciones; sin branding oficial de Spotify.
- Tracking deshabilitado por defecto (`configs/app.yaml`); Módulo 9 opcional.

## Comparativa de modelos evaluados

### Tabla 0 · Recomendador por canción — R1–R4

| Experimento | Escalador | Distancia | Similitud media | Latencia | Veredicto |
|---|---|---:|---:|---:|---|
| **R1** | Standard | coseno | 0,9794 | 14,6 ms | **Aprobado** (baseline) |
| R2 | Robust | coseno | 0,9680 | 15,4 ms | Cercano a R1, sin ventaja |
| R3 | Standard | euclídea | 0,4969 ¹ | 338,7 ms | Descartado (lentitud) |
| R4 | Robust | euclídea | 0,5696 ¹ | 6,9 ms | Descartado (no comparable) |

¹ La similitud euclídea no es directamente comparable con la coseno (`comparable_similarity=false`); R1 se aprueba por ser el baseline planificado.

Evaluación en producción (200 semillas, R1): 0 autorrecomendaciones · 0 duplicados de grupo · 100 % cumplimiento de filtros · similitud media 0,9729 · cobertura del catálogo 2,35 % · artistas únicos 0,94/lista · diversidad interna (std) 0,0059 · latencia p50 22,2 ms / p95 48,3 ms · estabilidad p50-Jaccard 0,919.

### Tabla 0b · Recomendador por preferencias — servicio desplegado

Sin comparación de modelos: es **no supervisado y determinista**. Distancia euclídea ponderada sobre variables escaladas con pesos 0–3, 7 presets editables en `configs/presets.yaml` (Entrenamiento intenso, Fiesta, Concentración instrumental, Relajación, Alegre y bailable, Melancólico, Acústico), detección de perfiles fuera de distribución (percentiles p95/p99) y diversidad MMR opcional con `lambda = 0,85` — el ranking puro es el predeterminado.

### Tabla 1 · Clasificador multietiqueta — métricas de validación

| Modelo | samples F1 | macro F1 | Hit@5 | Hamming | Tamaño | Latencia | Veredicto |
|---|---:|---:|---:|---:|---:|---:|---|
| M0 · frecuencia | 0,0000 | 0,0000 | 0,052 | 0,0108 | 0,0035 MB | 6,8 ms | Baseline trivial |
| **M1 · OvR logística** | **0,1386** | 0,1406 | 0,463 | 0,0354 | 0,08 MB | 133 ms | **Final** · test: samples F1 0,1360 |
| M2 · Classifier Chain | 0,0718 | 0,0744 | 0,192 | 0,0611 | 0,32 MB | 5.122 ms | No viable (peor que M1, ~38× lento) |
| M3 · Extra Trees 400 | 0,2522 | 0,2619 | 0,574 | 0,0150 | 27,8 GB | 89.978 ms | Inviable: memoria + latencia |
| M4 · Random Forest 400 | 0,2867 | 0,2883 | 0,599 | 0,0157 | 21,7 GB | 78.437 ms | Inviable: memoria + latencia |

M1 con experimento B (imputación de audio incompleto): samples F1 0,1395 → **no mejora a A (0,1386)**, por eso el experimento A es el de producción.

### Tabla 2 · Clasificador multiclase — métricas de validación

| Modelo | Accuracy | macro F1 | Top-5 | Tamaño | Latencia | Veredicto |
|---|---:|---:|---:|---:|---:|---|
| C0 · clase frecuente | 0,0120 | 0,0002 | 0,046 | 0,3 MB | 2,6 ms | Baseline trivial |
| **C1 · logística** | **0,2247** | 0,1627 | 0,491 | 0,5 MB | 35 ms | **Final** · test: accuracy 0,2202 |
| C2 · Extra Trees | 0,2527 | 0,1735 | 0,519 | 690 MB | 2.067 ms | No viable (peso + ~60× lento) |
| C3 · Random Forest | 0,2866 | 0,2097 | 0,562 | 657 MB | 2.171 ms | No viable (peso + ~60× lento) |

> Las latencias corresponden a la predicción sobre el **batch completo de validación** (~10,5–12,6 mil filas en una sola llamada), no a una consulta individual. Por fila: M1 ≈ 10 ms, C1 ≈ 3 ms, M3 ≈ 7 ms, M4 ≈ 6 ms, C2/C3 ≈ 0,2 ms. El criterio de descarte de árboles fue principalmente el tamaño de modelo frente a la memoria de despliegue, no la latencia por fila.

## Requisitos

- Python 3.12.x
- [uv](https://docs.astral.sh/uv/)

## Instalación

```powershell
uv sync
uv run python -c "import spotify_intelligence"
```

## Aplicación web

La app está desplegada en [https://music-intelligence-lab.streamlit.app/](https://music-intelligence-lab.streamlit.app/).
La app carga los artefactos ya construidos; no entrena. Si falta un artefacto, muestra cómo generarlo.

```powershell
uv run streamlit run streamlit_app.py
```

Requisitos previos por página:

| Página | Artefacto necesario | Script |
|---|---|---|
| Recomendar por canción | `models/recommender/v1/` | `scripts/build_recommender.py` |
| Recomendar por preferencias | `models/preferences/v1/` | `scripts/build_preference_recommender.py` |
| Laboratorio multietiqueta | `models/classifier/multilabel/` | `scripts/train_multilabel_classifier.py` |
| Laboratorio de género dominante | `models/classifier/multiclass/` | `scripts/train_multiclass_classifier.py` |

## Despliegue

Instrucciones para publicar en Streamlit Community Cloud (incluida la decisión de
versionar los artefactos `data/processed/` y `models/`, ≈ 100 MB) se detallan en
la documentación de despliegue. La app pública opera con tracking
deshabilitado.

## Calidad

```powershell
uv run ruff check .
uv run pytest
```

## Estructura

- `src/spotify_intelligence/` — código importable del proyecto.
- `app/` — páginas y componentes de la aplicación Streamlit.
- `streamlit_app.py` — router de la aplicación.
- `configs/` — configuración versionada (YAML).
- `data/` — dataset bruto, cuarentena, intermedios y procesados.
- `tests/` — pruebas unitarias e integración.
- `notebooks/` — exploración y experimentos.

## Empaquetado limpio

```powershell
uv run python scripts/package_source.py --dry-run
```
