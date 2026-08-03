# Model Card · C1 · Género acústico dominante (multiclase)

**Proyecto:** Spotify Music Intelligence
**Módulo 7:** Clasificador multiclase secundario (AGENTS.md §17)
**Experiment id:** `20260803-1619_multiclass_C1`
**Artefacto:** `models/classifier/multiclass/20260803-1619_multiclass_C1/`
**Modelo:** Logistic Regression (multiclass `lbfgs`)
**Fecha:** 3 de agosto de 2026

---

## 1. Tarea

Estimar un **género acústico dominante** entre 114 clases a partir de
características acústicas, sobre el subconjunto de grabaciones con una sola
etiqueta original (§17.1, §17.2).

> La predicción es exploratoria y **no reemplaza** las etiquetas originales de
> canciones multigénero (§17.5).

## 2. Datos

- Unidad: `recording_group_id`.
- Población: 69.943 grabaciones monoetiqueta (§17.1).
- Splits agrupados congelados (§16.4): train **48.987** / validación **10.496** / test **10.460**.
- Clases en train: **112 de 114**; conteos por clase 47–717 (desbalance ~15×).
- `split_sha256`: `7cdb3f42c405725ff341667b7ab27d9e928bd82605f599425f11857008d39b67`
- `dataset_sha256`: `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`

## 3. Características (18)

`danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, log_duration, key_sin, key_cos, mode` +
`time_signature` one-hot (§16.3, §32.4).

- Escalado: `StandardScaler` ajustado **solo en train**.
- Experiment: A (excluye `audio_analysis_incomplete`, §16.5).
- Prohibidas: `track_id`, `recording_group_id`, `track_name`, `artists`,
  `album_name`, `track_genre` (§16.3).

## 4. Configuración (§17.3, §32.5)

```yaml
solver: lbfgs
C: 1.0
max_iter: 3000
class_weight: balanced
random_state: 42
```

Convergencia: OK con datos escalados (sin `ConvergenceWarning`).

## 5. Métricas sobre validación (§17.4)

| Métrica | Valor |
|---|---:|
| Accuracy | 0.2247 |
| Macro F1 | 0.1627 |
| Balanced accuracy | 0.1822 |
| Top-3 accuracy | 0.3980 |
| Top-5 accuracy | 0.4914 |

Fuente: `models/classifier/multiclass/20260803-1619_multiclass_C1/metrics_validation.json`.

## 6. Evaluación exploratoria sobre filas multigénero (§17.5)

| Conjunto | Hit@1 | Hit@3 | Recall@5 | Filas |
|---|---:|---:|---:|---:|
| Validación | 0.1946 | 0.3931 | 0.2836 | 2.086 |
| Test | 0.1790 | 0.3665 | 0.2631 | 2.123 |

Fuentes: notebook `07_multiclass_classifier.ipynb` y `reports/metrics/multiclass_final_test_evaluation.json`.

## 7. Evaluación final única sobre test (§17.4)

Ejecutada una sola vez con `scripts/evaluate_final_multiclass_model.py --use-test`.

| Métrica | Test | Validación |
|---|---:|---:|
| Accuracy | **0.2202** | 0.2247 |
| Macro F1 | **0.1606** | 0.1627 |
| Balanced accuracy | 0.1818 | 0.1822 |
| Top-3 accuracy | 0.3946 | 0.3980 |
| Top-5 accuracy | 0.4845 | 0.4914 |
| Latencia media (ms) | 32.36 | — |

Consistente con validación (el modelo no degrada en test).

Pares más confundidos (test): `detroit-techno→minimal-techno` (29),
`hardstyle→happy` (29), `romance→swedish` (28).

## 8. Comparación de candidatos (validación)

| Modelo | Accuracy | Macro F1 | Tamaño |
|---|---:|---:|---:|
| C0 · clase frecuente | 0.0120 | 0.0002 | 0,3 MB |
| **C1 · logística** | **0.2247** | **0.1627** | **0,5 MB** |
| C2 · Extra Trees | 0.2527 | 0.1735 | 690 MB |
| C3 · Random Forest | 0.2866 | 0.2097 | 657 MB |

C2/C3 ganan en calidad (+26 % / +29 % de accuracy respecto a C1) pero pesan
~650 MB cada uno e impiden el despliegue en Streamlit Community Cloud. El
propietario aprobó **C1** (§7 de AGENTS.md) por viabilidad (0,5 MB, ~32 ms).
C2/C3 se eliminaron tras consolidar el reporte
`reports/experiments/classifier_multiclass_comparison.json`.

## 9. Limitaciones

- Laboratorio experimental (§17); no reemplaza etiquetas originales.
- Las puntuaciones no calibradas **no** son probabilidades (§16.10).
- La muestra es balanceada por bloque; no permite inferir prevalencia real de
  Spotify (§2.3).
- Dos clases no tienen ejemplos en train (112/114); sus predicciones dependen
  del comportamiento del solver fuera del soporte observado.
