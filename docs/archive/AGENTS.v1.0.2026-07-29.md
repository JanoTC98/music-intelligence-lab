# NOTA — Instantánea obsoleta

> Este archivo es una copia de seguridad del AGENTS.md versión 1.0 (29 de julio de 2026).
> Ya no se carga como instrucción activa.
> Puede contener configuraciones, estructuras y valores que ya no coinciden con el estado actual del repositorio.
> No se debe usar como referencia normativa.

---

# AGENTS.md — Spotify Music Intelligence

> **Estado:** especificación normativa de implementación  
> **Versión:** 1.0  
> **Fecha:** 29 de julio de 2026  
> **Proyecto:** Spotify Music Intelligence  
> **IDE principal:** PyCharm  
> **Agente de programación:** OpenCode  
> **Alcance de esta versión:** datos, recomendadores, clasificación, aplicación Streamlit, pruebas, base de datos de eventos, documentación y despliegue.  
> **Fuera de alcance:** Power BI y cualquier dashboard de Power BI.
> **Fuentes consolidadas:** `Spotify_Music_Intelligence_Contexto_Completo.md`, `Spotify_Music_Intelligence_Plan_Proyecto.pdf`, `Spotify_Music_Intelligence_Flujos_y_Estrategia_Modelado.pdf` y auditoría directa de `dataset.csv`.

---

## 0. Cómo debe usar OpenCode este archivo

Este archivo es la **fuente normativa de instrucciones del repositorio**. OpenCode debe leerlo antes de planificar, crear, modificar, ejecutar o eliminar archivos.

OpenCode admite instrucciones de proyecto mediante un archivo `AGENTS.md` en la raíz. Por tanto, al iniciar el repositorio, este documento debe copiarse a:

```text
spotify-music-intelligence/AGENTS.md
```

### 0.1 Orden de precedencia

Cuando existan contradicciones, se aplicará este orden:

1. Datos medidos directamente desde `data/raw/dataset.csv`.
2. Decisiones no negociables definidas en este documento.
3. Configuraciones versionadas bajo `configs/`.
4. Código y pruebas existentes en el repositorio.
5. Notebooks y documentos exploratorios.
6. Sugerencias nuevas de OpenCode.

OpenCode **no puede cambiar una decisión no negociable** sin:

1. Identificar la decisión concreta.
2. Explicar el problema técnico.
3. Proponer una alternativa.
4. Mostrar el impacto en datos, pruebas, modelos y aplicación.
5. Esperar aprobación explícita del propietario del proyecto.

### 0.2 Forma de trabajar obligatoria

Para cada tarea, OpenCode debe seguir este ciclo:

```text
Leer AGENTS.md y archivos relacionados
→ resumir el alcance
→ proponer un plan pequeño
→ identificar archivos que modificará
→ esperar aprobación cuando corresponda
→ implementar
→ ejecutar pruebas específicas
→ ejecutar linting
→ mostrar el diff y los resultados
→ no realizar git push
```

### 0.3 Acciones prohibidas para OpenCode

OpenCode no debe:

- Modificar, sobrescribir o formatear `data/raw/dataset.csv`.
- Borrar datos, modelos, métricas o reportes sin autorización.
- Ejecutar `git push`, `git reset --hard`, `git clean -fd` o comandos destructivos.
- Introducir claves, contraseñas o cadenas de conexión reales en Git.
- Entrenar con el conjunto de prueba durante selección de modelos.
- Dividir aleatoriamente por filas ignorando `recording_group_id`.
- Utilizar `track_name`, `artists`, `album_name`, `track_id` o `recording_group_id` como características del clasificador.
- Convertir automáticamente una canción multigénero en una sola etiqueta verdadera.
- Presentar una similitud matemática como probabilidad de gusto.
- Presentar puntuaciones no calibradas como probabilidades.
- Añadir Power BI al alcance.
- Cambiar tecnologías principales sin aprobación.
- Crear un archivo monolítico con toda la lógica en `streamlit_app.py`.
- Importar lógica de producción desde notebooks.
- reentrenar modelos dentro de una interacción de Streamlit.
- Inventar métricas, resultados, conteos o conclusiones.

### 0.4 Responsabilidad final

OpenCode puede generar código, pruebas, configuraciones y comandos, y puede ejecutar entrenamientos autorizados. El propietario del proyecto conserva la responsabilidad de:

- Aprobar filtros y reglas de negocio.
- Aprobar características y pesos.
- Aprobar hiperparámetros y experimentos.
- Revisar la fuga de información.
- Interpretar métricas.
- Elegir el modelo final.
- Autorizar cambios de alcance.
- Ejecutar manualmente operaciones administrativas de base de datos.

---

# 1. Definición del proyecto

## 1.1 Nombre

**Spotify Music Intelligence**

## 1.2 Objetivo

Construir un proyecto end-to-end de ciencia de datos y machine learning que transforme un catálogo musical en:

1. Un recomendador basado en una canción.
2. Un recomendador basado en preferencias y presets editables.
3. Un laboratorio supervisado de clasificación multietiqueta de géneros.
4. Un experimento multiclase de género acústico dominante.
5. Una aplicación web multipágina en Streamlit.
6. Un pipeline reproducible de datos y modelos.
7. Un sistema opcional de eventos y feedback almacenado en MySQL.
8. Un repositorio profesional con pruebas, documentación, CI y despliegue.

## 1.3 Producto central

El producto principal es el **recomendador musical basado en contenido**. Los clasificadores son módulos experimentales complementarios y no deben bloquear la publicación del MVP.

## 1.4 Fuera de alcance

No se desarrollará en esta especificación:

- Power BI.
- Filtrado colaborativo.
- Predicción de la siguiente canción de un usuario.
- Aprendizaje secuencial con GRU o LSTM.
- Integración con cuentas personales de Spotify.
- Registro de nombres, correos, IP o ubicación.
- Aplicación móvil.
- Frontend React, Angular o Vue.
- Backend separado con FastAPI o Spring Boot durante el MVP.
- Pagos, autenticación de usuarios o perfiles personales.

---

# 2. Hechos verificados del dataset

Archivo base:

```text
data/raw/dataset.csv
```

## 2.1 Dimensiones verificadas

| Métrica | Valor |
|---|---:|
| Filas | 114.000 |
| Columnas | 21 |
| Géneros | 114 |
| Filas por género | 1.000 |
| `track_id` únicos | 89.741 |
| Filas adicionales respecto a IDs únicos | 24.259 |
| Canciones con más de un género | 16.299 `track_id` |
| Máximo de géneros por canción | 9 |
| Popularidad igual a cero | 16.020 filas |
| Duración inferior a 60 segundos | 851 filas |
| Duración superior a 10 minutos | 603 filas |
| Tempo igual a cero | 157 filas |
| Nulos en `artists` | 1 |
| Nulos en `album_name` | 1 |
| Nulos en `track_name` | 1 |

## 2.2 Columnas originales

```text
Unnamed: 0
track_id
artists
album_name
track_name
popularity
duration_ms
explicit
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
time_signature
track_genre
```

## 2.3 Estructura por bloques y prevalencia

El CSV contiene 114 bloques consecutivos de 1.000 filas. Cada bloque corresponde al valor de `track_genre`.

Cada género ocupa artificialmente:

```text
1.000 / 114.000 = 0,0087719 ≈ 0,877 % del archivo
```

Esto no permite estimar la prevalencia real de géneros en Spotify. Solo permite analizar el conjunto balanceado entregado.

Sí puede estudiarse:

- Perfil acústico de cada género dentro de la muestra.
- Distribuciones y correlaciones.
- Solapamiento entre etiquetas.
- Calidad, duplicados y anomalías.
- Separabilidad de las etiquetas usando las características disponibles.

No puede concluirse:

- Qué proporción real del catálogo de Spotify pertenece a cada género.
- Qué género es el más abundante o escuchado en Spotify.
- Participación de mercado.
- Tendencias globales o temporales.

## 2.4 Reglas de interpretación

- `popularity = 0` se conserva y no significa falta de calidad.
- Duraciones extremas se conservan y se marcan; no son errores automáticos.
- `tempo = 0` forma parte de un patrón de análisis acústico incompleto.
- `Unnamed: 0` es un índice artificial y se elimina de los datos procesados.
- El dataset original nunca se modifica.

---

# 3. Identidad musical y prevención de fuga

## 3.1 Niveles de identidad

```text
track_id
    Identificador original del catálogo.

recording_group_id
    Identificador interno para agrupar entradas acústicamente equivalentes.

work_group_id
    Agrupación opcional de versiones de una misma composición.
```

### Uso de cada nivel

| Nivel | Uso |
|---|---|
| `track_id` | Trazabilidad, metadatos y presentación |
| `recording_group_id` | Unidad principal de recomendación, división y entrenamiento |
| `work_group_id` | Evaluación estricta opcional; no es requisito del MVP |

No se reemplaza ni se reescribe ningún `track_id`.

## 3.2 Consolidación por `track_id`

Una misma ID repetida mantiene iguales nombre, artistas, álbum, duración y características de audio. Puede cambiar:

- `track_genre`.
- En algunos casos, `popularity`.

La tabla `tracks.parquet` tendrá una fila por `track_id` y conservará:

```text
popularity_min
popularity_max
popularity_median
popularity_observations
```

## 3.3 Regla exacta inicial de `recording_group_id`

La agrupación automática será deliberadamente conservadora.

### Normalización de texto exacta

Para `track_name` y `artists`:

1. Convertir a Unicode NFKC.
2. Aplicar `casefold()`.
3. Eliminar espacios al inicio y al final.
4. Reemplazar grupos de espacios internos por un solo espacio.
5. No eliminar palabras como `live`, `remix`, `acoustic`, `remastered` o equivalentes.
6. No reordenar artistas en la agrupación exacta.
7. No eliminar puntuación en la agrupación exacta.

### Huella exacta

```text
track_name_normalized
artists_normalized
duration_ms
explicit
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
time_signature
```

Las características se tomarán tal como aparecen en el CSV. No se redondearán para la agrupación exacta.

### Generación del ID

```text
canonical_fingerprint
→ serialización estable UTF-8
→ SHA-256
→ recording_group_id de 64 caracteres hexadecimales
```

Con esta regla conservadora, el conteo de referencia esperado es aproximadamente:

```text
83.881 recording_group_id exactos
```

Este conteo debe comprobarse mediante una prueba de regresión. Si cambia, OpenCode debe detenerse y explicar qué regla o dato produjo el cambio.

## 3.4 Candidatos a casi duplicados

Se generará una tabla separada de candidatos usando:

- Título normalizado de forma más flexible.
- Artistas normalizados y opcionalmente ordenados.
- Diferencia de duración.
- Similitud de la huella acústica.
- RapidFuzz para similitud textual.
- Nearest Neighbors para candidatos acústicos.

Los candidatos aproximados **no se fusionan automáticamente**. Se exportan a:

```text
reports/identity/near_duplicate_candidates.csv
```

La fusión solo puede realizarse después de una revisión manual y una regla versionada.

## 3.5 Regla contra fuga

Un `recording_group_id` completo debe pertenecer a un único conjunto:

```text
train
validation
test
```

No puede aparecer en dos conjuntos.

---

# 4. Tratamiento de anomalías

## 4.1 Análisis acústico incompleto

Se define:

```text
audio_analysis_incomplete = True
```

cuando se detecta el patrón confirmado de análisis incompleto, incluyendo los casos con:

```text
tempo = 0
danceability = 0
speechiness = 0
valence = 0
time_signature = 0
```

### Política por módulo

| Módulo | Tratamiento |
|---|---|
| Datos brutos | Conservar sin alterar |
| Auditoría | Contar y documentar |
| Catálogo | Conservar con indicador |
| Recomendador por canción | Excluir como semilla y candidata |
| Recomendador por preferencias | Excluir como candidata |
| Clasificador baseline | Excluir |
| Clasificador experimental | Comparar exclusión contra imputación + indicador |

La imputación, cuando se pruebe, se ajustará solo con entrenamiento.

## 4.2 Duraciones extremas

Se crearán indicadores:

```text
is_short_track = duration_ms < 60_000
is_long_track = duration_ms > 600_000
```

Política:

- Mantener en datos procesados.
- No excluir del análisis general.
- Permitir filtros opcionales en la aplicación.
- No activar el filtro de duración por defecto.

## 4.3 Popularidad

La popularidad:

- No participa en la distancia acústica inicial.
- No es variable objetivo.
- Se muestra como metadato contextual.
- No se usa como desempate por defecto.
- Puede evaluarse en una variante secundaria explícitamente documentada.

## 4.4 Valores nulos de identidad

La fila sin `artists`, `album_name` y `track_name` se enviará a:

```text
data/quarantine/invalid_identity.parquet
```

No se eliminará silenciosamente.

---

# 5. Estrategia de géneros

## 5.1 Etiquetas originales

Se conservan los 114 géneros originales.

## 5.2 Clasificador principal: multietiqueta

Unidad:

```text
una fila por recording_group_id
```

Objetivo:

```text
vector binario de 114 etiquetas
```

Ejemplo:

```text
pop = 1
indie-pop = 1
synth-pop = 1
rock = 0
...
```

## 5.3 Clasificador secundario: multiclase

Se utiliza únicamente el subconjunto de grabaciones con una sola etiqueta.

Salida:

```text
un género acústico dominante estimado
```

Esta predicción no reemplaza las etiquetas originales de canciones multigénero.

## 5.4 Prohibición de pseudo-limpieza

No se entrenará un modelo para asignar una etiqueta única y usarla como verdad de entrenamiento del mismo proyecto.

---

# 6. Tecnologías oficiales

## 6.1 Herramientas principales

| Área | Tecnología | Decisión |
|---|---|---|
| IDE | PyCharm | IDE principal |
| Agente de código | OpenCode | Generación y automatización supervisada |
| Lenguaje | Python 3.12.x | Versión fijada del proyecto |
| Entorno y dependencias | uv | Crea `.venv`, resuelve y bloquea dependencias |
| Datos | pandas, NumPy, PyArrow | Procesamiento y Parquet |
| Machine learning | scikit-learn | Modelos principales |
| ML avanzado | XGBoost | Experimento opcional |
| Similitud textual | RapidFuzz | Candidatos a casi duplicados |
| Aplicación | Streamlit | Interfaz web multipágina |
| Gráficos | Matplotlib y Plotly | Visualización |
| Configuración | YAML y TOML | Configuración versionada |
| Persistencia | MySQL 8.0+ | Eventos y feedback opcionales |
| ORM/conexión | SQLAlchemy + PyMySQL | Acceso a MySQL |
| Pruebas | pytest + pytest-cov | Unitarias e integración |
| Calidad | Ruff | Linting y formato |
| Tipos | mypy | Comprobación estática gradual |
| Control de versiones | Git | Historial local |
| Repositorio | GitHub | Issues, PR, releases y CI |
| CI | GitHub Actions | Pruebas y linting |
| Despliegue | Streamlit Community Cloud | Aplicación pública |
| Serialización | joblib, JSON, YAML | Modelos y configuración |

## 6.2 Gestión de dependencias con uv

Archivos obligatorios:

```text
pyproject.toml
uv.lock
.python-version
requirements.txt
```

Reglas:

- `pyproject.toml` declara dependencias directas.
- `uv.lock` contiene las versiones exactas y se versiona en Git.
- `requirements.txt` se genera desde `uv.lock` para despliegue.
- No editar `requirements.txt` manualmente.

Comandos base:

```powershell
uv python install 3.12
uv python pin 3.12
uv sync
uv run python --version
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv export --format requirements.txt --no-dev --no-hashes -o requirements.txt
```

## 6.3 Dependencias directas requeridas

### Producción

```text
pandas
numpy
pyarrow
scipy
scikit-learn
joblib
pyyaml
rapidfuzz
streamlit
plotly
matplotlib
sqlalchemy
pymysql
cryptography
python-dotenv
```

### Opcional de modelado avanzado

```text
xgboost
```

### Desarrollo

```text
pytest
pytest-cov
ruff
mypy
jupyter
ipykernel
pre-commit
```

## 6.4 Rango de Python

```toml
requires-python = ">=3.12,<3.13"
```

No migrar a otra versión mayor durante el proyecto sin ejecutar toda la suite y regenerar el lockfile.

---

# 7. Reparto de responsabilidades: usuario y OpenCode

| Actividad | OpenCode | Propietario |
|---|---:|---:|
| Crear estructura del repositorio | Sí | Revisar |
| Generar código | Sí | Revisar y comprender |
| Crear pruebas | Sí | Aprobar casos |
| Ejecutar pruebas | Sí | Revisar resultados |
| Ejecutar auditorías | Sí | Validar conclusiones |
| Proponer filtros | Puede sugerir | Aprobar |
| Definir filtros definitivos | No | Sí |
| Proponer características | Puede sugerir | Aprobar |
| Definir características finales | No | Sí |
| Proponer hiperparámetros | Puede sugerir | Aprobar |
| Ejecutar entrenamiento | Sí, con autorización | Supervisar |
| Elegir modelo final | No | Sí |
| Crear base de datos | No como administrador | Sí, manualmente |
| Crear tablas con script autorizado | Sí | Ejecutar/revisar |
| Guardar contraseñas | No | Sí, fuera de Git |
| Hacer commits | Solo si se autoriza | Preferentemente el propietario |
| Hacer `git push` | No | Sí |
| Interpretar resultados finales | Apoyo | Sí |

---

# 8. Estructura oficial del repositorio

```text
spotify-music-intelligence/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .python-version
├── .gitignore
├── .env.example
├── opencode.json
├── streamlit_app.py
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
│
├── .opencode/
│   ├── agents/
│   │   ├── data-auditor.md
│   │   ├── ml-engineer.md
│   │   └── model-reviewer.md
│   └── commands/
│       ├── validate-data.md
│       ├── test.md
│       ├── train-baseline.md
│       ├── compare-models.md
│       └── review-leakage.md
│
├── app/
│   ├── __init__.py
│   ├── pages/
│   │   ├── home.py
│   │   ├── data_audit.py
│   │   ├── recommend_by_track.py
│   │   ├── recommend_by_preferences.py
│   │   ├── multilabel_genre_lab.py
│   │   ├── dominant_genre_lab.py
│   │   └── methodology.py
│   └── components/
│       ├── cards.py
│       ├── charts.py
│       ├── filters.py
│       ├── messages.py
│       └── tables.py
│
├── configs/
│   ├── data_rules.yaml
│   ├── recommender_features.yaml
│   ├── classifier_features.yaml
│   ├── presets.yaml
│   ├── model_parameters.yaml
│   └── app.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   ├── quarantine/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_identity_analysis.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_track_recommender_experiments.ipynb
│   ├── 05_preference_recommender_experiments.ipynb
│   ├── 06_multilabel_classifier.ipynb
│   └── 07_multiclass_classifier.ipynb
│
├── src/
│   └── spotify_intelligence/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load.py
│       │   ├── clean.py
│       │   ├── validate.py
│       │   ├── audit.py
│       │   ├── contracts.py
│       │   └── splits.py
│       │
│       ├── identity/
│       │   ├── __init__.py
│       │   ├── normalize.py
│       │   ├── fingerprints.py
│       │   ├── recording_groups.py
│       │   └── duplicate_candidates.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── audio_features.py
│       │   ├── encoders.py
│       │   └── presets.py
│       │
│       ├── recommenders/
│       │   ├── __init__.py
│       │   ├── track_based.py
│       │   ├── preference_based.py
│       │   ├── scoring.py
│       │   ├── diversity.py
│       │   ├── explanations.py
│       │   └── evaluation.py
│       │
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── multilabel.py
│       │   ├── multiclass.py
│       │   ├── thresholds.py
│       │   ├── predict.py
│       │   └── evaluation.py
│       │
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── database.py
│       │   ├── models.py
│       │   ├── repositories.py
│       │   └── events.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── hashing.py
│           ├── logging.py
│           └── timing.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── validate_processed_data.py
│   ├── build_recommender.py
│   ├── evaluate_recommender.py
│   ├── create_splits.py
│   ├── train_multilabel_classifier.py
│   ├── train_multiclass_classifier.py
│   ├── compare_models.py
│   ├── evaluate_final_model.py
│   ├── export_requirements.py
│   └── initialize_database.py
│
├── models/
│   ├── README.md
│   ├── recommender/
│   │   └── .gitkeep
│   └── classifier/
│       └── .gitkeep
│
├── database/
│   ├── mysql/
│   │   ├── 00_create_database_and_user.sql
│   │   ├── 01_create_schema.sql
│   │   └── 02_verify_schema.sql
│   └── sqlserver/
│       ├── 00_create_database_and_login.sql
│       ├── 01_create_schema.sql
│       └── 02_verify_schema.sql
│
├── reports/
│   ├── data_quality/
│   ├── identity/
│   ├── figures/
│   ├── metrics/
│   ├── experiments/
│   └── model_cards/
│
├── tests/
│   ├── fixtures/
│   ├── unit/
│   │   ├── test_load.py
│   │   ├── test_clean.py
│   │   ├── test_validate.py
│   │   ├── test_identity.py
│   │   ├── test_features.py
│   │   ├── test_track_recommender.py
│   │   ├── test_preference_recommender.py
│   │   ├── test_multilabel.py
│   │   ├── test_multiclass.py
│   │   └── test_persistence.py
│   └── integration/
│       ├── test_prepare_data_pipeline.py
│       ├── test_recommendation_pipeline.py
│       ├── test_training_pipeline.py
│       ├── test_database_roundtrip.py
│       └── test_streamlit_imports.py
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── data_quality.md
│   ├── identity_resolution.md
│   ├── methodology.md
│   ├── privacy.md
│   ├── deployment.md
│   └── roadmap.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

## 8.1 Reglas de estructura

- Código importable únicamente bajo `src/spotify_intelligence/`.
- Interfaz bajo `app/`.
- Notebooks solo para exploración y experimentos.
- Scripts pequeños que llaman funciones de `src/`.
- Configuración fuera del código bajo `configs/`.
- Modelos y métricas separados.
- Datos brutos no se suben si la licencia o tamaño no lo permiten.
- No versionar `.env`, `.streamlit/secrets.toml`, `.venv` ni credenciales.

---

# 9. Artefactos de datos y contratos

## 9.1 Salidas procesadas

```text
data/processed/tracks.parquet
data/processed/recordings.parquet
data/processed/recording_tracks.parquet
data/processed/track_genres.parquet
data/processed/recording_genres.parquet
data/processed/track_artists.parquet
data/processed/genre_catalog.parquet
data/processed/anomalies.parquet
data/processed/splits.parquet
reports/data_quality/data_quality_report.json
reports/identity/near_duplicate_candidates.csv
configs/generated/feature_manifest.json
```

## 9.2 Contrato de `tracks.parquet`

Una fila por `track_id` válido.

Columnas mínimas:

```text
track_id
track_name
track_name_normalized
artists
artists_normalized
album_name
popularity_min
popularity_max
popularity_median
popularity_observations
duration_ms
duration_min
explicit
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
time_signature
audio_analysis_incomplete
is_short_track
is_long_track
recording_group_id
```

## 9.3 Contrato de `recordings.parquet`

Una fila por `recording_group_id`.

Columnas mínimas:

```text
recording_group_id
representative_track_id
track_name
artists
album_name
track_id_count
genre_count
artist_count
popularity_median
duration_ms
explicit
características acústicas
audio_analysis_incomplete
```

## 9.4 Selección del track representativo

Dentro de un `recording_group_id`:

1. Mayor `popularity_median`.
2. En empate, menor `track_id` lexicográfico.

Esta regla selecciona la entrada mostrada; no afirma que sea la versión oficial o más reciente.

## 9.5 Manifiesto del pipeline

Cada preparación debe guardar:

```json
{
  "dataset_sha256": "...",
  "pipeline_version": "...",
  "git_commit": "...",
  "generated_at_utc": "...",
  "raw_rows": 114000,
  "valid_track_ids": 89740,
  "recording_group_count": 83881,
  "rules_config_sha256": "..."
}
```

Los conteos finales deben calcularse; los valores mostrados son referencias esperadas según las reglas actuales.

---

# 10. Módulo 0 — Inicialización y entorno

> **Estado:** ✅ Completado

## Objetivo

Crear un repositorio reproducible, un entorno Python y reglas para OpenCode.

## Flujo

```text
Crear repositorio
→ copiar AGENTS.md
→ configurar uv y Python 3.12
→ generar estructura mínima
→ instalar dependencias
→ configurar PyCharm
→ configurar Ruff y pytest
→ ejecutar prueba de humo
```

## Salidas

```text
pyproject.toml
uv.lock
.python-version
.gitignore
.env.example
opencode.json
src/spotify_intelligence/__init__.py
tests/unit/test_smoke.py
```

## Criterios de aceptación

```powershell
uv sync
uv run python -c "import spotify_intelligence"
uv run pytest
uv run ruff check .
```

Todos deben terminar con código 0.

---

# 11. Módulo 1 — Ingesta, auditoría y validación

> **Estado:** ✅ Completado

## Objetivo

Crear una auditoría reproducible del CSV sin modificarlo.

## Flujo

```text
dataset.csv
→ calcular hash
→ validar columnas
→ validar tipos
→ medir nulos y duplicados
→ detectar anomalías
→ comprobar bloques de género
→ generar reporte JSON y notebook
```

## Reglas

- Usar `pd.read_csv` con ruta configurable.
- No fijar rutas absolutas.
- Fallar con mensaje claro si falta una columna requerida.
- Validar rangos esperados de audio features.
- No corregir datos durante la auditoría.

## Rangos de validación

| Variable | Regla inicial |
|---|---|
| `popularity` | 0 a 100 |
| `duration_ms` | mayor que 0 |
| `danceability` | 0 a 1 |
| `energy` | 0 a 1 |
| `key` | -1 a 11 |
| `loudness` | valor numérico; reportar extremos |
| `mode` | 0 o 1 |
| `speechiness` | 0 a 1 |
| `acousticness` | 0 a 1 |
| `instrumentalness` | 0 a 1 |
| `liveness` | 0 a 1 |
| `valence` | 0 a 1 |
| `tempo` | mayor o igual a 0 |
| `time_signature` | 0 a 5 en esta muestra; reportar otros |

## ¿Usa ML?

No.

## ¿Se entrena?

No.

---

# 12. Módulo 2 — Limpieza, catálogo e identidad

> **Estado:** ✅ Completado

## Objetivo

Crear una única fuente procesada para recomendadores, clasificadores y aplicación.

## Flujo

```text
CSV original
→ eliminar índice artificial
→ cuarentena de identidad inválida
→ consolidar una fila por track_id
→ agregar popularidad
→ crear tablas de géneros y artistas
→ generar recording_group_id exacto
→ seleccionar track representativo
→ marcar anomalías
→ escribir Parquet y manifiesto
```

## ¿Usa ML?

La agrupación exacta no.

La generación de candidatos aproximados usa búsqueda no supervisada, pero no toma decisiones automáticas de fusión.

## ¿Se entrena?

No hay entrenamiento supervisado. Puede ajustarse un índice de vecinos para candidatos.

## Criterios de aceptación

- `tracks.track_id` es único.
- `recordings.recording_group_id` es único.
- Toda ID válida tiene un grupo.
- No hay duplicados exactos `track_id`–género.
- El dataset original conserva su hash.
- Los conteos se registran en el manifiesto.

---

# 13. Módulo 3 — Análisis exploratorio

> **Estado:** ✅ Completado

## Preguntas obligatorias

- ¿Cómo se distribuyen las características?
- ¿Qué correlaciones existen?
- ¿Cómo cambian las estadísticas al consolidar grabaciones?
- ¿Qué géneros comparten canciones?
- ¿Qué etiquetas presentan solapamiento total o elevado?
- ¿Qué variables son redundantes para distancia?
- ¿Qué géneros tienen perfiles acústicos similares?
- ¿Dónde aparecen anomalías?
- ¿Cómo se distribuyen las duraciones extremas?
- ¿Qué diferencias descriptivas existen entre explícitas y no explícitas?

## Formato de cada análisis

```text
Pregunta
→ método
→ resultado
→ interpretación
→ limitación
```

## Reglas

- No inferir causalidad.
- No inferir prevalencia real de Spotify.
- No usar gráficos sin texto interpretativo.
- Guardar figuras en `reports/figures/`.
- Trasladar cálculos reutilizables a `src/`; no dejarlos solo en notebooks.

---

# 14. Módulo 4 — Recomendador por canción

> **Estado:** ✅ Completado

## 14.1 Objetivo

Encontrar grabaciones acústicamente cercanas a una canción seleccionada.

## 14.2 Unidad

```text
recording_group_id
```

## 14.3 Características iniciales

```text
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
```

No se incluyen inicialmente:

```text
popularity
duration_ms
explicit
key
mode
time_signature
track_genre
metadatos de identidad
```

`duration_ms`, `explicit` y géneros funcionan como filtros opcionales.

## 14.4 Experimentos obligatorios

| ID | Escalador | Distancia |
|---|---|---|
| R1 | StandardScaler | coseno |
| R2 | RobustScaler | coseno |
| R3 | StandardScaler | euclídea |
| R4 | RobustScaler | euclídea |

Baseline inicial:

```text
StandardScaler + NearestNeighbors(metric="cosine", algorithm="brute")
```

## 14.5 Preparación offline

```text
recordings elegibles
→ seleccionar features
→ excluir audio incompleto
→ ajustar escalador
→ transformar matriz
→ ajustar índice de vecinos
→ guardar matriz, índice, escalador y configuración
```

Artefactos:

```text
models/recommender/<version>/scaler.joblib
models/recommender/<version>/neighbors.joblib
models/recommender/<version>/catalog_matrix.npy
models/recommender/<version>/catalog_index.parquet
models/recommender/<version>/manifest.json
```

## 14.6 Flujo online

```text
Usuario busca canción + artista
→ resolver track_id
→ obtener recording_group_id
→ validar audio completo
→ obtener vector
→ transformar con escalador guardado
→ buscar candidatos
→ excluir la propia grabación
→ excluir otras grabaciones de la misma obra (mismo título + artistas, §3.4)
→ aplicar filtros
→ eliminar grupos repetidos
→ ordenar
→ explicar diferencias
→ devolver Top-N
```

## 14.7 Recuperación progresiva

```text
Top-N solicitado: 5 a 20
Valor por defecto: 10

Primera búsqueda: max(100, Top-N × 10)
Si faltan resultados: 500
Después: 2.000
Último recurso: catálogo completo elegible
```

## 14.8 Filtros

| Filtro | Estado por defecto |
|---|---|
| Contenido explícito | Todos |
| Género | Todos |
| Duración | Desactivado |
| Intervalo sugerido al activarlo | 60 a 600 segundos |
| Artista distinto a la semilla | Desactivado |
| Popularidad mínima | Desactivado |

Los filtros son aprobados por el propietario. OpenCode no puede activarlos o cambiar valores por cuenta propia.

## 14.9 Salida

```text
track_name
artists
album_name
recording_group_id
representative_track_id
géneros asociados
similitud o distancia
diferencias por característica
duración
popularidad contextual
explicit
```

## 14.10 Explicabilidad

La similitud coseno se calcula como:

```text
similarity = 1 - cosine_distance
```

No llamarla probabilidad.

Ejemplo de explicación:

```text
Similitud coseno: 0,913
Energía: diferencia de 0,12 desviaciones estándar
Tempo: diferencia de 4,8 BPM
Bailabilidad: diferencia absoluta de 0,04
```

## 14.11 Evaluación

Métricas mínimas:

```text
autorrecomendación = 0
duplicados de recording_group_id = 0
cumplimiento de filtros = 100 %
similitud media
cobertura del catálogo
diversidad interna
artistas únicos por lista
latencia p50 y p95
estabilidad ante perturbaciones pequeñas
```

No existe accuracy de recomendación porque no hay ground truth de preferencias.

## 14.12 Tipo de aprendizaje

No supervisado.

## 14.13 ¿Se entrena?

No se entrena con etiquetas. Se ajustan escalador e índice.

---

# 15. Módulo 5 — Recomendador por preferencias

> **Estado:** ✅ Completado

## 15.1 Objetivo

Encontrar canciones cercanas a un vector configurado por el usuario.

## 15.2 Modos

```text
Preset editable
Perfil manual
```

## 15.3 Modo básico

```text
energy
danceability
valence
acousticness
instrumentalness
tempo
```

## 15.4 Modo avanzado

```text
speechiness
liveness
loudness
duration
key
mode
time_signature
```

El modo avanzado se incorpora después de validar el modo básico.

## 15.5 Regla de pesos

- Escala de peso: 0 a 3.
- 0 significa ignorar la variable.
- 1 baja importancia.
- 2 importancia media.
- 3 alta importancia.
- Toda variable no configurada tiene peso 0.
- Rechazar una consulta si todos los pesos son 0.

## 15.6 Distancia ponderada

Sobre variables escaladas:

```text
d(x, q) = sqrt( sum(w_i * (x_i - q_i)^2) / sum(w_i) )
```

Donde:

- `x` es una grabación.
- `q` es el perfil.
- `w_i` es el peso aprobado.

## 15.7 Presets iniciales

Estos valores son decisiones de diseño editables, no definiciones universales.

| Preset | Energy | Danceability | Valence | Acousticness | Instrumentalness | Tempo |
|---|---:|---:|---:|---:|---:|---:|
| Entrenamiento intenso | 0,90 | 0,75 | 0,65 | 0,05 | 0,05 | 145 |
| Fiesta | 0,85 | 0,90 | 0,75 | 0,05 | 0,02 | 128 |
| Concentración instrumental | 0,35 | 0,30 | 0,45 | 0,55 | 0,85 | 90 |
| Relajación | 0,25 | 0,35 | 0,55 | 0,75 | 0,45 | 80 |
| Alegre y bailable | 0,72 | 0,85 | 0,88 | 0,08 | 0,02 | 124 |
| Melancólico | 0,30 | 0,35 | 0,15 | 0,60 | 0,20 | 85 |
| Acústico | 0,40 | 0,45 | 0,55 | 0,90 | 0,05 | 95 |

Pesos iniciales sugeridos:

```text
energy = 2
danceability = 2
valence = 1
acousticness = 2
instrumentalness = 1
tempo = 2
```

Cada preset puede sobrescribirlos. Cualquier cambio debe realizarse en `configs/presets.yaml`, no dentro de Python.

## 15.8 Perfil fuera de distribución

Procedimiento:

```text
Calcular distancias de referencia del catálogo
→ obtener percentiles
→ comparar la consulta
```

Umbrales iniciales:

```text
percentil 95: advertencia de perfil poco frecuente
percentil 99: coincidencia muy débil
```

Los percentiles se calculan sobre datos y no se hardcodean como distancias absolutas.

## 15.9 Diversidad opcional

Se implementará MMR después del ranking puro:

```text
score = lambda * relevance - (1 - lambda) * similarity_to_selected
```

Valor inicial:

```text
lambda = 0,85
```

Debe existir control para activar o desactivar diversidad.

## 15.10 Tipo de aprendizaje

No supervisado/determinista.

## 15.11 ¿Se entrena?

Solo se ajusta el escalador del catálogo. No aprende gustos.

---

# 16. Módulo 6 — Clasificador multietiqueta

> **Estado:** ✅ Completado

## 16.1 Objetivo

Predecir un conjunto compatible de los 114 géneros a partir de características acústicas.

## 16.2 Dataset

Una fila por `recording_group_id`.

Objetivo `Y`:

```text
matriz binaria [n_grabaciones, 114]
```

## 16.3 Características primarias

```text
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
log_duration
key_sin
key_cos
mode
time_signature one-hot
```

Transformaciones:

```text
log_duration = log1p(duration_ms)
key_sin = sin(2*pi*key/12)
key_cos = cos(2*pi*key/12)
```

No usar como características:

```text
track_id
recording_group_id
track_name
artists
album_name
track_genre
```

Variante secundaria, separada del resultado principal:

```text
audio_plus_context = características primarias + popularity_median + explicit
```

## 16.4 Splits

Proporción objetivo:

```text
train = 70 %
validation = 15 %
test = 15 %
random_state = 42
```

Procedimiento:

1. Generar múltiples divisiones agrupadas por `recording_group_id`.
2. Verificar intersecciones vacías.
3. Comparar prevalencia de etiquetas entre splits.
4. Seleccionar la división con menor desviación.
5. Guardarla en `data/processed/splits.parquet`.
6. Congelar el test.

El test no se usa para elegir variables, umbrales o hiperparámetros.

## 16.5 Tratamiento de audio incompleto

Experimentos:

```text
A: excluir audio_analysis_incomplete
B: imputar con estadísticas de train + indicador binario
```

Baseline principal: A.

## 16.6 Modelos

### M0 — Baseline de frecuencia

Predice las etiquetas más frecuentes sin usar características.

### M1 — One-vs-Rest Logistic Regression

Configuración inicial:

```text
base_estimator = LogisticRegression
solver = liblinear
C = 1.0
max_iter = 2000
class_weight = balanced
random_state = 42
wrapper = OneVsRestClassifier
wrapper_n_jobs = -1
```

### M2 — Classifier Chain

Base:

```text
LogisticRegression con la configuración anterior
```

Entrenar tres cadenas:

```text
random_state = 42, 43, 44
order = random
```

Promediar puntuaciones para comparación experimental.

### M3 — Extra Trees multisalida

Configuración inicial:

```text
n_estimators = 400
max_depth = None
min_samples_split = 2
min_samples_leaf = 2
max_features = sqrt
bootstrap = False
class_weight = None
n_jobs = -1
random_state = 42
```

### M4 — Random Forest multisalida

Configuración inicial:

```text
n_estimators = 400
max_depth = None
min_samples_split = 2
min_samples_leaf = 2
max_features = sqrt
bootstrap = True
class_weight = None
n_jobs = -1
random_state = 42
```

### M5 — XGBoost One-vs-Rest opcional

No se ejecuta hasta completar M1–M4.

Configuración inicial orientativa:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.80
colsample_bytree = 0.80
min_child_weight = 1
reg_lambda = 1
objective = binary:logistic
eval_metric = logloss
tree_method = hist
random_state = 42
```

Para evitar sobresuscripción de CPU, no permitir simultáneamente paralelismo máximo en el wrapper y dentro de cada XGBoost.

## 16.7 Selección de hiperparámetros

No realizar búsquedas masivas inicialmente.

Orden:

1. Entrenar configuración inicial.
2. Revisar tiempos y errores.
3. Probar una búsqueda pequeña sobre validación.
4. Cambiar un grupo de parámetros por vez.
5. Registrar cada experimento.

## 16.8 Umbral

La aplicación siempre muestra Top-5.

Para convertir puntuaciones en etiquetas:

```text
buscar threshold global desde 0,10 hasta 0,90 en pasos de 0,05
optimizar samples F1 en validación
```

No ajustar umbrales con test.

Si ninguna etiqueta supera el umbral:

```text
mostrar top-1 y advertir que no superó el umbral
```

## 16.9 Métricas

```text
macro F1
micro F1
samples F1
LRAP
precision@3
recall@5
hit@3
hit@5
Hamming loss
coverage error
average precision por etiqueta
latencia
tamaño del modelo
tiempo de entrenamiento
```

## 16.10 Puntuaciones y probabilidades

- Llamar `puntuación` a una salida no calibrada.
- Solo llamar `probabilidad` si existe calibración evaluada.
- Guardar curvas o métricas de calibración si se implementa.

## 16.11 Tipo de aprendizaje

Supervisado multietiqueta.

## 16.12 ¿Se entrena?

Sí.

---

# 17. Módulo 7 — Clasificador multiclase secundario

> **Estado:** ✅ Completado

## 17.1 Dataset

Solo grabaciones con exactamente un género.

## 17.2 Objetivo

```text
una clase entre 114
```

## 17.3 Modelos

### C0 — Clase frecuente

Baseline trivial.

### C1 — Regresión logística

```text
solver = lbfgs
C = 1.0
max_iter = 3000
class_weight = balanced
random_state = 42
```

### C2 — Extra Trees

```text
n_estimators = 500
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced
n_jobs = -1
random_state = 42
```

### C3 — Random Forest

```text
n_estimators = 500
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced
n_jobs = -1
random_state = 42
```

### C4 — XGBoost opcional

Solo después de comparar C1–C3.

## 17.4 Métricas

```text
macro F1
balanced accuracy
accuracy
top-3 accuracy
top-5 accuracy
matriz de confusión normalizada
pares de géneros más confundidos
latencia
tamaño del modelo
tiempo de entrenamiento
```

## 17.5 Aplicación a grabaciones multigénero

Salida permitida:

```text
género acústico dominante estimado
```

Evaluación exploratoria:

```text
Hit@1: top-1 pertenece a alguna etiqueta original
Hit@3: alguna de las tres primeras pertenece al conjunto original
Recall@5: proporción de etiquetas originales en top-5
```

No reemplazar etiquetas originales.

## 17.6 Tipo de aprendizaje

Supervisado multiclase.

## 17.7 ¿Se entrena?

Sí.

---

# 18. Módulo 8 — Aplicación Streamlit

> **Estado:** ✅ Completado

## 18.1 Navegación

```text
Inicio
Auditoría y catálogo
Recomendar por canción
Recomendar por preferencias
Laboratorio multietiqueta
Laboratorio de género dominante
Metodología y limitaciones
```

`streamlit_app.py` actúa como router. La lógica vive en `src/`.

## 18.2 Carga de recursos

```text
datos procesados → st.cache_data
modelos, escaladores e índices → st.cache_resource
```

## 18.3 Flujo de ejecución

```text
Inicio de app
→ cargar configuración
→ cargar datos procesados
→ cargar artefactos existentes
→ comprobar versiones compatibles
→ habilitar páginas disponibles

Interacción
→ validar entrada
→ transformar consulta
→ buscar o predecir
→ renderizar salida
→ registrar evento si tracking está habilitado
```

## 18.4 Comportamiento ante artefactos ausentes

La aplicación no debe entrenar automáticamente.

Debe mostrar:

```text
“El artefacto requerido no existe. Ejecute el script de construcción correspondiente.”
```

## 18.5 Requisitos de UX

- Búsqueda desambiguada por canción y artista.
- Mensajes claros de audio incompleto.
- Estados vacíos.
- Control de errores sin stack traces visibles.
- Descarga de recomendaciones CSV.
- Definiciones de `valence`, `speechiness` y otras variables.
- Advertencias de limitaciones.
- No usar colores o branding que impliquen afiliación oficial con Spotify.

## 18.6 Despliegue

Destino inicial:

```text
Streamlit Community Cloud
```

La app pública puede operar con:

```text
TRACKING_ENABLED=false
```

hasta disponer de una base de datos remota accesible. Una base MySQL instalada en el PC local no es accesible desde Community Cloud.

---

# 19. Módulo 9 — Persistencia de eventos y feedback

## 19.1 ¿Es obligatoria la base de datos?

No para V1 ni V2.

Es obligatoria únicamente cuando se active:

- Registro pseudónimo de uso.
- Feedback de resultados.
- Análisis de latencia y cobertura con uso real.
- Historial de versiones consultadas.

El catálogo y los modelos no se almacenarán inicialmente en MySQL; permanecerán en Parquet y archivos de modelos.

## 19.2 Motor elegido

**MySQL 8.0 o superior** como base oficial.

Motivos:

- El propietario ya dispone de MySQL.
- PyMySQL evita dependencias de compilación u ODBC en Windows.
- SQLAlchemy proporciona una capa estable.
- El esquema solo necesita operaciones transaccionales sencillas.

SQL Server queda como alternativa, no se implementan ambos motores simultáneamente.

## 19.3 Qué debe hacerse manualmente

El propietario debe ejecutar una sola vez, con una cuenta administradora:

1. Crear la base de datos.
2. Crear el usuario de aplicación.
3. Conceder permisos limitados.
4. Guardar la contraseña fuera del repositorio.

OpenCode puede generar los scripts, pero no debe conocer ni guardar la contraseña real.

## 19.4 Seguimiento opcional y tolerancia a fallos

La aplicación debe arrancar aunque MySQL esté apagado.

Regla:

```text
TRACKING_ENABLED=false
→ no intentar conectar

TRACKING_ENABLED=true
→ probar conexión
→ si falla, registrar warning y continuar sin tracking
```

Una falla de tracking nunca debe impedir recomendaciones o predicciones.

---

# 20. Script exacto para MySQL

## 20.1 `database/mysql/00_create_database_and_user.sql`

Reemplazar `CAMBIAR_POR_UNA_PASSWORD_LARGA` antes de ejecutarlo. Ejecutar como `root` o administrador en MySQL Workbench.

```sql
-- Spotify Music Intelligence
-- MySQL 8.0+
-- Ejecutar una sola vez con una cuenta administradora.

CREATE DATABASE IF NOT EXISTS spotify_music_intelligence
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'spotify_app'@'localhost'
    IDENTIFIED BY 'CAMBIAR_POR_UNA_PASSWORD_LARGA';

GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, REFERENCES
    ON spotify_music_intelligence.*
    TO 'spotify_app'@'localhost';

FLUSH PRIVILEGES;

SHOW DATABASES LIKE 'spotify_music_intelligence';
SHOW GRANTS FOR 'spotify_app'@'localhost';
```

No usar `'%'` como host durante desarrollo local.

## 20.2 `database/mysql/01_create_schema.sql`

Ejecutar conectado como `spotify_app`.

```sql
USE spotify_music_intelligence;

CREATE TABLE IF NOT EXISTS app_sessions (
    session_id CHAR(36) NOT NULL,
    started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    last_seen_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    app_version VARCHAR(50) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    PRIMARY KEY (session_id),
    INDEX idx_app_sessions_last_seen (last_seen_at),
    CONSTRAINT chk_app_sessions_environment
        CHECK (environment IN ('development', 'test', 'production'))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation_events (
    event_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    recommendation_type VARCHAR(20) NOT NULL,
    selected_track_id VARCHAR(100) NULL,
    selected_recording_group_id CHAR(64) NULL,
    selected_preset VARCHAR(100) NULL,
    query_payload JSON NOT NULL,
    requested_result_count SMALLINT UNSIGNED NOT NULL,
    returned_result_count SMALLINT UNSIGNED NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    latency_ms INT UNSIGNED NOT NULL,
    PRIMARY KEY (event_id),
    INDEX idx_recommendation_events_created_at (created_at),
    INDEX idx_recommendation_events_session (session_id),
    INDEX idx_recommendation_events_model (model_version),
    CONSTRAINT fk_recommendation_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_recommendation_type
        CHECK (recommendation_type IN ('track', 'preferences')),
    CONSTRAINT chk_requested_result_count
        CHECK (requested_result_count BETWEEN 1 AND 100),
    CONSTRAINT chk_returned_result_count
        CHECK (returned_result_count BETWEEN 0 AND 100)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS recommendation_results (
    event_id CHAR(36) NOT NULL,
    result_position SMALLINT UNSIGNED NOT NULL,
    recommended_recording_group_id CHAR(64) NOT NULL,
    recommended_track_id VARCHAR(100) NOT NULL,
    similarity_score DECIMAL(10, 9) NULL,
    distance_score DECIMAL(14, 9) NULL,
    explanation_payload JSON NULL,
    PRIMARY KEY (event_id, result_position),
    INDEX idx_recommendation_results_recording (recommended_recording_group_id),
    CONSTRAINT fk_recommendation_results_event
        FOREIGN KEY (event_id)
        REFERENCES recommendation_events(event_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_result_position
        CHECK (result_position BETWEEN 1 AND 100),
    CONSTRAINT chk_similarity_score
        CHECK (similarity_score IS NULL OR similarity_score BETWEEN -1.0 AND 1.0),
    CONSTRAINT chk_distance_score
        CHECK (distance_score IS NULL OR distance_score >= 0.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS classifier_events (
    event_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    classifier_type VARCHAR(20) NOT NULL,
    selected_track_id VARCHAR(100) NOT NULL,
    selected_recording_group_id CHAR(64) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    true_labels_payload JSON NULL,
    threshold_value DECIMAL(8, 6) NULL,
    latency_ms INT UNSIGNED NOT NULL,
    PRIMARY KEY (event_id),
    INDEX idx_classifier_events_created_at (created_at),
    INDEX idx_classifier_events_session (session_id),
    INDEX idx_classifier_events_model (model_version),
    CONSTRAINT fk_classifier_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_classifier_type
        CHECK (classifier_type IN ('multilabel', 'multiclass')),
    CONSTRAINT chk_threshold_value
        CHECK (threshold_value IS NULL OR threshold_value BETWEEN 0.0 AND 1.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS classifier_predictions (
    event_id CHAR(36) NOT NULL,
    prediction_rank SMALLINT UNSIGNED NOT NULL,
    genre VARCHAR(100) NOT NULL,
    score DECIMAL(10, 9) NOT NULL,
    passed_threshold BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (event_id, prediction_rank),
    INDEX idx_classifier_predictions_genre (genre),
    CONSTRAINT fk_classifier_predictions_event
        FOREIGN KEY (event_id)
        REFERENCES classifier_events(event_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_prediction_rank
        CHECK (prediction_rank BETWEEN 1 AND 114),
    CONSTRAINT chk_prediction_score
        CHECK (score BETWEEN 0.0 AND 1.0)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_id CHAR(36) NOT NULL,
    session_id CHAR(36) NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    source_event_id CHAR(36) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    target_recording_group_id CHAR(64) NULL,
    feedback_value TINYINT NOT NULL,
    reason_code VARCHAR(50) NULL,
    PRIMARY KEY (feedback_id),
    INDEX idx_feedback_events_created_at (created_at),
    INDEX idx_feedback_events_source (source_event_id, source_type),
    CONSTRAINT fk_feedback_events_session
        FOREIGN KEY (session_id)
        REFERENCES app_sessions(session_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT chk_feedback_source_type
        CHECK (source_type IN ('recommendation_list', 'recommendation_item', 'classifier')),
    CONSTRAINT chk_feedback_value
        CHECK (feedback_value IN (-1, 1))
) ENGINE=InnoDB;

-- Los índices se crean dentro de cada CREATE TABLE para que el script sea reejecutable.
```

El script puede volver a ejecutarse porque las tablas e índices se crean conjuntamente con `CREATE TABLE IF NOT EXISTS`. Los cambios de esquema posteriores deben gestionarse mediante scripts de migración versionados.

## 20.3 `database/mysql/02_verify_schema.sql`

```sql
USE spotify_music_intelligence;

SELECT DATABASE() AS current_database;

SHOW TABLES;

SELECT
    table_name,
    engine,
    table_collation
FROM information_schema.tables
WHERE table_schema = 'spotify_music_intelligence'
ORDER BY table_name;

SELECT
    constraint_name,
    table_name,
    constraint_type
FROM information_schema.table_constraints
WHERE table_schema = 'spotify_music_intelligence'
ORDER BY table_name, constraint_type, constraint_name;
```

## 20.4 Variables locales

`.env.example`:

```dotenv
APP_ENV=development
APP_VERSION=0.1.0
DATASET_PATH=data/raw/dataset.csv
TRACKING_ENABLED=false
DATABASE_URL=mysql+pymysql://spotify_app:CAMBIAR_PASSWORD@127.0.0.1:3306/spotify_music_intelligence?charset=utf8mb4
RANDOM_STATE=42
```

Crear `.env` local y no versionarlo.

## 20.5 Prueba manual de conexión

```powershell
uv run python -c "from sqlalchemy import create_engine; import os; from dotenv import load_dotenv; load_dotenv(); engine=create_engine(os.environ['DATABASE_URL'], pool_pre_ping=True); print(engine.connect().exec_driver_sql('SELECT 1').scalar())"
```

Resultado esperado:

```text
1
```

---

# 21. Alternativa exacta para SQL Server

No implementar MySQL y SQL Server a la vez. Esta sección solo se usa si se decide formalmente sustituir MySQL.

Requisitos adicionales:

- SQL Server con autenticación mixta habilitada si se utilizará login SQL.
- Microsoft ODBC Driver 18 for SQL Server instalado.
- Dependencia Python `pyodbc`.

## 21.1 `database/sqlserver/00_create_database_and_login.sql`

Ejecutar en SQL Server Management Studio con una cuenta administradora. Cambiar la contraseña.

```sql
USE master;
GO

IF DB_ID(N'SpotifyMusicIntelligence') IS NULL
BEGIN
    CREATE DATABASE SpotifyMusicIntelligence;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.server_principals
    WHERE name = N'spotify_app'
)
BEGIN
    CREATE LOGIN spotify_app
        WITH PASSWORD = 'CAMBIAR_POR_UNA_PASSWORD_LARGA',
        CHECK_POLICY = ON,
        CHECK_EXPIRATION = OFF;
END;
GO

USE SpotifyMusicIntelligence;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.database_principals
    WHERE name = N'spotify_app'
)
BEGIN
    CREATE USER spotify_app FOR LOGIN spotify_app;
END;
GO

ALTER ROLE db_datareader ADD MEMBER spotify_app;
ALTER ROLE db_datawriter ADD MEMBER spotify_app;
GRANT CREATE TABLE TO spotify_app;
GRANT ALTER TO spotify_app;
GO
```

## 21.2 `database/sqlserver/01_create_schema.sql`

```sql
USE SpotifyMusicIntelligence;
GO

IF OBJECT_ID(N'dbo.app_sessions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.app_sessions (
        session_id UNIQUEIDENTIFIER NOT NULL,
        started_at DATETIME2(6) NOT NULL CONSTRAINT df_app_sessions_started_at DEFAULT SYSUTCDATETIME(),
        last_seen_at DATETIME2(6) NOT NULL CONSTRAINT df_app_sessions_last_seen_at DEFAULT SYSUTCDATETIME(),
        app_version NVARCHAR(50) NOT NULL,
        environment NVARCHAR(20) NOT NULL,
        CONSTRAINT pk_app_sessions PRIMARY KEY (session_id),
        CONSTRAINT chk_app_sessions_environment CHECK (environment IN (N'development', N'test', N'production'))
    );
END;
GO

IF OBJECT_ID(N'dbo.recommendation_events', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.recommendation_events (
        event_id UNIQUEIDENTIFIER NOT NULL,
        session_id UNIQUEIDENTIFIER NOT NULL,
        created_at DATETIME2(6) NOT NULL CONSTRAINT df_recommendation_events_created_at DEFAULT SYSUTCDATETIME(),
        recommendation_type NVARCHAR(20) NOT NULL,
        selected_track_id NVARCHAR(100) NULL,
        selected_recording_group_id CHAR(64) NULL,
        selected_preset NVARCHAR(100) NULL,
        query_payload NVARCHAR(MAX) NOT NULL,
        requested_result_count SMALLINT NOT NULL,
        returned_result_count SMALLINT NOT NULL,
        model_version NVARCHAR(100) NOT NULL,
        latency_ms INT NOT NULL,
        CONSTRAINT pk_recommendation_events PRIMARY KEY (event_id),
        CONSTRAINT fk_recommendation_events_session FOREIGN KEY (session_id)
            REFERENCES dbo.app_sessions(session_id) ON DELETE CASCADE,
        CONSTRAINT chk_recommendation_type CHECK (recommendation_type IN (N'track', N'preferences')),
        CONSTRAINT chk_recommendation_query_json CHECK (ISJSON(query_payload) = 1),
        CONSTRAINT chk_requested_result_count CHECK (requested_result_count BETWEEN 1 AND 100),
        CONSTRAINT chk_returned_result_count CHECK (returned_result_count BETWEEN 0 AND 100),
        CONSTRAINT chk_recommendation_latency CHECK (latency_ms >= 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.recommendation_results', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.recommendation_results (
        event_id UNIQUEIDENTIFIER NOT NULL,
        result_position SMALLINT NOT NULL,
        recommended_recording_group_id CHAR(64) NOT NULL,
        recommended_track_id NVARCHAR(100) NOT NULL,
        similarity_score DECIMAL(10,9) NULL,
        distance_score DECIMAL(14,9) NULL,
        explanation_payload NVARCHAR(MAX) NULL,
        CONSTRAINT pk_recommendation_results PRIMARY KEY (event_id, result_position),
        CONSTRAINT fk_recommendation_results_event FOREIGN KEY (event_id)
            REFERENCES dbo.recommendation_events(event_id) ON DELETE CASCADE,
        CONSTRAINT chk_result_position CHECK (result_position BETWEEN 1 AND 100),
        CONSTRAINT chk_similarity_score CHECK (similarity_score IS NULL OR similarity_score BETWEEN -1.0 AND 1.0),
        CONSTRAINT chk_distance_score CHECK (distance_score IS NULL OR distance_score >= 0.0),
        CONSTRAINT chk_explanation_json CHECK (explanation_payload IS NULL OR ISJSON(explanation_payload) = 1)
    );
END;
GO

IF OBJECT_ID(N'dbo.classifier_events', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.classifier_events (
        event_id UNIQUEIDENTIFIER NOT NULL,
        session_id UNIQUEIDENTIFIER NOT NULL,
        created_at DATETIME2(6) NOT NULL CONSTRAINT df_classifier_events_created_at DEFAULT SYSUTCDATETIME(),
        classifier_type NVARCHAR(20) NOT NULL,
        selected_track_id NVARCHAR(100) NOT NULL,
        selected_recording_group_id CHAR(64) NOT NULL,
        model_version NVARCHAR(100) NOT NULL,
        true_labels_payload NVARCHAR(MAX) NULL,
        threshold_value DECIMAL(8,6) NULL,
        latency_ms INT NOT NULL,
        CONSTRAINT pk_classifier_events PRIMARY KEY (event_id),
        CONSTRAINT fk_classifier_events_session FOREIGN KEY (session_id)
            REFERENCES dbo.app_sessions(session_id) ON DELETE CASCADE,
        CONSTRAINT chk_classifier_type CHECK (classifier_type IN (N'multilabel', N'multiclass')),
        CONSTRAINT chk_true_labels_json CHECK (true_labels_payload IS NULL OR ISJSON(true_labels_payload) = 1),
        CONSTRAINT chk_threshold_value CHECK (threshold_value IS NULL OR threshold_value BETWEEN 0.0 AND 1.0),
        CONSTRAINT chk_classifier_latency CHECK (latency_ms >= 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.classifier_predictions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.classifier_predictions (
        event_id UNIQUEIDENTIFIER NOT NULL,
        prediction_rank SMALLINT NOT NULL,
        genre NVARCHAR(100) NOT NULL,
        score DECIMAL(10,9) NOT NULL,
        passed_threshold BIT NOT NULL CONSTRAINT df_classifier_predictions_passed DEFAULT 0,
        CONSTRAINT pk_classifier_predictions PRIMARY KEY (event_id, prediction_rank),
        CONSTRAINT fk_classifier_predictions_event FOREIGN KEY (event_id)
            REFERENCES dbo.classifier_events(event_id) ON DELETE CASCADE,
        CONSTRAINT chk_prediction_rank CHECK (prediction_rank BETWEEN 1 AND 114),
        CONSTRAINT chk_prediction_score CHECK (score BETWEEN 0.0 AND 1.0)
    );
END;
GO

IF OBJECT_ID(N'dbo.feedback_events', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.feedback_events (
        feedback_id UNIQUEIDENTIFIER NOT NULL,
        session_id UNIQUEIDENTIFIER NOT NULL,
        created_at DATETIME2(6) NOT NULL CONSTRAINT df_feedback_events_created_at DEFAULT SYSUTCDATETIME(),
        source_event_id UNIQUEIDENTIFIER NOT NULL,
        source_type NVARCHAR(30) NOT NULL,
        target_recording_group_id CHAR(64) NULL,
        feedback_value SMALLINT NOT NULL,
        reason_code NVARCHAR(50) NULL,
        CONSTRAINT pk_feedback_events PRIMARY KEY (feedback_id),
        CONSTRAINT fk_feedback_events_session FOREIGN KEY (session_id)
            REFERENCES dbo.app_sessions(session_id) ON DELETE CASCADE,
        CONSTRAINT chk_feedback_source_type CHECK (source_type IN (N'recommendation_list', N'recommendation_item', N'classifier')),
        CONSTRAINT chk_feedback_value CHECK (feedback_value IN (-1, 1))
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_recommendation_events_created_at')
    CREATE INDEX idx_recommendation_events_created_at ON dbo.recommendation_events(created_at);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_recommendation_results_recording')
    CREATE INDEX idx_recommendation_results_recording ON dbo.recommendation_results(recommended_recording_group_id);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_classifier_events_created_at')
    CREATE INDEX idx_classifier_events_created_at ON dbo.classifier_events(created_at);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'idx_classifier_predictions_genre')
    CREATE INDEX idx_classifier_predictions_genre ON dbo.classifier_predictions(genre);
GO
```

Cadena SQLAlchemy de ejemplo:

```dotenv
DATABASE_URL=mssql+pyodbc://spotify_app:CAMBIAR_PASSWORD@localhost/SpotifyMusicIntelligence?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
```

---

# 22. Configuración inicial de OpenCode

## 22.1 `opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "default_agent": "plan",
  "share": "disabled",
  "snapshot": true,
  "permission": {
    "edit": "ask",
    "bash": "ask"
  },
  "watcher": {
    "ignore": [
      ".git/**",
      ".venv/**",
      "data/raw/**",
      "data/processed/**",
      "models/**",
      "reports/figures/**",
      "__pycache__/**",
      ".pytest_cache/**",
      ".ruff_cache/**"
    ]
  },
  "instructions": [
    "AGENTS.md"
  ]
}
```

## 22.2 Agentes recomendados

### `data-auditor`

- Puede leer datos y código.
- Puede crear reportes.
- No modifica `data/raw`.
- No entrena modelos.

### `ml-engineer`

- Implementa pipelines y modelos.
- Puede ejecutar entrenamientos autorizados.
- No evalúa test salvo orden explícita.

### `model-reviewer`

- Solo lectura.
- Revisa fuga, métricas, configuración y reproducibilidad.
- No edita archivos.

## 22.3 Comandos recomendados

### `/validate-data`

```text
Ejecuta las pruebas de contratos de datos y el script validate_processed_data.py.
No modifiques archivos. Detente ante cualquier diferencia de conteos, hashes o esquema.
```

### `/test`

```text
Ejecuta uv run ruff check ., uv run ruff format --check . y uv run pytest.
Resume fallos con archivo, causa probable y corrección mínima.
```

### `/train-baseline`

```text
Valida datos y splits. Entrena solamente el baseline indicado en configs/model_parameters.yaml.
No uses test. Guarda configuración, métricas, tiempos y manifiesto.
```

### `/review-leakage`

```text
Comprueba intersecciones de recording_group_id, ajuste de transformadores, columnas prohibidas y uso del test.
No modifiques archivos.
```

---

# 23. Convenciones de código

## 23.1 Python

- Tipado en funciones públicas.
- Docstrings breves en módulos y funciones públicas.
- No usar variables globales mutables.
- No capturar `Exception` sin registrar contexto.
- No usar rutas absolutas.
- Utilizar `pathlib.Path`.
- Utilizar logging, no `print`, salvo scripts CLI con salida intencional.
- Toda semilla aleatoria parte de `random_state = 42` salvo experimento documentado.
- Funciones pequeñas y de responsabilidad única.
- Evitar duplicar lógica entre notebook, script y aplicación.

## 23.2 Formato

Ruff será la fuente de verdad para linting y formato.

Comandos:

```powershell
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
```

OpenCode no debe aplicar `--fix` en todo el repositorio sin revisar el diff.

## 23.3 Configuración

- Parámetros de datos en `configs/data_rules.yaml`.
- Features en archivos de features.
- Presets en `configs/presets.yaml`.
- Hiperparámetros en `configs/model_parameters.yaml`.
- No hardcodear configuraciones experimentales dentro de funciones.

## 23.4 Errores

Definir excepciones específicas cuando corresponda:

```text
DataContractError
ArtifactNotFoundError
IncompatibleArtifactError
InvalidPreferenceProfileError
TrackingUnavailableError
```

---

# 24. Política de entrenamiento

## 24.1 Regla general

```text
configuración aprobada
→ validación de datos
→ validación de splits
→ entrenamiento
→ validación
→ guardado de artefactos
→ reporte
→ revisión humana
```

## 24.2 Nunca hacer

```text
fit_transform sobre todo el dataset antes del split
usar test para elegir modelo
mezclar grabaciones entre splits
sobrescribir un artefacto sin nueva versión
comparar modelos con splits distintos
entrenar sin guardar configuración
```

## 24.3 Identidad de experimento

Formato:

```text
YYYYMMDD-HHMM_<task>_<model>_<short_hash>
```

Ejemplo:

```text
20260914-1930_multilabel_ovr-logreg_a81f32c
```

## 24.4 Manifiesto mínimo

```json
{
  "experiment_id": "...",
  "task": "multilabel",
  "model": "OneVsRest LogisticRegression",
  "dataset_sha256": "...",
  "split_sha256": "...",
  "config_sha256": "...",
  "git_commit": "...",
  "random_state": 42,
  "test_used": false,
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "training_seconds": 0.0,
  "artifact_path": "..."
}
```

## 24.5 Entrenamientos largos

Para modelos de árboles grandes o XGBoost:

- Ejecutar desde una terminal o configuración de PyCharm independiente.
- Guardar logs incrementalmente.
- No depender de que la sesión de OpenCode permanezca abierta.
- Limitar paralelismo para evitar bloquear el equipo.
- Registrar uso aproximado de memoria cuando sea posible.

---

# 25. Pruebas obligatorias

## 25.1 Datos

- Columnas requeridas.
- Tipos válidos.
- Rangos.
- Hash del raw sin cambios.
- Una fila por `track_id` en tracks.
- Una fila por grupo en recordings.
- Toda ID válida tiene grupo.
- Sin duplicados `track_id`–género.
- Conteo de géneros igual a 114.

## 25.2 Identidad

- Normalización determinista.
- Huella estable.
- SHA-256 estable.
- El mismo input produce el mismo grupo.
- Versiones acústicamente distintas no se fusionan bajo casos de prueba.

## 25.3 Splits

```text
train ∩ validation = vacío
train ∩ test = vacío
validation ∩ test = vacío
```

- El test está congelado.
- Las etiquetas mantienen orden estable.

## 25.4 Recomendadores

- No recomendar la propia grabación.
- No repetir grupos.
- Cumplir filtros.
- Ignorar peso 0.
- Rechazar todos los pesos 0.
- Manejar audio incompleto.
- Manejar resultados insuficientes.
- Mantener orden estable ante empate.

## 25.5 Clasificadores

- Misma transformación en entrenamiento e inferencia.
- No incluir columnas prohibidas.
- Dimensión de salida igual a 114.
- Top-k ordenado.
- Threshold cargado.
- Métricas reproducibles.

## 25.6 Persistencia

- App funciona con tracking deshabilitado.
- App continúa si DB falla.
- Inserción de evento y resultados en una transacción.
- Rollback ante error.
- No almacenar datos personales.

## 25.7 Streamlit

- Importación de todas las páginas.
- Mensajes ante artefactos faltantes.
- No entrenamiento al importar.
- Caché aplicado a recursos.

---

# 26. Git y actualizaciones del repositorio

## 26.1 Ramas

```text
main
feature/data-audit
feature/identity-resolution
feature/track-recommender
feature/preference-recommender
feature/multilabel-classifier
feature/multiclass-classifier
feature/streamlit-app
feature/event-tracking
chore/ci-and-docs
```

## 26.2 Commits

Formato recomendado:

```text
type(scope): descripción imperativa
```

Ejemplos:

```text
chore(project): initialize Python package and tooling
feat(data): add reproducible CSV audit
feat(identity): generate exact recording groups
feat(recommender): add cosine nearest-neighbor baseline
feat(classifier): train multilabel logistic baseline
test(data): validate split isolation
fix(app): handle missing recommender artifact
docs(methodology): document genre overlap limitations
```

## 26.3 Antes de cada commit

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## 26.4 Archivos no versionados

```text
.env
.streamlit/secrets.toml
.venv/
data/raw/dataset.csv
data/interim/*
data/processed/*
models/**/*.joblib
models/**/*.npy
*.log
```

Agregar archivos generados solo cuando exista una decisión explícita sobre tamaño y licencia.

## 26.5 Push al finalizar un módulo

Al completar un módulo, OpenCode debe entregar al propietario las instrucciones
para publicar los cambios en la rama `main`, pero **no debe ejecutar** `git push`
por su cuenta.

Regla:

```text
Al finalizar un módulo, mostrar en la respuesta final:
1. Un resumen de los archivos modificados y creados.
2. Los comandos exactos para commitear y publicar:
   git add .
   git commit -m "type(scope): descripción"
   git push origin main
3. Recordatorio de que el push es responsabilidad exclusiva del propietario.
```

OpenCode nunca ejecuta `git push` ni `git push origin main` en nombre del
propietario.

---

# 27. Plan de trabajo

El plan se organiza por módulos en lugar de jornadas fijas. El ritmo se adapta a la disponibilidad del propietario; cada módulo se completa con pruebas y un commit pequeño antes de avanzar al siguiente.

## Módulo 0 — Repositorio y reglas ✅

**Tareas**

- Crear repositorio GitHub vacío.
- Clonarlo.
- Copiar este archivo como `AGENTS.md`.
- Crear `.gitignore` y README mínimo.
- Instalar/configurar OpenCode con permisos `ask`.

**Aceptación**

- Repo abre en PyCharm.
- OpenCode reconoce `AGENTS.md`.
- `git status` limpio después del commit.

**Commit**

```text
chore(project): initialize repository and agent rules
```

## Módulo 0 — Python y uv ✅

**Tareas**

- Instalar uv.
- Fijar Python 3.12.
- Crear `pyproject.toml`.
- Crear `.venv` y `uv.lock`.
- Configurar intérprete en PyCharm.

**Aceptación**

```powershell
uv run python --version
uv sync --locked
```

**Commit**

```text
chore(environment): configure Python 3.12 and uv
```

## Módulo 0 — Estructura mínima y calidad ✅

**Tareas**

- Crear `src/`, `tests/`, `configs/`, `scripts/`, `data/`.
- Configurar Ruff, pytest y cobertura.
- Crear prueba de humo.

**Aceptación**

- Ruff y pytest pasan.

**Commit**

```text
chore(project): add source layout and quality tools
```

## Módulo 1 — Dataset y contrato inicial ✅

**Tareas**

- Colocar `dataset.csv` en `data/raw/`.
- Crear `data/README.md`.
- Calcular SHA-256.
- Definir columnas requeridas.

**Aceptación**

- El raw está ignorado por Git.
- El hash se registra en un reporte local.

**Commit**

```text
docs(data): document dataset acquisition and raw data policy
```

## Módulo 1 — Cargador y validación básica ✅

**Tareas**

- Implementar `load.py` y `contracts.py`.
- Pruebas de archivo ausente, columnas faltantes y tipos.

**Aceptación**

- Carga 114.000 × 21.
- Falla correctamente con fixtures inválidos.

**Commit**

```text
feat(data): add typed dataset loader and contracts
```

## Módulo 1 — Auditoría reproducible ✅

**Tareas**

- Implementar conteos de nulos, géneros, duplicados y anomalías.
- Generar `data_quality_report.json`.
- Probar cifras verificadas.

**Commit**

```text
feat(data): generate reproducible quality audit
```

## Módulo 1 — Notebook de auditoría ✅

**Tareas**

- Crear `01_data_audit.ipynb`.
- Mostrar estructura por bloques y prevalencia.
- Documentar limitaciones.

**Commit**

```text
docs(data): add audited dataset notebook
```

## Módulo 2 — Limpieza y cuarentena ✅

**Tareas**

- Eliminar `Unnamed: 0` solo en procesados. ✅
- Cuarentena de fila inválida. ✅
- Indicadores de duración y audio incompleto. ✅
- Pruebas. ✅

**Commit**

```text
feat(data): clean records and quarantine invalid identity
```

## Módulo 2 — Catálogo por track_id ✅

**Tareas**

- Consolidar una fila por `track_id`. ✅
- Agregar popularidad min/max/mediana. ✅
- Crear relación track–género. ✅

**Commit**

```text
feat(data): build unique track catalog and genre bridge
```

## Módulo 2 — Relación de artistas ✅

**Tareas**

- Separar artistas por `;`. ✅
- Crear `track_artists.parquet`. ✅
- Mantener texto original. ✅
- Probar duplicados y nulos. ✅

**Commit**

```text
feat(data): create normalized track artist bridge
```

## Módulo 2 — Normalización exacta ✅

**Tareas**

- Implementar NFKC, casefold y espacios. ✅
- Conservar puntuación y orden de artistas. ✅
- Crear pruebas de casos límite. ✅

**Commit**

```text
feat(identity): add conservative text normalization
```

## Módulo 2 — Huellas y grupos exactos ✅

**Tareas**

- Implementar huella estable. ✅
- Generar SHA-256. ✅
- Medir conteo esperado. ✅
- Crear `recording_tracks`. ✅

**Aceptación**

- Conteo de referencia aproximado 83.881. ✅ (83.881 verificado)
- Sin IDs huérfanas. ✅

**Commit**

```text
feat(identity): generate exact recording groups
```

## Módulo 2 — Catálogo de grabaciones ✅

**Tareas**

- Crear `recordings.parquet`. ✅
- Elegir track representativo. ✅
- Unir etiquetas por grupo. ✅

**Commit**

```text
feat(identity): build canonical recording catalog
```

## Módulo 2 — Candidatos a casi duplicados ✅

**Tareas**

- Implementar generación de candidatos. ✅
- No fusionar. ✅
- Exportar reporte para revisión. ✅

**Commit**

```text
feat(identity): report near duplicate candidates
```

## Módulo 2 — Pipeline completo de preparación ✅

**Tareas**

- Crear `scripts/prepare_data.py`. ✅
- Crear manifiesto. ✅
- Prueba de integración end-to-end. ✅

**Commit**

```text
feat(data): add reproducible preparation pipeline
```

## Módulo 3 — Distribuciones y anomalías ✅

**Tareas**

- Notebook EDA de variables y duraciones. ✅
- Exportar figuras. ✅

**Commit**

```text
analysis(eda): profile audio features and anomalies
```

## Módulo 3 — Correlaciones y redundancia ✅

**Tareas**

- Correlaciones. ✅
- Comparar energía/volumen y energía/acústica. ✅
- Registrar limitaciones. ✅

**Commit**

```text
analysis(eda): analyze feature relationships
```

## Módulo 3 — Géneros y solapamientos ✅

**Tareas**

- Matriz de coocurrencia. ✅
- Pares con mayor solapamiento. ✅
- Documentar impacto en clasificación. ✅

**Commit**

```text
analysis(eda): analyze genre overlap and cooccurrence
```

## Módulo 4 — Baseline de vecinos ✅

**Tareas**

- Implementar R1 en notebook. ✅
- Probar semillas manuales. ✅
- Excluir audio incompleto y misma grabación. ✅

**Commit**

```text
experiment(recommender): add cosine neighbor baseline
```

## Módulo 4 — Comparación R1–R4 ✅

**Tareas**

- Ejecutar cuatro configuraciones. ✅
- Medir latencia, cobertura y casos manuales. ✅
- Aprobar configuración inicial de producción. ✅

**Commit**

```text
experiment(recommender): compare scaling and distance strategies
```

## Módulo 4 — Código productivo del recomendador ✅

**Tareas**

- Implementar `track_based.py`. ✅
- Guardar artefactos versionados. ✅
- CLI de construcción. ✅

**Commit**

```text
feat(recommender): implement track based service
```

## Módulo 4 — Explicaciones y filtros ✅

**Tareas**

- Implementar explicaciones. ✅
- Implementar filtros desactivados por defecto. ✅
- Pruebas unitarias. ✅

**Commit**

```text
feat(recommender): add filters and feature explanations
```

## Módulo 4 — Evaluación automática ✅

**Tareas**

- Cobertura, diversidad, duplicados, latencia. ✅
- Crear reporte JSON/CSV. ✅

**Commit**

```text
feat(recommender): add offline evaluation metrics
```

## Módulo 8 — Shell de Streamlit ✅

**Tareas**

- Router. ✅
- Home. ✅
- Metodología. ✅
- Carga de configuración y artefactos. ✅

**Commit**

```text
feat(app): create multipage Streamlit shell
```

## Módulo 8 — Página por canción ✅

**Tareas**

- Buscador desambiguado. ✅
- Resultados. ✅
- Explicaciones. ✅
- Descarga CSV. ✅

**Commit**

```text
feat(app): add track recommendation page
```

## Módulo 4/8 — Integración y caché ✅

**Tareas**

- Cachés. ✅
- Estados de error. ✅
- Pruebas de importación. ✅
- Medir latencia local.

**Commit**

```text
fix(app): harden recommender loading and cache behavior
```

## Módulo 4 — Despliegue V1

**Tareas**

- Exportar requirements.
- Configurar Community Cloud.
- Tracking deshabilitado.
- Crear release `v0.1.0`.

**Commit**

```text
chore(release): prepare track recommender v0.1.0
```

## Módulo 5 — Configuración de presets ✅

**Tareas**

- Crear `presets.yaml` con valores aprobados. ✅
- Validar rangos y pesos. ✅
- Pruebas. ✅

**Commit**

```text
feat(preferences): add editable preset configuration
```

## Módulo 5 — Distancia ponderada ✅

**Tareas**

- Implementar scoring. ✅
- Peso cero. ✅
- Validación de perfil. ✅
- Pruebas numéricas. ✅

**Commit**

```text
feat(preferences): implement weighted profile distance
```

## Módulo 5 — Perfil fuera de distribución ✅

**Tareas**

- Calcular referencia de distancias. ✅
- Implementar avisos p95 y p99. ✅

**Commit**

```text
feat(preferences): detect uncommon preference profiles
```

## Módulo 5 — Diversidad MMR ✅

**Tareas**

- Implementar reordenamiento opcional. ✅
- Comparar lambda. ✅
- Mantener ranking puro como predeterminado. ✅

**Commit**

```text
feat(preferences): add optional diversity reranking
```

## Módulo 8 — Página de preferencias ✅

**Tareas**

- Preset y modo manual. ✅
- Sliders y pesos. ✅
- Filtros. ✅
- Explicaciones. ✅

**Commit**

```text
feat(app): add preference recommendation page
```

## Módulo 5 — Evaluación y pruebas V2

**Tareas**

- Casos por preset.
- Cobertura y diversidad.
- Resultados insuficientes.

**Commit**

```text
test(preferences): validate presets filters and diversity
```

## Módulo 5 — Documentación V2

**Tareas**

- Metodología.
- Capturas.
- Limitaciones.
- Release `v0.2.0`.

**Commit**

```text
chore(release): prepare preference recommender v0.2.0
```

## Módulo 6 — Dataset de clasificación y splits ✅

**Tareas**

- Construir X/Y multietiqueta y subconjunto multiclase.
- Generar split congelado.
- Pruebas de fuga.

**Commit**

```text
feat(classifier): create grouped training datasets and splits
```

## Módulo 6 — Baseline de frecuencia y logística ✅

**Tareas**

- Entrenar M0 y M1.
- No usar test.
- Guardar métricas y tiempos.

**Commit**

```text
experiment(classifier): train multilabel logistic baseline
```

## Módulo 6 — Umbrales y métricas ✅

**Tareas**

- Optimizar threshold en validación.
- Calcular métricas multilabel.
- Crear reportes.

**Commit**

```text
feat(classifier): add multilabel threshold tuning and metrics
```

## Módulo 6 — Extra Trees y Random Forest ✅

**Tareas**

- Entrenar M3 y M4.
- Comparar con M1.
- Revisar memoria y latencia.

**Commit**

```text
experiment(classifier): compare multilabel tree ensembles
```

## Módulo 6 — Classifier Chain ✅

**Tareas**

- Entrenar tres cadenas.
- Promediar resultados.
- Documentar dependencia del orden.

**Commit**

```text
experiment(classifier): evaluate classifier chain ensemble
```

## Módulo 6 — XGBoost opcional y selección ✅

**Tareas**

- Ejecutar solo si recursos y baselines están completos.
- Seleccionar candidato final sin usar test.
- Autorizar evaluación final.

**Commit**

```text
experiment(classifier): finalize multilabel model comparison
```

## Módulo 7 — Clasificador multiclase ✅

**Tareas**

- Entrenar C0, C1, C2 y C3.
- Evaluar validación.
- Analizar confusiones.

**Commit**

```text
experiment(classifier): compare dominant genre models
```

## Módulo 7 — Evaluación final y model cards ✅

**Tareas**

- Congelar modelos.
- Ejecutar test una vez.
- Guardar métricas finales.
- Crear model cards.

**Commit**

```text
feat(classifier): publish final evaluation and model cards
```

## Módulo 8 — Laboratorios Streamlit ✅

**Tareas**

- Página multietiqueta. ✅
- Página de género dominante. ✅
- Mostrar puntuaciones y limitaciones. ✅

**Commit**

```text
feat(app): add genre classification laboratories
```

## Módulo 9 — MySQL y tracking opcional

**Parte manual**

- Ejecutar `00_create_database_and_user.sql`.
- Ejecutar `01_create_schema.sql`.
- Crear `.env`.
- Verificar conexión.

**Parte OpenCode**

- Implementar repositorios transaccionales.
- Probar fallback sin DB.
- Integrar eventos con `TRACKING_ENABLED`.

**Commit**

```text
feat(persistence): add optional MySQL event tracking
```

## Cierre — CI, despliegue y release final

**Tareas**

- GitHub Actions.
- Suite completa.
- Revisión de secretos.
- README final.
- Documentación de instalación.
- Despliegue con tracking deshabilitado o DB remota.
- Release `v1.0.0`.

**Aceptación final**

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=spotify_intelligence
uv run streamlit run streamlit_app.py
```

**Commit**

```text
chore(release): publish Spotify Music Intelligence v1.0.0
```

---

# 28. Criterios de cierre por versión

## V0 — Datos

- Pipeline reproducible.
- Raw intacto.
- Contratos y auditoría.
- Catálogo por track y grabación.
- Pruebas de identidad.

## V1 — Recomendador por canción

- App desplegada.
- Autorrecomendaciones 0.
- Duplicados de grupos 0.
- Filtros cumplidos 100 %.
- Artefactos versionados.

## V2 — Preferencias

- Presets editables.
- Pesos 0–3.
- Perfil manual.
- OOD.
- Diversidad opcional.
- Pruebas.

## V3 — Clasificación

- Split congelado sin fuga.
- Baseline lineal.
- Comparación con árboles.
- Métricas multietiqueta y multiclase.
- Test final ejecutado una vez.
- Model cards.

## V4 — Persistencia y madurez

- MySQL opcional.
- App tolerante a fallos de tracking.
- CI.
- README completo.
- Release final.

---

# 29. Decisiones cerradas

1. PyCharm es el IDE principal.
2. OpenCode es asistente de programación, no propietario de decisiones metodológicas.
3. Python 3.12 y uv son la base del entorno.
4. El CSV original es inmutable.
5. Se conserva `track_id` y se añade `recording_group_id`.
6. La unidad de modelado es `recording_group_id`.
7. La agrupación exacta es conservadora.
8. Casi duplicados no se fusionan automáticamente.
9. Audio incompleto se excluye del recomendador inicial.
10. Popularidad no participa en la similitud inicial.
11. El recomendador por canción usa vecinos y distancias.
12. El recomendador por preferencias usa distancia ponderada.
13. Variables no configuradas tienen peso 0.
14. El clasificador principal es multietiqueta.
15. El clasificador secundario es multiclase en grabaciones monoetiqueta.
16. Random Forest, Extra Trees y XGBoost se reservan para clasificación.
17. XGBoost es opcional y posterior a baselines.
18. El test se congela y no se usa durante selección.
19. Streamlit carga artefactos; no entrena en cada clic.
20. MySQL es opcional y se activa después del producto funcional.
21. La app funciona sin base de datos.
22. Power BI queda fuera de alcance.
23. No se almacenan datos personales.
24. No se presentan similitudes como probabilidades de gusto.
25. No se presentan puntuaciones no calibradas como probabilidades.

---

# 30. Decisiones que dependen de experimentación

No fijar como resultado antes de medir:

- Mejor escalador.
- Mejor distancia.
- Selección final de características.
- Umbral final de casi duplicados.
- Umbral multietiqueta final.
- Mejor modelo supervisado.
- Efecto de imputación de audio incompleto.
- Latencia en despliegue.
- Cobertura y diversidad finales.
- Beneficio de XGBoost.
- Valor final de lambda MMR.

Cada decisión experimental debe quedar en un reporte con datos, configuración y limitaciones.

---

# 31. Lista final antes de pedir una tarea a OpenCode

El propietario debe especificar:

```text
Objetivo de la tarea
Módulo del plan
Archivos permitidos
Filtros aprobados
Características aprobadas
Parámetros aprobados
Comandos autorizados
Pruebas esperadas
Si puede o no entrenar
Si puede o no modificar configs
```

Plantilla de prompt:

```text
Trabaja en el módulo __ del plan de AGENTS.md.
Objetivo: __.
Archivos permitidos: __.
No modifiques: __.
Configuración aprobada: __.
Puedes ejecutar: __.
No uses el conjunto de prueba.
Primero presenta un plan y espera aprobación antes de editar.
Al terminar, ejecuta las pruebas específicas y muestra el diff.
```

---


# 32. Archivos de configuración iniciales

Los siguientes bloques son la especificación mínima que OpenCode debe materializar. Puede adaptar comentarios o formato, pero no cambiar valores sin aprobación.

## 32.1 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "spotify-music-intelligence"
version = "0.1.0"
description = "Content-based music recommendation and genre classification project"
readme = "README.md"
requires-python = ">=3.12,<3.13"
authors = [
  { name = "Alejandro José Tineo Carranza" }
]
dependencies = [
  "pandas>=2.2,<3",
  "numpy>=2,<3",
  "pyarrow>=17,<25",
  "scipy>=1.13,<2",
  "scikit-learn>=1.6,<2",
  "joblib>=1.4,<2",
  "pyyaml>=6,<7",
  "rapidfuzz>=3.9,<4",
  "streamlit>=1.45,<2",
  "plotly>=5.24,<7",
  "matplotlib>=3.9,<4",
  "sqlalchemy>=2.0,<3",
  "pymysql>=1.1,<2",
  "cryptography>=43,<47",
  "python-dotenv>=1.0,<2"
]

[project.optional-dependencies]
ml-advanced = [
  "xgboost>=2.1,<4"
]

[dependency-groups]
dev = [
  "pytest>=8,<10",
  "pytest-cov>=5,<8",
  "ruff>=0.9,<1",
  "mypy>=1.13,<2",
  "jupyter>=1,<2",
  "ipykernel>=6,<8",
  "pre-commit>=4,<5"
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py312"
line-length = 100
extend-exclude = [
  ".venv",
  "data/raw",
  "data/interim",
  "data/processed",
  "models"
]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "C4"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
  "integration: tests that use multiple modules or external services",
  "slow: long-running tests or model training checks"
]

[tool.coverage.run]
source = ["spotify_intelligence"]
branch = true

[tool.coverage.report]
show_missing = true
skip_covered = true
fail_under = 75

[tool.mypy]
python_version = "3.12"
packages = ["spotify_intelligence"]
warn_unused_configs = true
warn_return_any = true
warn_unused_ignores = true
no_implicit_optional = true
check_untyped_defs = true
ignore_missing_imports = true
```

Después de crearlo:

```powershell
uv lock
uv sync --all-groups
uv export --format requirements.txt --no-dev --no-hashes -o requirements.txt
```

El lockfile resuelto es la fuente exacta de versiones. Los rangos anteriores no sustituyen a `uv.lock`.

## 32.2 `configs/data_rules.yaml`

```yaml
version: "1.0"
random_state: 42

paths:
  raw_dataset: "data/raw/dataset.csv"
  quarantine_dir: "data/quarantine"
  interim_dir: "data/interim"
  processed_dir: "data/processed"

required_columns:
  - "Unnamed: 0"
  - "track_id"
  - "artists"
  - "album_name"
  - "track_name"
  - "popularity"
  - "duration_ms"
  - "explicit"
  - "danceability"
  - "energy"
  - "key"
  - "loudness"
  - "mode"
  - "speechiness"
  - "acousticness"
  - "instrumentalness"
  - "liveness"
  - "valence"
  - "tempo"
  - "time_signature"
  - "track_genre"

cleaning:
  drop_columns:
    - "Unnamed: 0"
  identity_required:
    - "track_id"
    - "artists"
    - "track_name"
  short_track_threshold_ms: 60000
  long_track_threshold_ms: 600000
  preserve_popularity_zero: true
  preserve_duration_outliers: true

identity:
  text_unicode_normalization: "NFKC"
  casefold: true
  trim: true
  collapse_whitespace: true
  remove_punctuation_exact_grouping: false
  sort_artists_exact_grouping: false
  fingerprint_hash: "sha256"
  expected_exact_recording_groups: 83881

incomplete_audio:
  required_zero_pattern:
    tempo: 0
    danceability: 0
    speechiness: 0
    valence: 0
    time_signature: 0
  exclude_from_recommenders: true
  classifier_baseline_strategy: "exclude"
```

## 32.3 `configs/recommender_features.yaml`

```yaml
version: "recommender-features-v1"

track_recommender:
  features:
    - "danceability"
    - "energy"
    - "loudness"
    - "speechiness"
    - "acousticness"
    - "instrumentalness"
    - "liveness"
    - "valence"
    - "tempo"
  baseline:
    scaler: "standard"
    metric: "cosine"
    algorithm: "brute"
  experiments:
    - scaler: "standard"
      metric: "cosine"
    - scaler: "robust"
      metric: "cosine"
    - scaler: "standard"
      metric: "euclidean"
    - scaler: "robust"
      metric: "euclidean"
  retrieval:
    default_top_n: 10
    min_top_n: 5
    max_top_n: 20
    initial_candidate_floor: 100
    candidate_multiplier: 10
    expansion_steps: [500, 2000]
  exclusions:
    audio_analysis_incomplete: true
    same_recording_group: true
  filters:
    explicit_default: "all"
    genre_default: "all"
    duration_enabled_default: false
    duration_suggested_min_seconds: 60
    duration_suggested_max_seconds: 600
    different_artist_default: false
    popularity_min_enabled_default: false

preference_recommender:
  basic_features:
    - "energy"
    - "danceability"
    - "valence"
    - "acousticness"
    - "instrumentalness"
    - "tempo"
  advanced_features:
    - "speechiness"
    - "liveness"
    - "loudness"
    - "duration_ms"
    - "key"
    - "mode"
    - "time_signature"
  weight_min: 0
  weight_max: 3
  reject_all_zero_weights: true
  distance: "weighted_euclidean"
  out_of_distribution:
    warning_percentile: 95
    weak_match_percentile: 99
  diversity:
    enabled_default: false
    method: "mmr"
    lambda_default: 0.85
```

## 32.4 `configs/classifier_features.yaml`

```yaml
version: "classifier-features-v1"

primary_audio_features:
  continuous:
    - "danceability"
    - "energy"
    - "loudness"
    - "speechiness"
    - "acousticness"
    - "instrumentalness"
    - "liveness"
    - "valence"
    - "tempo"
  engineered:
    - "log_duration"
    - "key_sin"
    - "key_cos"
  binary:
    - "mode"
  categorical_one_hot:
    - "time_signature"

context_variant:
  enabled: false
  additional_features:
    - "popularity_median"
    - "explicit"

forbidden_features:
  - "track_id"
  - "recording_group_id"
  - "track_name"
  - "artists"
  - "album_name"
  - "track_genre"

splits:
  train_fraction: 0.70
  validation_fraction: 0.15
  test_fraction: 0.15
  group_column: "recording_group_id"
  random_state: 42
  candidate_split_count: 50
  freeze_test: true

incomplete_audio_experiments:
  - "exclude"
  - "train_only_imputation_with_indicator"
```

## 32.5 `configs/model_parameters.yaml`

```yaml
version: "model-parameters-v1"
random_state: 42

multilabel:
  frequency_baseline:
    enabled: true

  ovr_logistic:
    enabled: true
    estimator:
      solver: "liblinear"
      C: 1.0
      max_iter: 2000
      class_weight: "balanced"
      random_state: 42
    wrapper_n_jobs: -1

  classifier_chain:
    enabled: true
    base_estimator: "logistic_regression"
    order: "random"
    seeds: [42, 43, 44]

  extra_trees:
    enabled: true
    n_estimators: 400
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 2
    max_features: "sqrt"
    bootstrap: false
    class_weight: null
    n_jobs: -1
    random_state: 42

  random_forest:
    enabled: true
    n_estimators: 400
    max_depth: null
    min_samples_split: 2
    min_samples_leaf: 2
    max_features: "sqrt"
    bootstrap: true
    class_weight: null
    n_jobs: -1
    random_state: 42

  xgboost_ovr:
    enabled: false
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.80
    colsample_bytree: 0.80
    min_child_weight: 1
    reg_lambda: 1.0
    objective: "binary:logistic"
    eval_metric: "logloss"
    tree_method: "hist"
    random_state: 42

  threshold_search:
    min: 0.10
    max: 0.90
    step: 0.05
    optimize_metric: "samples_f1"

multiclass:
  frequent_class_baseline:
    enabled: true

  logistic:
    enabled: true
    solver: "lbfgs"
    C: 1.0
    max_iter: 3000
    class_weight: "balanced"
    random_state: 42

  extra_trees:
    enabled: true
    n_estimators: 500
    max_depth: null
    min_samples_leaf: 2
    max_features: "sqrt"
    class_weight: "balanced"
    n_jobs: -1
    random_state: 42

  random_forest:
    enabled: true
    n_estimators: 500
    max_depth: null
    min_samples_leaf: 2
    max_features: "sqrt"
    class_weight: "balanced"
    n_jobs: -1
    random_state: 42

  xgboost:
    enabled: false
```

## 32.6 `configs/presets.yaml`

```yaml
version: "presets-v1"
weight_scale:
  min: 0
  max: 3

presets:
  entrenamiento_intenso:
    label: "Entrenamiento intenso"
    values:
      energy: 0.90
      danceability: 0.75
      valence: 0.65
      acousticness: 0.05
      instrumentalness: 0.05
      tempo: 145
    weights:
      energy: 3
      danceability: 2
      valence: 1
      acousticness: 2
      instrumentalness: 1
      tempo: 2

  fiesta:
    label: "Fiesta"
    values:
      energy: 0.85
      danceability: 0.90
      valence: 0.75
      acousticness: 0.05
      instrumentalness: 0.02
      tempo: 128
    weights:
      energy: 2
      danceability: 3
      valence: 2
      acousticness: 1
      instrumentalness: 1
      tempo: 2

  concentracion_instrumental:
    label: "Concentración instrumental"
    values:
      energy: 0.35
      danceability: 0.30
      valence: 0.45
      acousticness: 0.55
      instrumentalness: 0.85
      tempo: 90
    weights:
      energy: 1
      danceability: 1
      valence: 1
      acousticness: 2
      instrumentalness: 3
      tempo: 1

  relajacion:
    label: "Relajación"
    values:
      energy: 0.25
      danceability: 0.35
      valence: 0.55
      acousticness: 0.75
      instrumentalness: 0.45
      tempo: 80
    weights:
      energy: 2
      danceability: 1
      valence: 1
      acousticness: 2
      instrumentalness: 1
      tempo: 2

  alegre_y_bailable:
    label: "Alegre y bailable"
    values:
      energy: 0.72
      danceability: 0.85
      valence: 0.88
      acousticness: 0.08
      instrumentalness: 0.02
      tempo: 124
    weights:
      energy: 2
      danceability: 3
      valence: 3
      acousticness: 1
      instrumentalness: 1
      tempo: 2

  melancolico:
    label: "Melancólico"
    values:
      energy: 0.30
      danceability: 0.35
      valence: 0.15
      acousticness: 0.60
      instrumentalness: 0.20
      tempo: 85
    weights:
      energy: 2
      danceability: 1
      valence: 3
      acousticness: 2
      instrumentalness: 1
      tempo: 1

  acustico:
    label: "Acústico"
    values:
      energy: 0.40
      danceability: 0.45
      valence: 0.55
      acousticness: 0.90
      instrumentalness: 0.05
      tempo: 95
    weights:
      energy: 1
      danceability: 1
      valence: 1
      acousticness: 3
      instrumentalness: 1
      tempo: 1
```

## 32.7 `configs/app.yaml`

```yaml
version: "app-config-v1"
app:
  title: "Spotify Music Intelligence"
  page_icon: "🎵"
  layout: "wide"
  default_top_n: 10
  min_top_n: 5
  max_top_n: 20

tracking:
  enabled_default: false
  fail_open: true
  collect_personal_data: false

artifacts:
  strict_version_check: true
  train_on_missing: false
```

El icono puede cambiarse si se desea evitar emojis, sin afectar la lógica.

# 33. `.gitignore` mínimo

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# Environments
.venv/
.env

# IDE
.idea/

# Streamlit secrets
.streamlit/secrets.toml

# Jupyter
.ipynb_checkpoints/

# Raw and generated data
data/raw/*
!data/raw/.gitkeep
data/quarantine/*
!data/quarantine/.gitkeep
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Models and generated reports
models/**/*.joblib
models/**/*.npy
models/**/*.npz
models/**/*.pkl
reports/figures/*
reports/experiments/*

# Logs and OS
*.log
.DS_Store
Thumbs.db
```

# 34. Referencias oficiales operativas

- OpenCode, reglas y `AGENTS.md`: https://opencode.ai/docs/rules/
- OpenCode, permisos: https://opencode.ai/docs/permissions/
- OpenCode, agentes: https://opencode.ai/docs/agents/
- OpenCode, comandos: https://opencode.ai/docs/commands/
- uv, proyectos y lockfile: https://docs.astral.sh/uv/guides/projects/
- uv, exportación a requirements: https://docs.astral.sh/uv/concepts/projects/sync/
- Streamlit, secretos: https://docs.streamlit.io/develop/concepts/connections/secrets-management
- Streamlit, conexiones de datos: https://docs.streamlit.io/develop/concepts/connections/connecting-to-data
- MySQL, `CREATE DATABASE`: https://dev.mysql.com/doc/refman/8.0/en/create-database.html
- MySQL, creación de cuentas y privilegios: https://dev.mysql.com/doc/refman/8.0/en/creating-accounts.html
- SQL Server, `CREATE DATABASE`: https://learn.microsoft.com/sql/t-sql/statements/create-database-transact-sql
- SQL Server, `CREATE LOGIN`: https://learn.microsoft.com/sql/t-sql/statements/create-login-transact-sql
- SQL Server, `CREATE USER`: https://learn.microsoft.com/sql/t-sql/statements/create-user-transact-sql

---

# 35. Resumen operativo

```text
Datos brutos inmutables
→ auditoría reproducible
→ catálogo por track_id
→ recording_group_id exacto
→ datos procesados
→ recomendador por canción
→ recomendador por preferencias
→ clasificación multietiqueta
→ clasificación multiclase
→ Streamlit
→ tracking MySQL opcional
→ pruebas, CI, documentación y release
```

La regla central del proyecto es:

> **OpenCode automatiza la implementación y la ejecución controlada; las reglas de datos, filtros, características, parámetros, interpretación y aprobación final permanecen bajo control humano.**
