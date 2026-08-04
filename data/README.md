# Datos

## `raw/`

- `dataset.csv`: dataset original (*Spotify Tracks Dataset*). **No se versiona**
  (§8.1, §26.4) y permanece únicamente en el entorno local. Nunca se modifica (§2).
- Fuente: <https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset>
  (verificar la licencia exacta en la página de origen).

## `processed/`

- Parquet y manifiestos derivados por `scripts/prepare_data.py` y
  `scripts/create_splits.py`. Se versionan desde el despliegue V1 (Opción A, §27).
- Los datos derivados se distribuyen bajo la licencia del dataset fuente
  (habitualmente CC BY-NC-SA 4.0: uso no comercial, atribución y share-alike).

## `interim/` y `quarantine/`

- Temporales y datos descartados (identidad inválida). No versionados.
