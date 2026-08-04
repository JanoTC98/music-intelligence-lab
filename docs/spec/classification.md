# Clasificación

## Multietiqueta (géneros)

- Laboratorio supervisado de clasificación multietiqueta de géneros.
- Usa features acústicas y el encoder de géneros.
- Modelos: Extra Trees y Random Forest.

## Multiclase (género acústico dominante)

- Experimento multiclase de género acústico dominante.
- Usa features acústicas normalizadas.
- Modelos: Extra Trees y Random Forest.

## Reglas

- No se usan `track_name`, `artists`, `album_name`, `track_id` ni `recording_group_id` como características.
- El conjunto de prueba permanece congelado y no participa en selección de modelos.
- Los hiperparámetros se leen de `configs/model_parameters.yaml`.
- Los modelos multiclase C2 y C3 usan `n_estimators: 300` y `max_depth: 12`.