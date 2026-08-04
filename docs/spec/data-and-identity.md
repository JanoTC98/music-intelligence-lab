# Datos e identidad

## Dataset

- **Ubicación:** `data/raw/dataset.csv` (inmutable).
- **SHA-256:** `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`.
- **Filas totales:** 114.000.
- **Filas con identidad inválida:** 1, enviada a `data/quarantine/invalid_identity.parquet`.
- **Filas válidas después de cuarentena:** 113.999.
- **`track_id` válidos consolidados:** 89.740.
- **`recording_group_id` válidos:** exactamente 83.881.

## Proceso de identidad

1. Consolidación por `track_id`.
2. Consolidación por huella exacta.
3. Puesta en cuarentena de filas con identidad inválida.
4. Asignación de `recording_group_id` mediante huella normalizada + SHA-256.

## Invariantes

- El dataset bruto no se modifica.
- Los splits se agrupan por `recording_group_id`.
- No se usan identificadores, nombres, artistas, álbumes ni la etiqueta objetivo como características del clasificador.
- El conjunto de prueba permanece congelado y no participa en selección de modelos ni umbrales.
- Los multigéneros no se fuerzan automáticamente a una única verdad.

## Regresión

Para el dataset cuyo SHA-256 es `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`, después de poner en cuarentena la única fila con identidad inválida, el pipeline debe producir exactamente 83.881 `recording_group_id`. Una diferencia debe hacer fallar la regresión; no es un conteo aproximado del CSV bruto.