# Arquitectura

## Implementado actualmente

### Código fuente
- `src/spotify_intelligence/` — Paquete importable con toda la lógica de producción.
  - `data/pipeline.py` — Pipeline de datos.
  - `data/clean.py` — Limpieza y normalización.
  - `data/splits.py` — División de datos por `recording_group_id`.
  - `data/contracts.py` — Contratos de validación.
  - `data/audit.py` — Auditoría de datos.
  - `data/validate.py` — Validación de datos.
  - `data/load.py` — Carga de datos.
  - `identity/` — Normalización y consolidación de identidad.
  - `features/` — Extracción de características acústicas.
  - `recommenders/` — Recomendadores por canción y por preferencias.
  - `classification/` — Clasificadores multietiqueta y multiclase.
  - `analysis/` — Análisis exploratorio.
  - `config.py` — Configuración centralizada.
- `app/` — Interfaz Streamlit multipágina.
- `configs/` — Configuración versionada en YAML.
- `scripts/` — Scripts CLI.
- `tests/` — Pruebas unitarias e integración.

### Datos y modelos
- `data/raw/` — Dataset original inmutable.
- `data/processed/` — Artefactos Parquet.
- `data/quarantine/` — Filas con identidad inválida.
- `models/` — Artefactos serializados.
- `reports/` — Métricas y figuras.

### Persistencia
- `database/` — Esquemas opcionales MySQL y SQL Server. La persistencia de eventos y feedback está **fuera del alcance actual**; no se implementa código de tracking.

## Planificado

- Módulos futuros identificados en el roadmap del proyecto.
- No se crean archivos Python vacíos únicamente para hacer coincidir el árbol histórico.