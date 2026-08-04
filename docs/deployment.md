# Despliegue V1 — Streamlit Community Cloud

> Documento operativo del **Módulo 4 — Despliegue V1** (AGENTS.md §27) y §18.6.
> Alcance: recomendador por canción (V1). La aplicación carga artefactos; nunca
> entrena en la nube (§18.4).

## 1. Requisitos previos

- Repositorio en GitHub con la rama `main` actualizada.
- Artefactos generados localmente y versionados (ver §2).
- `requirements.txt` exportado desde `uv.lock` (ya versionado, §6.2).
- `.streamlit/config.toml` con `headless = true` (ya versionado).

## 2. Artefactos necesarios y decisión de versionado

La aplicación necesita dos grupos de archivos que **no se versionan por defecto**
(§26.4):

| Grupo | Ruta | Tamaño actual | Archivo mayor |
|---|---|---|---|
| Datos procesados | `data/processed/` | ≈ 58 MB | `tracks.parquet` (16,1 MB) |
| Recomendador por canción | `models/recommender/v1/` | ≈ 23,8 MB | `catalog_index.parquet` (12,3 MB) |
| Recomendador por preferencias | `models/preferences/v1/` | ≈ 15,6 MB | `catalog_index.parquet` (11,8 MB) |
| Clasificadores | `models/classifier/` | ≈ 1,3 MB | < 1 MB |

Total ≈ **100 MB**. Todos los archivos individuales están muy por debajo del
límite de 100 MB por archivo de GitHub.

### Opciones

- **Opción A (recomendada para V1): versionar artefactos.** Quitar
  `data/processed/*` y `models/**/*.joblib`/`.npy` de `.gitignore` y subirlos al
  repositorio. Community Cloud clona el repo y la app funciona directamente.
  **Decisión del propietario** (§26.4): implica añadir ≈ 100 MB al repositorio y
  requiere revisar la licencia de los datos derivados.
- **Opción B: no versionar nada.** La app arranca, pero cada página muestra
  *"El artefacto requerido no existe"*. No es apta para un MVP real.
- **Opción C: artefactos externos.** Guardar los archivos en un release asset o
  almacenamiento de objetos y descargarlos al arranque. Más compleja; se descarta
  para V1.

## 3. Pasos en Streamlit Community Cloud

1. Publicar la rama `main` en GitHub (responsabilidad del propietario, §26.5).
2. Entrar en <https://share.streamlit.io> y crear una nueva app conectada al
   repositorio.
3. Configurar:
   - **Repository**: `spotify-music-intelligence`.
   - **Branch**: `main`.
   - **Main file path**: `streamlit_app.py`.
   - **Python version**: 3.12.
4. Desplegar. Community Cloud instalará `requirements.txt` (versiones exactas del
   lockfile, §6.2).
5. **No se requieren secretos** para V1: el tracking está deshabilitado por
   defecto (`configs/app.yaml` → `tracking.enabled_default: false`) y la app no
   contiene código de tracking todavía (Módulo 9).

## 4. Verificación post-despliegue

- Página *Inicio* carga sin errores.
- *Recomendar por canción*: buscar una canción, seleccionarla y obtener el Top-N.
  La página incluye el toggle experimental *"Priorizar canciones del mismo género de
  la semilla"* (afinidad de género, §30), desactivado por defecto.
- *Recomendar por preferencias*: un preset produce resultados.
- *Laboratorios de género*: muestran puntuaciones de los modelos versionados.
- *Auditoría y catálogo*: muestra el reporte `data_quality_report.json`.

## 5. Release v0.1.0

1. Etiquetar y publicar en GitHub (responsabilidad del propietario):

   ```bash
   git tag -a v0.1.0 -m "Track recommender V1"
   git push origin v0.1.0
   ```

2. Crear el release `v0.1.0` en la interfaz de GitHub con una nota breve de
   alcance: recomendador por canción, aplicación multipágina y datos procesados.

## 6. Limitaciones del despliegue V1

- El tracking de eventos y feedback (Módulo 9) está deshabilitado; una base MySQL
  local no es accesible desde Community Cloud (§18.6). Se activará solo cuando
  exista una base remota accesible.
- No se entrena ningún modelo en la nube; todo se construye localmente y se
  versiona.
- El catálogo es la muestra de 114.000 filas del `dataset.csv`; las conclusiones
  no representan la prevalencia real de géneros en Spotify (§2.3).
