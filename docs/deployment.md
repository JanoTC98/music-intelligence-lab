# Despliegue V1 — Streamlit Community Cloud

> Documento operativo del **Módulo 4 — Despliegue V1**.
> Alcance: recomendador por canción (V1). La aplicación carga artefactos; nunca
> entrena en la nube.

## 1. Requisitos previos

- Repositorio en GitHub con la rama `main` actualizada.
- Artefactos generados localmente y versionados.
- `requirements.txt` exportado desde `uv.lock` (ya versionado).
- `.streamlit/config.toml` con `headless = true` (ya versionado).

## 2. Artefactos necesarios y decisión de versionado

**Decisión tomada: Opción A — versionar artefactos** (aprobada por el
propietario). Se versionan los datos procesados y los artefactos de producción
que la aplicación carga; los experimentos de clasificación y el CSV crudo
permanecen fuera de Git.

La aplicación necesita dos grupos de archivos que **no se versionaban por
defecto** y que la Opción A des-ignora en `.gitignore`:

| Grupo | Ruta | Tamaño versionado | Archivo mayor |
|---|---|---|---|
| Datos procesados | `data/processed/` | 58 MB (9 parquet + 2 JSON) | `tracks.parquet` (16,1 MB) |
| Recomendador por canción | `models/recommender/v1/` | 23,8 MB (nuevo: 11,8 MB) | `catalog_index.parquet` (12,3 MB) |
| Recomendador por preferencias | `models/preferences/v1/` | 15,6 MB (nuevo: 3,9 MB) | `catalog_index.parquet` (11,8 MB) |
| Clasificadores (serving) | `models/classifier/` (M1_A, M1_B y C1) | 0,2 MB (nuevo) | < 1 MB |

Total añadido ≈ **74 MB**. Todos los archivos individuales están muy por debajo
del límite de 100 MB por archivo de GitHub.

Notas:

- `data/raw/dataset.csv` **no** se versiona; solo viajan los parquet derivados.
  Los datos derivados se publican bajo la licencia del dataset fuente (ODbL 1.0:
  atribución requerida, ver la documentación de la carpeta `data/`).
- Los joblibs de experimentos de clasificación no expuestos en la aplicación
  (M0, M2 y C0) no se cargan y permanecen ignorados; M1_A, M1_B y C1 sí se
  versionan porque el laboratorio multietiqueta y el de género dominante los sirven.
- Opciones descartadas: **B** (sin artefactos, la app no funciona) y **C**
  (artefactos externos, complejidad innecesaria para V1).

## 3. Pasos en Streamlit Community Cloud

1. Publicar la rama `main` en GitHub (responsabilidad del propietario).
2. Entrar en <https://share.streamlit.io> y crear una nueva app conectada al
   repositorio.
3. Configurar:
   - **Repository**: `JanoTC98/music-intelligence-lab`.
   - **Branch**: `main`.
   - **Main file path**: `streamlit_app.py`.
   - **Python version**: 3.12.
4. Desplegar. Community Cloud instalará `requirements.txt` (versiones exactas del
   lockfile).
5. **No se requieren secretos** para V1: la persistencia de eventos y feedback en
   MySQL está fuera del alcance actual y la app no contiene código de tracking.

## 4. Verificación post-despliegue

- Página *Inicio* carga sin errores.
- *Recomendar por canción*: buscar una canción, seleccionarla y obtener el Top-N.
  La página incluye el toggle experimental *"Priorizar canciones del mismo género de
  la semilla"* (afinidad de género), desactivado por defecto.
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

- La persistencia de eventos y feedback en MySQL está **fuera del alcance actual**
  (ver `docs/roadmap.md`); no se implementa código de tracking.
- No se entrena ningún modelo en la nube; todo se construye localmente y se
  versiona.
- El catálogo es la muestra de 114.000 filas del `dataset.csv`; las conclusiones
  no representan la prevalencia real de géneros en Spotify.
