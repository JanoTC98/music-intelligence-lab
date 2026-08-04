# AGENTS.md — Spotify Music Intelligence

> **Estado:** especificación normativa de implementación
> **Versión:** 2.0
> **Creada originalmente:** 29 de julio de 2026
> **Última actualización:** 3 de agosto de 2026
> **Proyecto:** Spotify Music Intelligence
> **IDE principal:** PyCharm
> **Agente de programación:** OpenCode
> **Fuentes verificables:** dataset identificado por SHA-256, `configs/*.yaml`, `pyproject.toml`, `uv.lock`, código, pruebas y artefactos versionados.

---

## 1. Propósito y alcance

Este repositorio es un proyecto end-to-end de ciencia de datos y machine learning que transforma un catálogo musical en:

1. Un recomendador basado en una canción.
2. Un recomendador basado en preferencias y presets editables.
3. Un laboratorio supervisado de clasificación multietiqueta de géneros.
4. Un experimento multiclase de género acústico dominante.
5. Una aplicación web multipágina en Streamlit.
6. Un pipeline reproducible de datos y modelos.
7. Un sistema opcional de eventos y feedback almacenado en MySQL.
8. Un repositorio profesional con pruebas, documentación, CI y despliegue.

El producto central es el recomendador musical basado en contenido. Los clasificadores son módulos experimentales complementarios.

---

## 2. Reglas de trabajo y prohibiciones

### 2.1 Prohibiciones absolutas

- Modificar `data/raw/dataset.csv`.
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
- Reentrenar modelos dentro de una interacción de Streamlit.
- Inventar métricas, resultados, conteos o conclusiones.

### 2.2 Reglas de comportamiento del agente

1. Prohibiciones y reglas de seguridad del presente archivo.
2. Instrucción explícita del propietario para la tarea actual.
3. Convenciones operativas del repositorio.

---

## 3. Precedencia de información

### 3.1 Conducta del agente

1. Prohibiciones y reglas de seguridad del `AGENTS.md`.
2. Instrucción explícita del propietario para la tarea actual.
3. Convenciones operativas del repositorio.

### 3.2 Valores, esquemas y parámetros

1. Datos medidos del dataset identificado por su SHA-256 esperado.
2. `configs/*.yaml` para reglas y parámetros de producto/modelado.
3. `uv.lock` para versiones resueltas y `pyproject.toml` para dependencias declaradas.
4. `database/**/*.sql` para esquemas de persistencia.
5. Contratos y pruebas automatizadas.
6. Código productivo.
7. Documentación descriptiva y roadmap.

La documentación no debe copiar valores de configuración que ya existan en archivos ejecutables.

---

## 4. Comandos esenciales

```powershell
uv sync
uv run python --version
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/spotify_intelligence
```

---

## 5. Arquitectura del repositorio

### 5.1 Código fuente

- `src/spotify_intelligence/` — Paquete importable con toda la lógica de producción.
- `app/` — Interfaz Streamlit multipágina. `streamlit_app.py` actúa como router.
- `configs/` — Configuración versionada en YAML.
- `scripts/` — Scripts CLI pequeños que llaman funciones de `src/`.
- `notebooks/` — Solo exploración y experimentos.

### 5.2 Datos y modelos

- `data/raw/` — Dataset original inmutable.
- `data/processed/` — Artefactos Parquet resultantes del pipeline.
- `data/quarantine/` — Filas con identidad inválida.
- `models/` — Artefactos serializados de modelos.
- `reports/` — Métricas, figuras y documentos de resultados.

### 5.3 Calidad y pruebas

- `tests/unit/` y `tests/integration/` — Pruebas con pytest.
- `docs/spec/` — Especificaciones detalladas por dominio.
- `docs/architecture.md` — Arquitectura actual y planificada.
- `docs/roadmap.md` — Estado de módulos.

---

## 6. Invariantes de datos e identidad

- El dataset bruto no se modifica.
- La identidad se consolida por `track_id` y después por huella exacta.
- Los splits se agrupan por `recording_group_id`.
- No se usan identificadores, nombres, artistas, álbumes ni la etiqueta objetivo como características prohibidas.
- El conjunto de prueba permanece congelado y no participa en selección.
- Los multigéneros no se fuerzan automáticamente a una única verdad.
- Para el dataset cuyo SHA-256 es `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`, después de poner en cuarentena la única fila con identidad inválida, el pipeline debe producir exactamente 83.881 `recording_group_id`. Una diferencia debe hacer fallar la regresión.

---

## 7. Reglas contra fuga de información

- Un `recording_group_id` completo debe pertenecer a un único conjunto: train, validation o test.
- No puede aparecer en dos conjuntos.
- El test se congela y no se usa durante selección de modelos ni umbrales.
- Las transformaciones se ajustan solo con train.

---

## 8. Reglas esenciales de recomendadores y clasificación

- Similitud no significa probabilidad de gusto.
- Puntuaciones no calibradas no se llaman probabilidades.
- El recomendador por canción usa vecinos y distancias.
- El recomendador por preferencias usa distancia ponderada.
- Variables no configuradas tienen peso 0.
- Streamlit no reentrena modelos durante una interacción.
- Artefactos ausentes producen un error claro, no entrenamiento silencioso.

---

## 9. Política de artefactos y entrenamiento

- Configuración aprobada → validación de datos → validación de splits → entrenamiento → validación → guardado de artefactos → reporte → revisión humana.
- Nunca hacer `fit_transform` sobre todo el dataset antes del split.
- Nunca usar test para elegir modelo.
- Nunca mezclar grabaciones entre splits.
- Nunca sobrescribir un artefacto sin nueva versión.
- Nunca comparar modelos con splits distintos.
- Nunca entrenar sin guardar configuración.
- La base de datos es opcional y el tracking debe fallar de forma abierta.

---

## 10. Política Git

1. Trabajar en una rama temática.
2. Usar `git add <archivos-específicos>` en vez de agregar todos los archivos de golpe como instrucción predeterminada.
3. Ejecutar validaciones antes del commit.
4. Crear commits locales coherentes.
5. Mostrar el comando de push como instrucción manual opcional para el propietario.
6. OpenCode no ejecuta `git push`.

---

## 11. Referencias a documentación detallada

OpenCode debe leer solo los documentos relevantes para la tarea, no todos preventivamente.

| Tipo de tarea | Documento que debe leerse |
|---|---|
| Datos, limpieza o identidad | `docs/spec/data-and-identity.md` |
| Recomendadores | `docs/spec/recommenders.md` |
| Clasificación | `docs/spec/classification.md` |
| Streamlit | `docs/spec/application.md` |
| Base de datos | `docs/spec/persistence.md` |
| Estado o siguiente módulo | `docs/roadmap.md` |
| Arquitectura | `docs/architecture.md` |

---

## 12. Metadatos

| Campo | Valor |
|---|---|
| Versión de la guía | 2.0 |
| Creada originalmente | 29 de julio de 2026 |
| Última actualización | 3 de agosto de 2026 |
| Estado de implementación | consultar `docs/roadmap.md` |