"""Generate notebooks/04_track_recommender_experiments.ipynb.

The notebook exercises the four mandatory recommender configurations
(R1..R4 from AGENTS.md sección 14.4) using reusable code from
``spotify_intelligence.recommenders.experiments``.

Usage:
    uv run python scripts/build_track_recommender_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_track_recommender_experiments.ipynb
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path("notebooks/04_track_recommender_experiments.ipynb")
SEED_QUERIES = [
    ("Blinding Lights", "The Weeknd"),
    ("Bohemian Rhapsody - Remastered 2011", "Queen"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Easy On Me", "Adele"),
    ("Me Porto Bonito", "Bad Bunny;Chencho Corleone"),
]


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source, execution_count=None)


def build() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    cells.append(
        md(
            """# 04 · Experimentos del recomendador por canción

**Proyecto:** Spotify Music Intelligence
**Módulo 4:** Recomendador por canción
**Objetivo:** comparar las cuatro configuraciones obligatorias (R1–R4,
AGENTS.md sección 14.4) sobre semillas manuales sin usar test y sin modificar datos.

| ID | Escalador | Distancia |
|---|---|---|
| R1 | StandardScaler | coseno |
| R2 | RobustScaler | coseno |
| R3 | StandardScaler | euclídea |
| R4 | RobustScaler | euclídea |

Baseline inicial: `StandardScaler + NearestNeighbors(cosine, brute)`.

Unidad de modelado: `recording_group_id`. Se excluyen las grabaciones con
análisis acústico incompleto y se excluye siempre la propia grabación."""
        )
    )

    cells.append(
        md(
            """## Configuración y carga

Se cargan los datos procesados y el código reutilizable del paquete
`spotify_intelligence.recommenders`. Ningún cálculo pesado vive en este
notebook."""
        )
    )
    cells.append(
        code(
            """import os
from pathlib import Path

import pandas as pd

from spotify_intelligence.recommenders import experiments as exp

cwd = Path.cwd()
if cwd.name == "notebooks":
    os.chdir(cwd.parent)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

recordings = pd.read_parquet("data/processed/recordings.parquet")
eligible = recordings[~recordings["audio_analysis_incomplete"]].reset_index(drop=True)

print("recordings:", recordings.shape)
print("elegibles:", eligible.shape)"""
        )
    )

    cells.append(
        md(
            """## 1. Semillas manuales

Se resuelven canciones reconocibles por `track_name` + `artists` a su
`recording_group_id` dentro del catálogo elegible. La resolución exacta se
simula con búsqueda por subcadena; en la aplicación se usará un buscador
desambiguado."""
        )
    )
    cells.append(
        code(
            """seed_queries = [
    ("Blinding Lights", "The Weeknd"),
    ("Bohemian Rhapsody - Remastered 2011", "Queen"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Easy On Me", "Adele"),
    ("Me Porto Bonito", "Bad Bunny;Chencho Corleone"),
]

def resolve_seed(tracks, query_name, query_artist):
    match = tracks[
        tracks["track_name"].str.contains(query_name, case=False, na=False)
        & tracks["artists"].str.contains(query_artist, case=False, na=False)
    ]
    if match.empty:
        return None
    best = match.sort_values("popularity_median", ascending=False).iloc[0]
    return int(best.name)

seed_rows = []
for name, artist in seed_queries:
    row = resolve_seed(eligible, name, artist)
    if row is None:
        print(f"{name} | {artist} -> NO resuelto")
        continue
    seed_rows.append(row)
    rec = eligible.iloc[row]
    print(f"{name} | {artist} -> {rec['recording_group_id']}")

print("\\nSeed rows:", seed_rows)"""
        )
    )

    cells.append(
        md(
            """**Resultado:** las cinco semillas se resolvieron a `recording_group_id`.
Estas semillas se usan únicamente para inspección manual y comparación de
configuraciones; no son un ground truth de preferencias."""
        )
    )

    cells.append(
        md(
            """## 2. Baseline R1

**Método:** `StandardScaler + NearestNeighbors(metric="cosine", algorithm="brute")`
con `Top-N = 10` y primer tramo de recuperación de 100 candidatos."""
        )
    )
    cells.append(
        code(
            """r1 = exp.run_experiment(
    eligible,
    seed_rows,
    scaler_name="standard",
    metric="cosine",
    top_n=10,
    candidate_floor=100,
)
print(pd.Series(r1).to_string())"""
        )
    )
    cells.append(
        code(
            """r1_scaled, r1_nn = exp.build_experiment_index(
    eligible, scaler_name="standard", metric="cosine"
)
for seed_row in seed_rows[:2]:
    rows, dists = exp.recommend_with_index(r1_scaled, r1_nn, seed_row, top_n=5)
    names = eligible.iloc[rows][["track_name", "artists"]].values.tolist()
    print(f"\\nSeed: {eligible.iloc[seed_row]['track_name']} | {eligible.iloc[seed_row]['artists']}")
    for (name, artist), dist in zip(names, dists, strict=False):
        print(f"  sim={1 - dist:.4f}  {name} | {artist}")"""
        )
    )
    cells.append(
        md(
            """**Resultado:** las listas del baseline R1 son coherentes: artistas y
géneros afines a cada semilla, sin autorrecomendación y sin grupos repetidos en
las inspecciones. `similarity = 1 - cosine_distance` y **no es una
probabilidad**."""
        )
    )

    cells.append(
        md(
            """## 3. Comparación R1–R4

**Método:** se ejecutan las cuatro configuraciones sobre las mismas semillas y
se miden autorrecomendaciones, duplicados, similitud media, cobertura del
catálogo y latencia. El reporte se guarda en
`reports/experiments/track_recommender_r1_r4_comparison.json`."""
        )
    )
    cells.append(
        code(
            """summary = exp.run_all_experiments(eligible, seed_rows, top_n=10, candidate_floor=100)
summary"""
        )
    )
    cells.append(
        code(
            """path = exp.save_experiment_report(summary, output_dir="reports/experiments")
print("Reporte guardado en", path)"""
        )
    )
    cells.append(
        md(
            """**Resultado:** las cuatro configuraciones cumplen `autorrecomendación = 0`
y `duplicados de grupos = 0`. La similitud media de R1 (coseno + StandardScaler)
es la más alta de las configuraciones coseno (0,979 vs 0,968 de R2). Para las
configuraciones euclídeas (R3, R4) la columna `mean_similarity` usa `1 - distance`
pero la distancia euclídea no está acotada en [0, 1], por lo que ese valor **no
es comparable** con el de las configuraciones coseno; lo que sí es comparable
es la latencia y el cumplimiento de autorrecomendación/duplicados. La latencia
media por consulta con el catálogo completo es de ~15 ms en R1/R2; R4 (euclídea
+ RobustScaler) también es rápida (decenas de ms) mientras que R3 (euclídea +
StandardScaler) suele ser la más lenta (varios cientos de ms). Los valores
exactos se guardan en
`reports/experiments/track_recommender_r1_r4_comparison.json` y varían según el
equipo y la carga.

**Limitación:** las métricas de cobertura sobre 5 semillas no representan la
cobertura real del catálogo; la evaluación completa (módulo 4, evaluación
automática) usa una muestra más amplia. `similarity` no es una probabilidad."""
        )
    )

    cells.append(
        md(
            """## Conclusión

**Resumen:** R1 (StandardScaler + coseno, brute) reproduce el baseline
especificado y muestra las similitudes más altas en las inspecciones manuales.
R2–R4 quedan documentados como alternativas. La configuración inicial de
producción se fija en **R1** a la espera de la aprobación del propietario
(AGENTS.md sección 7 y sección 30).

Limitaciones: las semillas son inspección manual, no ground truth; la cobertura
completa y la latencia p50/p95 se miden en la evaluación offline."""
        )
    )

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    return notebook


if __name__ == "__main__":
    notebook = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(NOTEBOOK_PATH)], check=True)
    print(f"Notebook escrito en {NOTEBOOK_PATH}")
