"""Generate notebooks/05_preference_recommender_experiments.ipynb.

The notebook demonstrates the preference recommender (AGENTS.md §15): preset
loading, weighted distance ranking, out-of-distribution detection and optional
MMR diversity reranking.

Usage:
    uv run python scripts/build_preference_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_preference_recommender_experiments.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path("notebooks/05_preference_recommender_experiments.ipynb")


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source, execution_count=None)


def build() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    cells.append(
        md(
            """# 05 · Experimentos del recomendador por preferencias

**Proyecto:** Spotify Music Intelligence
**Módulo 5:** Recomendador por preferencias
**Objetivo:** validar presets editables, distancia ponderada (§15.6), detección
de perfiles fuera de distribución (§15.8) y diversidad MMR opcional (§15.9).

Unidad de modelado: `recording_group_id`. Se excluyen las grabaciones con
análisis acústico incompleto.

Las puntuaciones de similitud derivan de una distancia ponderada y **no son
probabilidades**."""
        )
    )

    cells.append(
        md(
            """## 1. Carga de configuración y presets

Los presets viven en `configs/presets.yaml` (AGENTS.md §32.6) y son editables
sin tocar Python. Cada preset define valores y pesos por variable (escala 0–3)."""
        )
    )
    cells.append(
        code(
            """import os
from pathlib import Path

from spotify_intelligence.features.presets import load_presets, preset_names

cwd = Path.cwd()
if cwd.name == "notebooks":
    os.chdir(cwd.parent)

presets = load_presets()
print("Presets:", preset_names())
print("Total presets:", len(presets))
print()
for key, preset in presets.items():
    print(f"{key}: {preset['label']} | pesos: {preset['weights']}")"""
        )
    )

    cells.append(
        md(
            """## 2. Construcción de artefactos

Se construyen los artefactos versionados en `models/preferences/v1/`
(escalador, matriz escalada, catálogo e referencia OOD)."""
        )
    )
    cells.append(
        code(
            """import json

from scripts.build_preference_recommender import build

manifest = build()
print(json.dumps(manifest, indent=2))"""
        )
    )

    cells.append(
        md(
            """## 3. Recomendación por preset

Se recomienda con el preset «Fiesta» y se muestra el Top-5 ordenado por
distancia ponderada ascendente (mayor similitud primero)."""
        )
    )
    cells.append(
        code(
            """from spotify_intelligence.recommenders.preference_based import (
    PreferenceProfile,
    PreferenceRecommender,
)

recommender = PreferenceRecommender("models/preferences/v1")

profile = PreferenceProfile.from_preset("fiesta")
print("Perfil:", profile.label, profile.values)

results = recommender.recommend(profile, top_n=5)
print(results[["track_name", "artists", "distance", "similarity"]].to_string(index=False))"""
        )
    )

    cells.append(
        md(
            """## 4. Perfil manual y rechazo de pesos cero

Se valida que un perfil con todos los pesos en cero es rechazado (§15.5)."""
        )
    )
    cells.append(
        code(
            """from spotify_intelligence.recommenders.errors import InvalidPreferenceProfileError
from spotify_intelligence.recommenders.preference_based import PreferenceProfile

try:
    PreferenceProfile.from_manual(
        values={"energy": 0.8, "danceability": 0.9},
        weights={"energy": 0, "danceability": 0},
    )
except InvalidPreferenceProfileError as exc:
    print("Rechazado como se esperaba:", exc)

manual = PreferenceProfile.from_manual(
    values={
        "energy": 0.8,
        "danceability": 0.9,
        "valence": 0.5,
        "acousticness": 0.2,
        "instrumentalness": 0.1,
        "tempo": 120,
    },
    weights={
        "energy": 2,
        "danceability": 3,
        "valence": 1,
        "acousticness": 0,
        "instrumentalness": 0,
        "tempo": 2,
    },
)
print("Peso 0 ignora la variable; perfil manual válido.")"""
        )
    )

    cells.append(
        md(
            """## 5. Perfiles fuera de distribución (OOD)

Se compara la distancia del perfil al centroide del catálogo con los
percentiles p95/p99 precomputados (§15.8)."""
        )
    )
    cells.append(
        code(
            """for key in ["fiesta", "melancolico", "concentracion_instrumental"]:
    profile = PreferenceProfile.from_preset(key)
    status = recommender.out_of_distribution_status(profile)
    print(
        f"{key}: {status['status']} | distancia centroide={status['distance_to_centroid']:.3f} "
        f"| p95={status['p95']:.3f} | p99={status['p99']:.3f}"
    )"""
        )
    )

    cells.append(
        md(
            """## 6. Diversidad MMR opcional

Se compara el ranking puro contra el reranking MMR (`lambda = 0,85`). El primer
resultado coincide; las posiciones posteriores pueden reordenarse para aumentar
la diversidad interna."""
        )
    )
    cells.append(
        code(
            """profile = PreferenceProfile.from_preset("fiesta")
pure = recommender.recommend(profile, top_n=10)
mmr = recommender.recommend(profile, top_n=10, diversity_enabled=True, lambda_=0.85)

print(
    "Primer resultado idéntico:",
    pure.iloc[0]["recording_group_id"] == mmr.iloc[0]["recording_group_id"],
)
print("Similitud media pura:", round(pure["similarity"].mean(), 4))
print("Similitud media MMR: ", round(mmr["similarity"].mean(), 4))
print("Desviación interna pura:", round(pure["similarity"].std(), 4))
print("Desviación interna MMR: ", round(mmr["similarity"].std(), 4))"""
        )
    )

    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = cells
    return notebook


def main() -> None:
    nb = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
