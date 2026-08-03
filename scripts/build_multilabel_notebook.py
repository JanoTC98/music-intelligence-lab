"""Generate notebooks/06_multilabel_classifier.ipynb.

The notebook exercises the multilabel dataset, frozen grouped splits, the M0
(frequency) and M1 (OneVsRest logistic) baselines, global threshold tuning on
validation and the §16.9 metric set, reusing code from
``spotify_intelligence.classification``. Heavy models (M3/M4) are trained via
``scripts/compare_models.py``, not inside the notebook.

Usage:
    uv run python scripts/build_multilabel_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_multilabel_classifier.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path("notebooks/06_multilabel_classifier.ipynb")


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source, execution_count=None)


def build() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    cells.append(
        md(
            """# 06 · Laboratorio supervisado: clasificación multietiqueta de géneros

**Proyecto:** Spotify Music Intelligence
**Módulo 6:** Clasificador multietiqueta (AGENTS.md §16)
**Objetivo:** predecir un conjunto compatible de los 114 géneros a partir de
características acústicas, sin usar el conjunto de test durante la selección.

Unidad de modelado: `recording_group_id`. El split agrupado (70/15/15) está
congelado en `data/processed/splits.parquet`. Este notebook entrena solo los
baselines M0 (frecuencia) y M1 (OneVsRest logistic); los modelos de árboles
M3/M4 se entrenan con `scripts/compare_models.py`."""
        )
    )

    cells.append(
        md(
            """## Configuración y datos

Se cargan los datos procesados y el dataset multilabel construido por
`spotify_intelligence.classification.datasets`. No se modifica ningún dato."""
        )
    )
    cells.append(
        code(
            """import os
from pathlib import Path

import numpy as np
import pandas as pd

from spotify_intelligence.classification import datasets as ds
from spotify_intelligence.classification import evaluation as ev
from spotify_intelligence.classification import predict as pr
from spotify_intelligence.classification import thresholds as th
from spotify_intelligence.classification.multilabel import (
    build_model,
    load_model_parameters,
    predict_proba_scores,
)
from spotify_intelligence.classification.training import (
    prepare_training_data,
    prepare_base_dataset,
    split_map_from_dir,
)
from spotify_intelligence.data.splits import load_splits

cwd = Path.cwd()
if cwd.name == "notebooks":
    os.chdir(cwd.parent)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

dataset = prepare_base_dataset("data/processed")
split_map = split_map_from_dir("data/processed")
print("n_samples:", dataset.n_samples)
print("n_labels:", dataset.n_labels)
print("train:", len(split_map["train"]),
      "validation:", len(split_map["validation"]),
      "test:", len(split_map["test"]))
print("incomplete:", int(dataset.incomplete_mask.sum()))"""
        )
    )

    cells.append(
        md(
            """## Split agrupado y verificación de fuga

Se comprueba que ningún `recording_group_id` aparece en dos conjuntos (§3.5)."""
        )
    )
    cells.append(
        code(
            """from spotify_intelligence.data.splits import verify_disjoint_splits

verify_disjoint_splits(split_map)
print("OK: intersecciones vacías entre train/validation/test")"""
        )
    )

    cells.append(
        md(
            """## Baseline M0 · Frecuencia de etiquetas

No usa características; predice la prevalencia observada de cada etiqueta en
train. El umbral global se optimiza en validación (§16.8)."""
        )
    )
    cells.append(
        code(
            """model_params = load_model_parameters("configs/model_parameters.yaml")
data = prepare_training_data(experiment="A")

m0 = build_model("M0", model_params)
m0.fit(data.X_train, data.Y_train)
scores_m0 = predict_proba_scores(m0, data.X_val)
t0 = th.tune_global_threshold(scores_m0, data.Y_val)
print("M0 best threshold:", t0.best_threshold, "score:", round(t0.best_score, 4))"""
        )
    )
    cells.append(
        code(
            """pred_m0 = pr.predict_with_threshold(
    scores_m0, data.dataset.genre_encoder, t0.best_threshold
)
ev.evaluate_multilabel(data.Y_val, scores_m0, pred_m0["labels"])"""
        )
    )

    cells.append(
        md(
            """## Baseline M1 · One-vs-Rest Logistic Regression

Configuración inicial §16.6: `liblinear`, `C=1.0`, `max_iter=2000`,
`class_weight=balanced`, wrapper `n_jobs=-1`."""
        )
    )
    cells.append(
        code(
            """m1 = build_model("M1", model_params)
m1.fit(data.X_train, data.Y_train)
scores_m1 = predict_proba_scores(m1, data.X_val)
t1 = th.tune_global_threshold(scores_m1, data.Y_val)
print("M1 best threshold:", t1.best_threshold, "score:", round(t1.best_score, 4))"""
        )
    )
    cells.append(
        code(
            """pred_m1 = pr.predict_with_threshold(
    scores_m1, data.dataset.genre_encoder, t1.best_threshold
)
metrics_m1 = ev.evaluate_multilabel(data.Y_val, scores_m1, pred_m1["labels"])
{k: round(v, 4) for k, v in metrics_m1.items() if k != "per_label_average_precision"}"""
        )
    )

    cells.append(
        md(
            """## Ejemplo de predicción Top-5

La aplicación muestra siempre Top-5. Si ninguna etiqueta supera el umbral, se
muestra el top-1 con un aviso (§16.8)."""
        )
    )
    cells.append(
        code(
            """sample_scores = scores_m1[:3]
topk = pr.top_k_genres(sample_scores, data.dataset.genre_encoder, k=5)
for row in topk:
    print(row)"""
        )
    )

    cells.append(
        md(
            """## Limitaciones

- Las puntuaciones no calibradas **no** se llaman probabilidades (§16.10).
- El test congelado se usa solo en la evaluación final autorizada
  (`scripts/evaluate_final_model.py --use-test`).
- La prevalencia real de géneros de Spotify no puede inferirse de esta muestra
  balanceada (§2.3)."""
        )
    )

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = cells
    return notebook


if __name__ == "__main__":
    nbformat.write(build(), NOTEBOOK_PATH)
    print("Notebook escrito en", NOTEBOOK_PATH)
