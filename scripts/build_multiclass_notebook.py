"""Generate notebooks/07_multiclass_classifier.ipynb.

The notebook exercises the single-label multiclass dataset (AGENTS.md §17),
the frozen grouped splits, the C0 (frequent class) baseline and the approved
final model C1 (logistic) loaded from its versioned artifact. C2/C3 (tree
ensembles) are trained and compared via ``scripts/compare_multiclass_models.py``
because of their size; the notebook references that report instead of
retraining them.

Usage:
    uv run python scripts/build_multiclass_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_multiclass_classifier.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path("notebooks/07_multiclass_classifier.ipynb")


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source, execution_count=None)


def build() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    cells.append(
        md(
            """# 07 · Clasificador multiclase: género acústico dominante

**Proyecto:** Spotify Music Intelligence
**Módulo 7:** Clasificador multiclase secundario (AGENTS.md §17)
**Objetivo:** estimar un género acústico dominante entre 114 clases usando
únicamente grabaciones con una sola etiqueta, sin usar el test durante la
selección de modelo.

La predicción no reemplaza las etiquetas originales de canciones multigénero
(§17.5). El split agrupado congelado (70/15/15) está en
`data/processed/splits.parquet`. Los árboles C2/C3 se compararon con
`scripts/compare_multiclass_models.py`; el modelo final aprobado es C1."""
        )
    )

    cells.append(
        md(
            """## Configuración y datos

Se cargan los datos procesados y el dataset multiclase construido por
`spotify_intelligence.classification.training.prepare_multiclass_data`.
No se modifica ningún dato."""
        )
    )
    cells.append(
        code(
            """import os
from pathlib import Path

import pandas as pd

from spotify_intelligence.classification.multiclass import (
    build_model,
    expand_to_full_label_space,
    load_model_parameters,
    model_classes,
    predict_proba_scores,
)
from spotify_intelligence.classification.multiclass_evaluation import (
    dominant_genre_exploratory,
    evaluate_multiclass,
)
from spotify_intelligence.classification.training import (
    feature_matrix,
    prepare_base_dataset,
    prepare_multiclass_data,
    split_map_from_dir,
    subset_dataset,
)
from spotify_intelligence.data.splits import verify_disjoint_splits

cwd = Path.cwd()
if cwd.name == "notebooks":
    os.chdir(cwd.parent)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

dataset = prepare_base_dataset("data/processed")
split_map = split_map_from_dir("data/processed")
print("n_samples:", dataset.n_samples)
print("n_labels:", dataset.n_labels)
print(
    "train:",
    len(split_map["train"]),
    "validation:",
    len(split_map["validation"]),
    "test:",
    len(split_map["test"]),
)"""
        )
    )

    cells.append(
        md(
            """## Dataset de grabaciones monoetiqueta y split sin fuga

Solo se usan grabaciones con exactamente un género (§17.1). Se verifica que
ningún `recording_group_id` aparece en dos conjuntos (§3.5)."""
        )
    )
    cells.append(
        code(
            """verify_disjoint_splits(split_map)
print("OK: intersecciones vacías entre train/validation/test")

data = prepare_multiclass_data(experiment="A")
print("X_train:", data.X_train.shape)
print("X_val:", data.X_val.shape)
print("X_test:", "None (solo con --use-test)", None if data.X_test is None else data.X_test.shape)

y_train = data.y_train
class_names = data.dataset.genre_encoder.classes_
counts = pd.Series(y_train).value_counts()
print("clases en train:", int(counts.shape[0]), "de", len(class_names))
print("min/max por clase:", int(counts.min()), "/", int(counts.max()))
print("media por clase:", round(float(counts.mean()), 1))"""
        )
    )

    cells.append(
        md(
            """## Baseline C0 · Clase frecuente

Predice siempre la clase más frecuente de train; no usa características."""
        )
    )
    cells.append(
        code(
            """model_params = load_model_parameters("configs/model_parameters.yaml")
m0 = build_model("C0", model_params)
m0.fit(data.X_train, data.y_train)
dense0 = predict_proba_scores(m0, data.X_val)
full0 = expand_to_full_label_space(dense0, model_classes(m0), len(class_names))
evaluate_multiclass(data.y_val, full0, class_names)"""
        )
    )

    cells.append(
        md(
            """## Modelo final C1 · Regresión logística

C1 se entrenó una vez con `scripts/train_multiclass_classifier.py`
(`solver=lbfgs`, `C=1.0`, `max_iter=3000`, `class_weight=balanced`). Aquí se
carga el artefacto versionado y se evalúa sobre validación. C2 (Extra Trees) y
C3 (Random Forest) alcanzaron mejor accuracy pero pesan ~650 MB cada uno;
C1 (~0,5 MB) fue aprobado como modelo final por el propietario."""
        )
    )
    cells.append(
        code(
            """import json

import joblib

c1_dirs = sorted(Path("models/classifier/multiclass").glob("*multiclass_C1"))
if not c1_dirs:
    raise SystemExit(
        "No se encontró el artefacto C1. Ejecute scripts/train_multiclass_classifier.py --model C1"
    )
artifact_dir = c1_dirs[-1]
manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
model = joblib.load(artifact_dir / "model.joblib")
scaler = joblib.load(artifact_dir / "scaler.joblib")
print("artefacto:", artifact_dir.name)
print("split_sha256:", manifest["split_sha256"][:12] + "...")

dense = predict_proba_scores(model, data.X_val)
full = expand_to_full_label_space(dense, model_classes(model), len(class_names))
metrics = evaluate_multiclass(data.y_val, full, class_names)
{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}"""
        )
    )

    cells.append(
        md(
            """## Evaluación exploratoria sobre grabaciones multigénero (§17.5)

C1 estima un único género dominante. Se evalúa con `Hit@1`, `Hit@3` y
`Recall@5` sobre filas de validación que tienen más de una etiqueta original."""
        )
    )
    cells.append(
        code(
            """val = subset_dataset(dataset, split_map["validation"], experiment="A")
mask = val.Y.sum(axis=1) > 1
print("filas multigénero en validación:", int(mask.sum()))

X_mg = scaler.transform(feature_matrix(val, "A")[mask])
Y_mg = val.Y[mask]
dense_mg = predict_proba_scores(model, X_mg)
full_mg = expand_to_full_label_space(dense_mg, model_classes(model), len(class_names))
dominant_genre_exploratory(Y_mg, full_mg, class_names)"""
        )
    )

    cells.append(
        md(
            """## Limitaciones

- Es un laboratorio experimental (§17); la predicción **no** reemplaza las
  etiquetas originales de canciones multigénero.
- El test congelado se usa solo en la evaluación final autorizada
  (`scripts/evaluate_final_multiclass_model.py --use-test`).
- Las puntuaciones no calibradas **no** se llaman probabilidades (§16.10).
- La muestra es balanceada por bloque (1.000 filas por género); no permite
  inferir prevalencia real de Spotify (§2.3)."""
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
