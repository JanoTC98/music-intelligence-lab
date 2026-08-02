from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig, name: str, output_dir: str | Path = "reports/figures") -> Path:
    """Save a matplotlib figure to ``reports/figures`` and close it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path
