from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_FONT: str = "DejaVu Sans"


def _pick_available_font(preferred: str) -> str:
    candidates = [preferred, "Arial", "Helvetica"]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        # Some font managers list family names
        if any(name.lower() in a.lower() for a in available):
            return name
    return mpl.rcParams.get("font.family", ["sans-serif"])


def apply_plot_style(font: Optional[str] = None) -> None:
    """Apply a consistent plotting style across all case studies."""

    font_family = _pick_available_font(font or PLOT_FONT)
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
        }
    )

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
