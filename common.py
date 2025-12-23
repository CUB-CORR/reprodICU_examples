from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Preferred font family (fallbacks handled in apply_plot_style)
PLOT_FONT: str = "DejaVu Sans"


def _pick_available_font(preferred: str) -> str:
    """
    Return the first available font family among sensible defaults.

    Tries the provided preferred font first, then common cross-platform
    fonts. Falls back to Matplotlib's default if none are available.
    """

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

    # Base seaborn theme
    sns.set_theme(context="notebook", style="whitegrid")

    # Matplotlib rcParams tuned for scientific plots
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

    # Improve PDF/SVG text rendering when saved
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"  # Keep text as text
