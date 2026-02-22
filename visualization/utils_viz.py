"""
Visualization Utilities — shared helpers to reduce code duplication.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Optional
from pathlib import Path


def safe_plot(plot_func: Callable, *args, **kwargs):
    """Execute a plotting function safely, catching exceptions."""
    try:
        plot_func(*args, **kwargs)
        return True
    except Exception as e:
        print(f"  [WARN] {plot_func.__name__} failed: {e}")
        return False


def get_feature_pairs(features: List[str]) -> List[tuple]:
    """Generate all pairwise combinations of features."""
    return [(feat1, feat2) for i, feat1 in enumerate(features)
            for feat2 in features[i+1:]]


def filter_features(feature_list: List[str], available_features: List[str]) -> List[str]:
    """Keep only features that exist in the available set."""
    return [f for f in feature_list if f in available_features]


def get_top_features_by_importance(shap_values: np.ndarray,
                                   feature_names: List[str],
                                   top_k: int) -> List[str]:
    """Return the top-K features ranked by mean |SHAP|."""
    importances = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importances)[::-1][:top_k]
    return [feature_names[i] for i in top_indices]


def save_figure(fig, save_path: Path, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save a figure and close it."""
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"  [OK] Saved: {save_path.name}")


def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: Tuple[int, int] = (10, 6)):
    """Create a figure and return (fig, axes) with axes always as a flat list."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    return fig, axes


def apply_common_style(ax, xlabel: str = '', ylabel: str = '', title: str = '',
                       grid: bool = True, legend: bool = False):
    """Apply common styling to an axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc='best', fontsize=9)
