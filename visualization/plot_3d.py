"""
3D Surface Plotting Module — publication-quality 3D visualisations.

Provides:
  - create_meshgrid: utility to build a 2D coordinate grid
  - predict_surface_2d: build a prediction surface grid over two named features
  - Plot3D: class for enhanced 3D surface, scatter, and contour plots
  - plot_all_pairwise_surfaces: batch-generate 3D surfaces for feature pairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.feature_names import (
    FEATURE_ABBREV, FEATURE_LATEX,
    get_display_name, get_filename_safe,
)


# ============================================================
# Utility: create a 2D meshgrid
# ============================================================
def create_meshgrid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D coordinate grid for surface plotting.

    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_points: Number of grid points per axis (default 50).

    Returns:
        (X, Y): Meshgrid arrays, each shaped (n_points, n_points).
        Note: the y-axis is reversed (y_max → y_min) so that the surface
        orientation matches the natural view angle.
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[1], y_range[0], n_points)  # y-axis reversed
    X, Y = np.meshgrid(x, y)
    return X, Y


# ============================================================
# Utility: build a 2D prediction surface grid
# ============================================================
def predict_surface_2d(
    model,
    feature_names: List[str],
    feat1: str,
    feat2: str,
    X_train: np.ndarray,
    other_features: Optional[Dict[str, float]] = None,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a prediction surface grid over two named features.

    All other features are held at their training-set mean values unless
    explicit values are supplied via ``other_features``.

    Args:
        model: A fitted model with a .predict() method.
        feature_names: List of feature names matching columns of X_train.
        feat1: Name of the first feature (x-axis).
        feat2: Name of the second feature (y-axis).
        X_train: Training feature matrix (n_samples, n_features).
        other_features: Optional dict of {feature_name: fixed_value} for
                        features that should be held at a custom value.
        n_points: Number of grid points along each axis (default 50).

    Returns:
        Tuple of (X_grid, Y_grid, Z_grid), each shaped (n_points, n_points).
    """
    idx1 = feature_names.index(feat1)
    idx2 = feature_names.index(feat2)

    feat1_range = (X_train[:, idx1].min(), X_train[:, idx1].max())
    feat2_range = (X_train[:, idx2].min(), X_train[:, idx2].max())

    X, Y = create_meshgrid(feat1_range, feat2_range, n_points)

    n_samples = X.size
    X_pred = np.tile(X_train.mean(axis=0), (n_samples, 1))

    if other_features:
        for feat_name, feat_value in other_features.items():
            if feat_name in feature_names:
                feat_idx = feature_names.index(feat_name)
                X_pred[:, feat_idx] = feat_value

    X_pred[:, idx1] = X.ravel()
    X_pred[:, idx2] = Y.ravel()

    Z = model.predict(X_pred).reshape(X.shape)
    return X, Y, Z


# ============================================================
# Plot3D Class — enhanced 3D surface plotting
# ============================================================
class Plot3D:
    """
    Publication-quality 3D plotter with smart view angle,
    contour projection, and enhanced colour contrast.

    Methods:
        plot_surface    — 3D surface with contour projection
        plot_scatter_3d — 3D scatter plot
        plot_contour_3d — 3D contour plot with optional surface
    """

    def __init__(self, style: str = 'default'):
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Apply rcParams for publication-quality 3D figures."""
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18

    def plot_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        title: str,
        save_path: Path,
        cmap: str = 'RdYlBu_r',
        alpha: float = 0.75,
        show_points: bool = False,
        points_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        colorbar_label: str = 'Value',
    ):
        """
        Render an enhanced 3D surface plot with contour projection,
        smart view angle, and enhanced colour contrast.

        Args:
            X, Y: Meshgrid coordinates.
            Z: Predicted values on the grid.
            xlabel, ylabel, zlabel: Axis labels.
            title: Plot title.
            save_path: Path to save the figure.
            cmap: Matplotlib colourmap (default 'RdYlBu_r').
            alpha: Surface transparency (default 0.75).
            show_points: If True, overlay scatter points.
            points_data: (x, y, z) 1-D arrays for scatter overlay.
            colorbar_label: Colour bar label.
        """
        fig = plt.figure(figsize=(14, 11))
        ax = fig.add_subplot(111, projection='3d')

        z_min, z_max = Z.min(), Z.max()
        z_range = z_max - z_min

        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm,
                               alpha=0.85, linewidth=0.2, antialiased=True,
                               edgecolor='none', shade=True)

        # Contour lines on the surface
        contour_levels = np.linspace(z_min, z_max, 15)
        ax.contour(X, Y, Z, zdir='z', offset=z_min - z_range * 0.1,
                   levels=contour_levels, cmap=cmap, alpha=0.6, linewidths=1.5)
        ax.contour(X, Y, Z, zdir='z', offset=z_min - z_range * 0.15,
                   levels=contour_levels, cmap=cmap, alpha=0.4, linewidths=1)

        # Axis labels
        ax.set_xlabel(xlabel, fontsize=16, labelpad=15, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, labelpad=15, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=16, labelpad=15, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)

        # Colour bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.15)
        cbar.set_label(colorbar_label, rotation=270, labelpad=30,
                       fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

        # Statistics annotation
        stats_text = (f'Range: [{z_min:.1f}, {z_max:.1f}]\n'
                      f'Mean: {Z.mean():.1f}\nStd: {Z.std():.1f}')
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                  family='monospace')

        # Smart view angle: put the high-value region in the foreground
        x_center = X.mean()
        y_center = Y.mean()
        near_mask = (X >= x_center) & (Y <= y_center)
        far_mask = (X <= x_center) & (Y >= y_center)
        near_avg = Z[near_mask].mean() if near_mask.any() else Z.mean()
        far_avg = Z[far_mask].mean() if far_mask.any() else Z.mean()
        azim = 45 + 180 if near_avg > far_avg else 45
        ax.view_init(elev=30, azim=azim)

        ax.set_zlim(z_min - z_range * 0.2, z_max + z_range * 0.1)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {Path(save_path).name}")

    def plot_scatter_3d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        c: Optional[np.ndarray] = None,
        xlabel: str = '',
        ylabel: str = '',
        zlabel: str = '',
        title: str = '',
        save_path: Path = None,
        cmap: str = 'viridis',
        s: int = 20,
        alpha: float = 0.6,
    ):
        """
        Render a 3D scatter plot.

        Args:
            x, y, z: 1-D coordinate arrays.
            c: Optional colour values array.
            xlabel, ylabel, zlabel: Axis labels.
            title: Plot title.
            save_path: Path to save the figure.
            cmap: Colourmap (default 'viridis').
            s: Marker size.
            alpha: Marker transparency.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if c is not None:
            scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, alpha=alpha,
                                 edgecolors='black', linewidths=0.3)
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label('Value', rotation=270, labelpad=20, fontsize=10)
        else:
            ax.scatter(x, y, z, s=s, alpha=alpha,
                       edgecolors='black', linewidths=0.3)

        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
        ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved: {Path(save_path).name}")
        plt.close()

    def plot_contour_3d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        title: str,
        save_path: Path,
        levels: int = 20,
        cmap: str = 'RdYlBu_r',
        show_surface: bool = True,
    ):
        """
        Render a 3D contour plot (optionally overlaid with a surface).

        Args:
            X, Y: Meshgrid coordinates.
            Z: Height values.
            xlabel, ylabel, zlabel: Axis labels.
            title: Plot title.
            save_path: Path to save the figure.
            levels: Number of contour levels (default 20).
            cmap: Colourmap (default 'RdYlBu_r').
            show_surface: If True, render the surface behind the contours.
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if show_surface:
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.7,
                                   linewidth=0, antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)

        contour = ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=7)

        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
        ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {Path(save_path).name}")


# ============================================================
# Batch generation: all pairwise 3D surfaces
# ============================================================
def plot_all_pairwise_surfaces(
    model,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_pairs: List[Tuple[str, str]],
    save_dir: Path,
    target_name: str = 'Efficiency',
    prefix: str = 'fig_',
):
    """
    Batch-generate 3D prediction surface plots for all given feature pairs.

    Args:
        model: A fitted model with a .predict() method.
        feature_names: List of feature names matching columns of X_train.
        X_train: Training feature matrix (n_samples, n_features).
        y_train: Training target vector (n_samples,).
        feature_pairs: List of (feature_name_1, feature_name_2) tuples.
        save_dir: Directory to save individual surface plots.
        target_name: Target variable name (z-label and title).
        prefix: Filename prefix (e.g., 'fig_efficiency' or 'fig_reuse').
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plotter = Plot3D()
    colorbar_label = (f'{target_name} (%)'
                      if 'Efficiency' in target_name or 'efficiency' in target_name
                      else target_name)

    for i, (feat1, feat2) in enumerate(feature_pairs, 1):
        try:
            X, Y, Z = predict_surface_2d(model, feature_names, feat1, feat2, X_train)

            idx1 = feature_names.index(feat1)
            idx2 = feature_names.index(feat2)
            display_name1 = get_display_name(feat1)
            display_name2 = get_display_name(feat2)
            filename = f"{prefix}{get_filename_safe(feat1)}_vs_{get_filename_safe(feat2)}_3D.png"

            plotter.plot_surface(
                X, Y, Z,
                xlabel=display_name1,
                ylabel=display_name2,
                zlabel=target_name,
                title=f'{target_name} Surface: {display_name1} × {display_name2}',
                save_path=save_dir / filename,
                show_points=False,
                points_data=None,
                colorbar_label=colorbar_label,
            )
        except Exception as e:
            print(f"  [WARN] Failed to plot {feat1} vs {feat2}: {e}")
            continue
