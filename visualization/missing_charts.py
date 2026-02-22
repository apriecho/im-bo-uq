"""
Missing Charts Module — supplementary figures not covered in the main pipeline.

Includes:
  - Figure 5: GPR high-performance region analysis (3-panel)
  - Figure S: Top-6 SHAP feature scatter matrix
  - Figure S2a: Oxidiser type distribution bar chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.feature_names import get_display_name, get_filename_safe, FEATURE_LATEX

# ============================================================
# Matplotlib rcParams — Publication-quality style
# Arial font, 22 pt base, 300 DPI, full frame (all spines)
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True


# ============================================================
# Figure 5: GPR High-Performance Region Analysis
# ============================================================
def plot_figure5_gpr_high_performance_region(
    gpr_model,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feat1: str,
    feat2: str,
    save_path: Path,
    threshold_percentile: int = 85,
):
    """
    Generate a 3-panel figure for GPR-based high-performance region analysis.

    Panels:
      (a) 3D prediction surface over the two selected features
      (b) 2D uncertainty (std) heatmap
      (c) Sample scatter coloured by target, with high-performance region overlay

    Args:
        gpr_model: A fitted sklearn GaussianProcessRegressor.
        feature_names: List of feature names matching columns of X_train.
        X_train: Training feature matrix (n_samples, n_features).
        y_train: Training target vector (n_samples,).
        feat1: Name of the first feature (x-axis).
        feat2: Name of the second feature (y-axis).
        save_path: Output file path for the saved figure.
        threshold_percentile: Percentile of predicted values used to
            define the high-performance region (default 85).
    """
    from sklearn.preprocessing import StandardScaler
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Locate feature indices ---
    feat1_idx = feature_names.index(feat1)
    feat2_idx = feature_names.index(feat2)

    # --- Standardise for GPR prediction ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # --- Build a dense grid over the two selected features ---
    n_grid = 60
    f1_min, f1_max = X_train[:, feat1_idx].min(), X_train[:, feat1_idx].max()
    f2_min, f2_max = X_train[:, feat2_idx].min(), X_train[:, feat2_idx].max()
    f1_range = np.linspace(f1_min, f1_max, n_grid)
    f2_range = np.linspace(f2_min, f2_max, n_grid)
    F1, F2 = np.meshgrid(f1_range, f2_range)

    # Construct prediction matrix — hold all other features at their mean
    X_mean = X_train.mean(axis=0)
    X_grid = np.tile(X_mean, (n_grid * n_grid, 1))
    X_grid[:, feat1_idx] = F1.ravel()
    X_grid[:, feat2_idx] = F2.ravel()

    X_grid_scaled = scaler.transform(X_grid)
    y_pred, y_std = gpr_model.predict(X_grid_scaled, return_std=True)

    Z_pred = y_pred.reshape(n_grid, n_grid)
    Z_std = y_std.reshape(n_grid, n_grid)

    # --- High-performance threshold ---
    threshold = np.percentile(y_pred, threshold_percentile)
    high_perf_mask = Z_pred >= threshold

    # --- Label strings ---
    label1 = get_display_name(feat1)
    label2 = get_display_name(feat2)

    # ==========================================================
    # Create 3-panel figure
    # ==========================================================
    fig = plt.figure(figsize=(24, 9))

    # ----------------------------------------------------------
    # Panel (a): 3D prediction surface
    # ----------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(
        F1, F2, Z_pred,
        cmap='RdYlBu_r', alpha=0.80, edgecolor='none',
        antialiased=True,
    )
    ax1.set_xlabel(label1, labelpad=14)
    ax1.set_ylabel(label2, labelpad=14)
    ax1.set_zlabel('Predicted Value', labelpad=12)
    ax1.set_title('(a) GPR Prediction Surface', fontsize=22, pad=18)
    ax1.view_init(elev=30, azim=225)
    fig.colorbar(surf, ax=ax1, shrink=0.55, pad=0.10, label='Predicted Value')

    # ----------------------------------------------------------
    # Panel (b): Uncertainty heatmap
    # ----------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.pcolormesh(
        F1, F2, Z_std,
        cmap='YlOrRd', shading='auto',
    )
    ax2.set_xlabel(label1)
    ax2.set_ylabel(label2)
    ax2.set_title('(b) Prediction Uncertainty (Std)', fontsize=22)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.03)
    cbar.set_label('Standard Deviation', fontsize=18)

    # Overlay contour lines for reference
    contour_levels = np.linspace(Z_std.min(), Z_std.max(), 6)
    ax2.contour(
        F1, F2, Z_std,
        levels=contour_levels, colors='black',
        linewidths=0.8, alpha=0.5,
    )

    # ----------------------------------------------------------
    # Panel (c): Sample scatter with high-performance region
    # ----------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)

    # Scatter of training samples coloured by target value
    sc = ax3.scatter(
        X_train[:, feat1_idx],
        X_train[:, feat2_idx],
        c=y_train, cmap='RdYlBu_r',
        s=35, alpha=0.75, edgecolors='grey', linewidths=0.4,
        zorder=3,
    )
    cbar3 = fig.colorbar(sc, ax=ax3, shrink=0.85, pad=0.03)
    cbar3.set_label('Target Value', fontsize=18)

    # Overlay the high-performance region as a shaded contour
    ax3.contourf(
        F1, F2, high_perf_mask.astype(float),
        levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.20,
    )
    ax3.contour(
        F1, F2, high_perf_mask.astype(float),
        levels=[0.5], colors=['#2ca02c'], linewidths=2.5,
    )

    ax3.set_xlabel(label1)
    ax3.set_ylabel(label2)
    ax3.set_title(
        f'(c) Samples & High-Perf Region (top {100 - threshold_percentile}%)',
        fontsize=22,
    )

    # Legend patch for high-performance region
    from matplotlib.patches import Patch
    legend_patch = Patch(
        facecolor='#2ca02c', alpha=0.30,
        label=f'High-Performance (≥P{threshold_percentile})',
    )
    ax3.legend(handles=[legend_patch], loc='upper right', fontsize=16)

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    plt.tight_layout(w_pad=4.0)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Figure 5 saved: {save_path.name}")


# ============================================================
# Figure S: Top-6 SHAP Feature Scatter Matrix
# ============================================================
def plot_figureS_top6_scatter_matrix(
    df: pd.DataFrame,
    top6_features: List[str],
    target_col: str,
    save_path: Path,
):
    """
    Pairwise scatter matrix for the top 6 SHAP features, coloured by target.

    Uses a seaborn-style pair plot rendered manually with matplotlib so that
    LaTeX feature labels from FEATURE_LATEX are applied consistently.

    Args:
        df: DataFrame containing feature columns and the target column.
        top6_features: List of 6 feature names (must exist in df).
        target_col: Name of the target column used for colour mapping.
        save_path: Output file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(top6_features)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    # Colour normalisation based on target
    target_vals = df[target_col].values
    vmin, vmax = np.nanpercentile(target_vals, [2, 98])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlBu_r

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            fi = top6_features[i]
            fj = top6_features[j]

            if i == j:
                # Diagonal: histogram coloured by target median in each bin
                vals = df[fi].dropna().values
                ax.hist(vals, bins=25, color='#4C72B0', alpha=0.75,
                        edgecolor='white', linewidth=0.5)
                ax.set_ylabel('Count' if j == 0 else '')
            else:
                # Off-diagonal: scatter coloured by target
                x_vals = df[fj].values
                y_vals = df[fi].values
                colours = cmap(norm(target_vals))
                ax.scatter(
                    x_vals, y_vals,
                    c=target_vals, cmap='RdYlBu_r',
                    norm=norm, s=12, alpha=0.55,
                    edgecolors='none', rasterized=True,
                )

            # Axis labels only on edges
            if i == n - 1:
                ax.set_xlabel(get_display_name(fj), fontsize=16)
            else:
                ax.set_xticklabels([])

            if j == 0:
                if i != j:
                    ax.set_ylabel(get_display_name(fi), fontsize=16)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=12)

    # Global colour bar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(get_display_name(target_col), fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle(
        'Top-6 SHAP Feature Pairwise Scatter Matrix',
        fontsize=26, fontweight='bold', y=0.98,
    )
    plt.subplots_adjust(hspace=0.08, wspace=0.08, right=0.91)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Top-6 scatter matrix saved: {save_path.name}")


# ============================================================
# Figure S2a: Oxidiser Type Distribution Bar Chart
# ============================================================
def plot_figureS2a_oxidiser_type_distribution(
    df: pd.DataFrame,
    save_path: Path,
):
    """
    Bar chart showing the distribution of oxidant types in the dataset.

    Reads the 'Oxidiser type' column from df, computes value counts,
    and renders a horizontal bar chart with count and percentage annotations.

    Args:
        df: DataFrame that must contain an 'Oxidiser type' column.
        save_path: Output file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    col = 'Oxidiser type'
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    counts = df[col].value_counts().sort_values(ascending=True)
    total = counts.sum()
    categories = counts.index.tolist()
    values = counts.values

    # Colour palette — distinct hues for each oxidant type
    n_cats = len(categories)
    palette = plt.cm.Set2(np.linspace(0, 1, max(n_cats, 3)))

    fig, ax = plt.subplots(figsize=(10, max(6, n_cats * 0.9)))

    bars = ax.barh(
        range(n_cats), values,
        color=palette[:n_cats], edgecolor='black', linewidth=1.2,
        height=0.65,
    )

    # Annotate each bar with count and percentage
    for idx, (bar, val) in enumerate(zip(bars, values)):
        pct = val / total * 100
        # Place text outside bar if bar is short, inside otherwise
        x_pos = bar.get_width() + total * 0.01
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f'{val}  ({pct:.1f}%)',
            va='center', ha='left', fontsize=18, fontweight='bold',
        )

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(categories, fontsize=20)
    ax.set_xlabel('Number of Samples', fontsize=22)
    ax.set_title('Oxidant Type Distribution', fontsize=24, fontweight='bold')

    # Extend x-axis to leave room for annotations
    ax.set_xlim(0, values.max() * 1.28)
    ax.invert_yaxis()  # Largest bar on top

    # Grid lines on x-axis only, behind bars
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Oxidiser type distribution saved: {save_path.name}")


# ============================================================
# CLI entry point for quick testing
# ============================================================
if __name__ == '__main__':
    print("missing_charts.py — run individual functions with your data.")
    print("Available figures:")
    print("  plot_figure5_gpr_high_performance_region()")
    print("  plot_figureS_top6_scatter_matrix()")
    print("  plot_figureS2a_oxidiser_type_distribution()")
