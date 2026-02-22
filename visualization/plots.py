"""
Consolidated Visualization Module for Publication-Quality Figures.

This module provides four publication-quality plotting functions for analyzing
nitrogen-doped carbon (N-C) catalyst datasets:

  1. plot_target_distributions  -- Histogram+KDE / bar chart for target variables
  2. plot_feature_distributions -- Half-Violin + Box + Scatter for each feature
  3. plot_correlation_with_mi   -- Spearman heatmap (lower) + MI circles (upper)
  4. plot_prim_split_violin     -- Symmetric Raincloud: original vs PRIM-filtered

All figures follow professional standards:
  - Font: Arial, 22 pt base
  - Resolution: 300 DPI
  - Full frame (all four spines visible)
  - Tight bounding box on save
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.feature_names import (
    FEATURE_LATEX, FEATURE_ABBREV, get_display_name, get_filename_safe
)

# ============================================================
# Global rcParams — Publication-quality style
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
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
# 1. Target Variable Distributions
# ============================================================
def plot_target_distributions(df, target_eff, target_reuse, save_path):
    """
    Plot target variable distributions.

    Left panel: degradation efficiency — histogram with KDE overlay and mean line.
    Right panel: reuse cycles — bar chart (discrete integer values).

    Parameters
    ----------
    df : pd.DataFrame
    target_eff : str
        Column name for degradation efficiency.
    target_reuse : str
        Column name for reuse cycles.
    save_path : str or Path
        File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Efficiency: histogram + KDE
    sns.histplot(df[target_eff], kde=True, ax=axes[0], color='steelblue', bins=30)
    axes[0].axvline(df[target_eff].mean(), color='red', linestyle='--',
                    label='Mean', linewidth=2)
    axes[0].set_title('Efficiency Distribution', fontsize=22, fontweight='bold')
    axes[0].set_xlabel('Degradation Efficiency (%)', fontsize=20)
    axes[0].set_ylabel('Count', fontsize=20)
    axes[0].legend(fontsize=20)

    # Reuse cycles: bar chart (discrete)
    reuse_counts = df[target_reuse].value_counts().sort_index()
    axes[1].bar(reuse_counts.index, reuse_counts.values,
                color='coral', alpha=0.8, edgecolor='black')
    axes[1].set_title('Reuse Cycles Distribution', fontsize=22, fontweight='bold')
    axes[1].set_xlabel('Reuse Cycles', fontsize=20)
    axes[1].set_ylabel('Count', fontsize=20)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {Path(save_path).name}")


# ============================================================
# 2. Feature Distributions (Half-Violin + Box + Scatter)
# ============================================================
def plot_feature_distributions(df, feature_cols, save_path, colors=None):
    """
    Plot professional Half-Violin + Box + Scatter distribution figures.

    Layout: left jittered scatter | centre box plot | right half-violin.

    Generates one individual 10x10 figure per feature, plus a combined
    3x6 overview grid.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
    save_path : str or Path
        Path for the combined grid. Individual figures are saved in the
        same directory with suffixes ``_01_<name>.png`` etc.
    colors : list, optional
        Per-feature colours. Defaults to seaborn Set2 palette.
    """
    if colors is None:
        colors = sns.color_palette("Set2", len(feature_cols))

    save_path = Path(save_path)
    save_dir = save_path.parent
    base_name = save_path.stem

    # --- Individual feature plots ---
    for i, col in enumerate(feature_cols):
        fig, ax = plt.subplots(figsize=(10, 10))
        data = df[col].dropna().values
        color = colors[i]
        display_name = get_display_name(col, use_latex=True)
        filename_safe = get_filename_safe(col)

        pos = 0

        # 1. Right half-violin (clip to right side)
        parts = ax.violinplot([data], positions=[pos], showmeans=False,
                              showextrema=False, widths=0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
            pc.set_edgecolor(color)
            pc.set_linewidth(3)
            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], pos, pos + 1)

        # 2. Centre box plot
        ax.boxplot([data], positions=[pos], widths=0.15, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=color, alpha=0.7, linewidth=4),
                   whiskerprops=dict(color=color, linewidth=4),
                   capprops=dict(color=color, linewidth=4),
                   medianprops=dict(color='white', linewidth=4))

        # 3. Left jittered scatter
        n_sample = min(len(data), 200)
        if n_sample < len(data):
            sample_idx = np.random.choice(len(data), n_sample, replace=False)
            sample_data = data[sample_idx]
        else:
            sample_data = data

        x_jitter = np.random.uniform(pos - 0.35, pos - 0.15, size=len(sample_data))
        ax.scatter(x_jitter, sample_data, alpha=0.6, s=70, color=color,
                   edgecolors='white', linewidths=1.5, zorder=3)

        # Mean diamond marker
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        ax.scatter([pos], [mean_val], marker='D', s=150,
                   color='darkred', edgecolors='white', linewidths=2.5, zorder=5)

        # Statistics text box
        stats_text = (f'n = {len(data)}\nMean = {mean_val:.2f}\n'
                      f'Median = {median_val:.2f}\nSD = {std_val:.2f}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=18, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                family='Arial')

        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_ylabel('Value', fontsize=22, fontweight='bold')
        ax.tick_params(axis='y', labelsize=20, width=1.5, length=6)
        ax.set_title(display_name, fontsize=24, fontweight='bold', pad=15)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_edgecolor('black')

        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=2.0)

        individual_path = save_dir / f"{base_name}_{i+1:02d}_{filename_safe}.png"
        fig.savefig(individual_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Saved: {individual_path.name}")

    # --- Combined 3x6 grid ---
    n_rows, n_cols = 3, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 18))
    axes = axes.flatten() if len(feature_cols) > 1 else [axes]

    for i, col in enumerate(feature_cols):
        if i >= len(axes):
            break
        ax = axes[i]
        data = df[col].dropna().values
        color = colors[i]
        display_name = get_display_name(col, use_latex=True)
        pos = 0

        # Right half-violin
        parts = ax.violinplot([data], positions=[pos], showmeans=False,
                              showextrema=False, widths=0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
            pc.set_edgecolor(color)
            pc.set_linewidth(2.5)
            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], pos, pos + 1)

        # Centre box plot
        ax.boxplot([data], positions=[pos], widths=0.15, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=color, alpha=0.7, linewidth=3),
                   whiskerprops=dict(color=color, linewidth=3),
                   capprops=dict(color=color, linewidth=3),
                   medianprops=dict(color='white', linewidth=3))

        # Left jittered scatter
        n_sample = min(len(data), 100)
        if n_sample < len(data):
            sample_idx = np.random.choice(len(data), n_sample, replace=False)
            sample_data = data[sample_idx]
        else:
            sample_data = data

        x_jitter = np.random.uniform(pos - 0.35, pos - 0.15, size=len(sample_data))
        ax.scatter(x_jitter, sample_data, alpha=0.5, s=40, color=color,
                   edgecolors='white', linewidths=1, zorder=3)

        ax.scatter([pos], [np.mean(data)], marker='D', s=80,
                   color='darkred', edgecolors='white', linewidths=1.5, zorder=5)

        stats_text = f'n={len(data)}\nμ={np.mean(data):.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=16, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))

        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_ylabel('Value', fontsize=18)
        ax.tick_params(axis='y', labelsize=16, width=1.5, length=5)
        ax.set_title(display_name, fontsize=20, fontweight='bold', pad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=1)

    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distribution', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path.name}")


# ============================================================
# 3. Correlation Heatmap with Mutual Information Overlay
# ============================================================
def plot_correlation_with_mi(df, numeric_cols, save_path):
    """
    Plot Spearman correlation matrix with mutual information overlay.

    Lower triangle + diagonal: Spearman correlation (colour-mapped heatmap).
    Upper triangle: pairwise mutual information shown as scaled circles
    (YlOrRd colour scheme, size 100–800).  Normalised by the 95th percentile
    of all MI values for visual balance.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str
    save_path : str or Path

    Returns
    -------
    spearman_corr : pd.DataFrame
    """
    df_clean = df[numeric_cols].dropna()

    latex_rename = {col: get_display_name(col, use_latex=True) for col in numeric_cols}
    df_abbrev = df_clean.rename(columns=latex_rename)
    cols_abbrev = df_abbrev.columns.tolist()

    print("  Calculating Spearman correlation...")
    spearman_corr = df_abbrev.corr(method='spearman')

    print("  Calculating mutual information...")
    n_features = len(cols_abbrev)
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            X_i = df_abbrev.iloc[:, [i]].values
            y_j = df_abbrev.iloc[:, j].values
            mi = mutual_info_regression(X_i, y_j, random_state=42)[0]
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # Normalise by 95th percentile of upper-triangle values
    upper_tri_indices = np.triu_indices(n_features, k=1)
    upper_tri_values = mi_matrix[upper_tri_indices]
    positive_values = upper_tri_values[upper_tri_values > 0]
    mi_max = np.percentile(positive_values, 95) if len(positive_values) > 0 else 1.0
    mi_normalized = np.clip(mi_matrix / mi_max, 0, 1)

    print(f"    MI range: [{mi_matrix.min():.3f}, {mi_matrix.max():.3f}]")
    print(f"    MI 95th percentile: {mi_max:.3f}")

    print("  Creating heatmap...")
    fig, ax = plt.subplots(figsize=(20, 18))

    # Lower-triangle Spearman heatmap
    mask_lower = np.triu(np.ones_like(spearman_corr, dtype=bool), k=0)
    sns.heatmap(spearman_corr,
                mask=mask_lower,
                annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Spearman Correlation'},
                annot_kws={'size': 7}, linewidths=0.5, square=True)

    # White background for upper triangle
    for i in range(n_features):
        for j in range(i, n_features):
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       fill=True, facecolor='white',
                                       edgecolor='lightgray', linewidth=0.5,
                                       zorder=5))

    # MI circles in upper triangle (YlOrRd, size 100–800)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi_val = mi_normalized[i, j]
            size = 100 + mi_val * 700
            color = plt.cm.YlOrRd(mi_val * 0.7 + 0.3)
            edge_color = plt.cm.YlOrRd(min(mi_val * 0.7 + 0.4, 1.0))
            ax.scatter(j + 0.5, i + 0.5, s=size, c=[color],
                       alpha=0.85, edgecolors=edge_color, linewidths=2, zorder=10)
            if mi_val > 0.3:
                ax.text(j + 0.5, i + 0.5, f'{mi_val:.2f}',
                        ha='center', va='center', fontsize=16,
                        color='white' if mi_val > 0.6 else 'black',
                        fontweight='bold', zorder=11)

    ax.set_title('Correlation Matrix\n'
                 '(Lower: Spearman Correlation | Upper: Mutual Information)',
                 fontsize=22, fontweight='bold', pad=20)

    legend_elements = [
        plt.scatter([], [], s=300, c=plt.cm.YlOrRd(0.4), alpha=0.85,
                    edgecolors=plt.cm.YlOrRd(0.5), linewidths=2, label='Low MI (< 0.3)'),
        plt.scatter([], [], s=550, c=plt.cm.YlOrRd(0.65), alpha=0.85,
                    edgecolors=plt.cm.YlOrRd(0.75), linewidths=2, label='Medium MI (0.3–0.7)'),
        plt.scatter([], [], s=800, c=plt.cm.YlOrRd(0.9), alpha=0.85,
                    edgecolors=plt.cm.YlOrRd(1.0), linewidths=2, label='High MI (> 0.7)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1),
              title='MI Strength', frameon=True, fontsize=20, title_fontsize=22)

    plt.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {Path(save_path).name}")

    return spearman_corr


# ============================================================
# Helper: single-feature symmetric Raincloud panel
# ============================================================
def _plot_single_prim_feature(ax, feature, data_original, data_filtered,
                              display_name, bounds_abbrev):
    """
    Draw one symmetric Raincloud panel for a PRIM feature comparison.

    Layout (mirror symmetry):
      Left  — original data:  violin(left) | boxplot | scatter(right)
      Right — PRIM-filtered:  scatter(left) | boxplot | violin(right)

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    feature : str
        Column name (used for bounds lookup).
    data_original : np.ndarray
    data_filtered : np.ndarray
    display_name : str
        LaTeX-formatted label for title and bounds lookup.
    bounds_abbrev : dict
        Mapping of LaTeX display name to (min_val, max_val) PRIM bounds.
    """
    data_min = min(data_original.min(), data_filtered.min())
    data_max = max(data_original.max(), data_filtered.max())
    data_range = data_max - data_min
    margin = data_range * 0.08

    pos_left = -0.35
    pos_right = 0.35

    # === Left side: original data (violin→box→scatter) ===
    # 1. Half-violin (clip to left side)
    parts_left = ax.violinplot([data_original], positions=[pos_left],
                               vert=True, widths=0.8,
                               showmeans=False, showextrema=False)
    for pc in parts_left['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.5)
        pc.set_edgecolor('steelblue')
        pc.set_linewidth(3)
        vertices = pc.get_paths()[0].vertices
        vertices[:, 0] = np.clip(vertices[:, 0], -1, pos_left)

    # 2. Box plot (centre on pos_left)
    ax.boxplot([data_original], positions=[pos_left], vert=True, widths=0.15,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='steelblue', alpha=0.7, linewidth=4),
               whiskerprops=dict(color='steelblue', linewidth=4),
               capprops=dict(color='steelblue', linewidth=4),
               medianprops=dict(color='white', linewidth=4))

    # 3. Jittered scatter (right of box, toward centre)
    n_sample_left = min(len(data_original), 150)
    if n_sample_left < len(data_original):
        sample_idx_left = np.random.choice(len(data_original), n_sample_left, replace=False)
        data_left_sample = data_original[sample_idx_left]
    else:
        data_left_sample = data_original

    x_jitter_left = np.random.uniform(pos_left + 0.15, pos_left + 0.30,
                                       size=len(data_left_sample))
    ax.scatter(x_jitter_left, data_left_sample, alpha=0.6, s=70, color='steelblue',
               edgecolors='white', linewidths=1.5, zorder=3)

    ax.scatter([pos_left], [np.mean(data_original)], marker='D', s=150,
               color='darkblue', edgecolors='white', linewidths=2.5, zorder=5)

    # === Right side: PRIM-filtered data (scatter→box→violin) ===
    # 1. Jittered scatter (left of box, toward centre)
    n_sample_right = min(len(data_filtered), 150)
    if n_sample_right < len(data_filtered):
        sample_idx_right = np.random.choice(len(data_filtered), n_sample_right, replace=False)
        data_right_sample = data_filtered[sample_idx_right]
    else:
        data_right_sample = data_filtered

    x_jitter_right = np.random.uniform(pos_right - 0.30, pos_right - 0.15,
                                        size=len(data_right_sample))
    ax.scatter(x_jitter_right, data_right_sample, alpha=0.6, s=70, color='darkorange',
               edgecolors='white', linewidths=1.5, zorder=3)

    ax.scatter([pos_right], [np.mean(data_filtered)], marker='D', s=150,
               color='darkred', edgecolors='white', linewidths=2.5, zorder=5)

    # 2. Box plot (centre on pos_right)
    ax.boxplot([data_filtered], positions=[pos_right], vert=True, widths=0.15,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='darkorange', alpha=0.7, linewidth=4),
               whiskerprops=dict(color='darkorange', linewidth=4),
               capprops=dict(color='darkorange', linewidth=4),
               medianprops=dict(color='white', linewidth=4))

    # 3. Half-violin (clip to right side)
    parts_right = ax.violinplot([data_filtered], positions=[pos_right],
                                vert=True, widths=0.8,
                                showmeans=False, showextrema=False)
    for pc in parts_right['bodies']:
        pc.set_facecolor('orange')
        pc.set_alpha(0.5)
        pc.set_edgecolor('darkorange')
        pc.set_linewidth(3)
        vertices = pc.get_paths()[0].vertices
        vertices[:, 0] = np.clip(vertices[:, 0], pos_right, 1)

    # Centre dividing line
    ax.axvline(0, color='gray', linestyle='-', linewidth=2, alpha=0.5, zorder=1)

    # PRIM boundary lines (horizontal dashed)
    if display_name in bounds_abbrev:
        min_val, max_val = bounds_abbrev[display_name]
        ax.axhline(min_val, color='red', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
        ax.axhline(max_val, color='red', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
        ax.text(1.02, min_val, f'{min_val:.1f}',
                transform=ax.get_yaxis_transform(),
                fontsize=18, color='red', fontweight='bold', ha='left', va='center')
        ax.text(1.02, max_val, f'{max_val:.1f}',
                transform=ax.get_yaxis_transform(),
                fontsize=18, color='red', fontweight='bold', ha='left', va='center')

    # Axes and frame
    ax.set_xlim(-0.80, 0.80)
    ax.set_ylim(data_min - margin, data_max + margin)
    ax.set_xticks([])
    ax.set_ylabel('Value', fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18, width=2, length=6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
    ax.set_axisbelow(True)
    ax.set_title(display_name, fontsize=22, fontweight='bold', pad=10)

    # Sample-size annotations
    ax.text(0.02, 0.97, f'n={len(data_original)}',
            transform=ax.transAxes, fontsize=18, ha='left', va='top',
            color='steelblue', fontweight='bold')
    ax.text(0.98, 0.97, f'n={len(data_filtered)}',
            transform=ax.transAxes, fontsize=18, ha='right', va='top',
            color='darkorange', fontweight='bold')


# ============================================================
# 4. PRIM Split Violin (symmetric Raincloud)
# ============================================================
def plot_prim_split_violin(df, prim_bounds, top_features, save_path):
    """
    Plot PRIM process-window symmetric Raincloud figures.

    Left side (blue):   original data (all samples).
    Right side (orange): PRIM-window data (filtered samples).
    Red dashed lines:    PRIM boundary values.

    Saves N individual per-feature figures (12x9 each) plus one combined
    3x6 overview grid.

    Parameters
    ----------
    df : pd.DataFrame
    prim_bounds : dict
        Mapping of feature column name to (min_val, max_val).
    top_features : list of str or None
        Features to display.  If None, all PRIM-constrained features.
    save_path : str or Path
        Path for the combined grid.  Individual figures are saved in the
        same directory.
    """
    df_clean = df.dropna()

    # Apply PRIM filter
    mask = pd.Series([True] * len(df_clean), index=df_clean.index)
    for feature, (min_val, max_val) in prim_bounds.items():
        if feature in df_clean.columns:
            mask &= (df_clean[feature] >= min_val) & (df_clean[feature] <= max_val)

    df_original = df_clean.copy()
    df_filtered = df_clean[mask].copy()

    # Build bounds dict keyed by LaTeX display name (for boundary annotation)
    bounds_abbrev = {
        get_display_name(k, use_latex=True): v
        for k, v in prim_bounds.items()
    }

    if top_features is None:
        all_features = list(prim_bounds.keys())
    else:
        all_features = top_features

    all_features = [f for f in all_features if f in df_clean.columns]
    n_features = len(all_features)

    print(f"  Plotting {n_features} features in symmetric Raincloud layout...")

    save_path = Path(save_path)
    save_dir = save_path.parent
    base_name = save_path.stem

    # --- Individual per-feature figures ---
    for i, feature in enumerate(all_features):
        display_name = get_display_name(feature, use_latex=True)
        filename_safe = get_filename_safe(feature)

        data_original = df_original[feature].values
        data_filtered = df_filtered[feature].values

        fig, ax = plt.subplots(figsize=(12, 9))
        _plot_single_prim_feature(ax, feature, data_original, data_filtered,
                                   display_name, bounds_abbrev)
        plt.tight_layout(pad=1.5)

        individual_path = save_dir / f"{base_name}_{i+1:02d}_{filename_safe}.png"
        fig.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {individual_path.name}")

    # --- Combined 3x6 grid ---
    n_rows, n_cols_grid = 3, 6
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(30, 14))
    axes = axes.flatten()

    for i, feature in enumerate(all_features):
        ax = axes[i]
        display_name = get_display_name(feature, use_latex=True)
        data_original = df_original[feature].values
        data_filtered = df_filtered[feature].values
        _plot_single_prim_feature(ax, feature, data_original, data_filtered,
                                   display_name, bounds_abbrev)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'PRIM Process Window: Feature Distribution Comparison\n'
        'Left (Blue): Original Data | Right (Orange): PRIM Window | '
        'Red Lines: PRIM Boundaries',
        fontsize=22, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path.name}")
