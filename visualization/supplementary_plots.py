"""
Supplementary Material Visualization Module
=============================================

Generates all supplementary figures and tables for the IM-BO-UQ framework.
Each function corresponds to a specific supplementary item (S1--S11).

All figures follow professional standards: Arial font, 22pt base size, 300 DPI,
full frame (all four spines visible).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy import stats
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.feature_names import FEATURE_ABBREV, FEATURE_LATEX, get_display_name, get_filename_safe

# ============================================================
# Publication-Quality Style Configuration
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
# S1: Key Feature Distributions
# ============================================================

def plot_s1_key_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Figure S1: KDE + histogram distributions for 4 key features.

    Creates a 2x2 subplot layout showing the empirical distribution of each
    feature overlaid with a fitted normal density curve and descriptive
    statistics (mean, std, skewness).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature columns.
    features : list of str, optional
        Four feature column names to plot.  Defaults to
        ['ID/IG', 'C=O (%)', 'Catalyst dosage (g/L)',
         'Degradation efficiency (%)'].
    save_path : Path, optional
        File path to save the figure.  If None, the figure is shown
        interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Default features
    if features is None:
        features = [
            'ID/IG',
            'C=O (%)',
            'Catalyst dosage (g/L)',
            'Degradation efficiency (%)'
        ]

    # Display-name mapping for panel titles
    display_map = {
        'ID/IG': r'$I_D/I_G$',
        'C=O (%)': 'C=O (%)',
        'Catalyst dosage (g/L)': 'Catalyst dosage (g/L)',
        'Degradation efficiency (%)': 'Degradation efficiency (%)',
    }

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    colors = ['steelblue', 'coral', 'lightgreen', 'orange']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (feat, ax) in enumerate(zip(features, axes)):
        data = df[feat].dropna().values

        # Histogram with KDE
        ax.hist(
            data, bins=30, density=True, alpha=0.55,
            color=colors[idx], edgecolor='white', linewidth=0.8,
            label='Histogram'
        )

        # KDE overlay
        kde_x = np.linspace(data.min() - 0.1 * np.ptp(data),
                            data.max() + 0.1 * np.ptp(data), 300)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color=colors[idx], linewidth=3,
                label='KDE')

        # Normal fit overlay
        mu, sigma = stats.norm.fit(data)
        norm_y = stats.norm.pdf(kde_x, mu, sigma)
        ax.plot(kde_x, norm_y, '--', color='black', linewidth=2.5,
                label=f'Normal fit')

        # Statistics annotation
        skew = stats.skew(data)
        stats_text = (
            f'$\\mu$ = {mu:.2f}\n'
            f'$\\sigma$ = {sigma:.2f}\n'
            f'Skew = {skew:.2f}\n'
            f'n = {len(data)}'
        )
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85)
        )

        # Panel label and axis labels
        disp_name = display_map.get(feat, get_display_name(feat))
        ax.set_title(f'{panel_labels[idx]}  {disp_name}',
                     fontsize=24, fontweight='bold', loc='left')
        ax.set_xlabel(disp_name, fontsize=22)
        ax.set_ylabel('Density', fontsize=22)
        ax.legend(fontsize=16, loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    fig.tight_layout(pad=3.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S2: VIF Table
# ============================================================

def create_s2_vif_table(
    X_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for all features and save as CSV.

    VIF quantifies multicollinearity.  Features with VIF > 10 are typically
    flagged for potential removal (unless domain-mandated).

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix (samples x features).  All columns must be numeric.
    save_path : Path, optional
        Where to save the resulting CSV table.

    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame with columns ['Feature', 'Display Name', 'VIF'], sorted
        descending by VIF.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Drop rows with NaN for a clean OLS fit
    X_clean = X_df.dropna().copy()

    # Standardize to improve numerical stability of VIF computation
    X_std = (X_clean - X_clean.mean()) / X_clean.std().replace(0, 1)
    X_arr = X_std.values.astype(np.float64)

    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif_val = variance_inflation_factor(X_arr, i)
        except Exception:
            vif_val = np.nan
        vif_data.append({
            'Feature': col,
            'Display Name': get_display_name(col, use_latex=False),
            'VIF': round(vif_val, 2)
        })

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False).reset_index(drop=True)

    # Flag multicollinearity
    vif_df['Flag'] = vif_df['VIF'].apply(
        lambda v: 'HIGH' if v > 10 else ('MODERATE' if v > 5 else '')
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vif_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  [OK] VIF table saved: {save_path.name}")

    return vif_df


# ============================================================
# S3: Interaction Importance
# ============================================================

def plot_s3_interaction_importance(
    model,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Optional[Path] = None,
    degree: int = 2,
    top_k: int = 12
) -> plt.Figure:
    """
    Plot Figure S3: Interaction feature importance via polynomial expansion.

    Creates degree-2 interaction features using PolynomialFeatures
    (interaction_only=True), then fits a RandomForestRegressor to obtain
    feature importances.  The top-K interaction terms are displayed as a
    horizontal bar chart with LaTeX-formatted labels.

    Parameters
    ----------
    model : estimator
        Fitted primary model (not used here; included for API consistency).
    feature_names : list of str
        Names corresponding to columns of X_train.
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Training target vector.
    save_path : Path, optional
        File path to save the figure.
    degree : int
        Polynomial degree for interaction expansion (default 2).
    top_k : int
        Number of top interactions to display (default 12).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Create polynomial interaction features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_train)
    feature_names_poly = poly.get_feature_names_out(feature_names)

    # Fit RandomForest to compute importances on expanded feature space
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_poly, y_train)
    importances = rf.feature_importances_

    # Keep only interaction features (contain space or '^' in sklearn notation)
    interaction_mask = [any(' ' in name or '^' in name for name in [f])
                        for f in feature_names_poly]
    interaction_indices = np.where(interaction_mask)[0]

    if len(interaction_indices) == 0:
        interaction_indices = np.arange(len(feature_names_poly))

    interaction_importances = importances[interaction_indices]
    interaction_names = [feature_names_poly[i] for i in interaction_indices]

    # Sort and take top-K
    sorted_idx = np.argsort(interaction_importances)[::-1][:top_k]
    top_importances = interaction_importances[sorted_idx]
    top_names = [interaction_names[i] for i in sorted_idx]

    # Map to LaTeX display names
    top_names_display = []
    for name in top_names:
        name_clean = name.replace(' ', ' × ').replace('^2', '²')
        for orig in FEATURE_LATEX.keys():
            if orig in name_clean:
                name_clean = name_clean.replace(orig, get_display_name(orig, use_latex=True))
        top_names_display.append(name_clean[:50])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 11))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_importances)))
    bars = ax.barh(range(len(top_importances)), top_importances,
                   color=colors, edgecolor='black', linewidth=0.5)

    for i, (name, imp) in enumerate(zip(top_names_display, top_importances)):
        ax.text(imp, i, f' {imp:.3f}', va='center', fontsize=18)

    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names_display, fontsize=20)
    ax.set_xlabel('Importance Score', fontsize=22)
    ax.set_title('Figure S3: Interaction Feature Importance Ranking',
                 fontsize=22, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(pad=1.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S4: Local Correlation Analysis
# ============================================================

def plot_s4_local_correlation(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str, str]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Figure S4: Scatter + regression lines within specified value ranges.

    For each (feature, target, range_str) triple, the data is filtered to
    the specified range and a linear OLS regression is fitted.  R-squared
    and the p-value of the slope are annotated on each panel.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    pairs : list of (str, str, str)
        Each element is (feature_column, target_column, range_string).
        range_string follows the format "low-high" (e.g., "0.8-1.2").
    save_path : Path, optional
        File path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_pairs = len(pairs)
    ncols = min(n_pairs, 3)
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    panel_labels = [f'({chr(97 + i)})' for i in range(n_pairs)]

    for idx, (feat, target, range_str) in enumerate(pairs):
        ax = axes[idx]

        # Parse range
        low, high = [float(x.strip()) for x in range_str.split('-')]
        mask = (df[feat] >= low) & (df[feat] <= high)
        subset = df.loc[mask].dropna(subset=[feat, target])

        x = subset[feat].values
        y = subset[target].values

        # Scatter
        ax.scatter(x, y, alpha=0.5, s=50, color='steelblue',
                   edgecolor='white', linewidth=0.5, zorder=3)

        # Linear regression
        if len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', color='crimson', linewidth=3, zorder=4)

            # Confidence band (95%)
            n = len(x)
            x_mean = np.mean(x)
            se_line = std_err * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
            t_crit = stats.t.ppf(0.975, df=n - 2)
            ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                            alpha=0.15, color='crimson', zorder=2)

            # Annotation
            r2_text = (
                f'$R^2$ = {r_value**2:.3f}\n'
                f'p = {p_value:.2e}\n'
                f'n = {n}'
            )
            ax.text(
                0.97, 0.05, r2_text,
                transform=ax.transAxes, fontsize=17,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9)
            )
        else:
            ax.text(0.5, 0.5, 'Insufficient data',
                    transform=ax.transAxes, ha='center', va='center', fontsize=18)

        feat_disp = get_display_name(feat)
        target_disp = get_display_name(target)
        ax.set_xlabel(feat_disp, fontsize=20)
        ax.set_ylabel(target_disp, fontsize=20)
        ax.set_title(
            f'{panel_labels[idx]}  {feat_disp} vs {target_disp}\n'
            f'Range: [{low}, {high}]',
            fontsize=20, fontweight='bold', loc='left'
        )
        ax.grid(True, alpha=0.25, linewidth=0.8)

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(pad=3.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S5: Model Training Curves
# ============================================================

def plot_s5_training_curves(
    models_dict: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Optional[Path] = None,
    max_iter: int = 300
) -> plt.Figure:
    """
    Plot Figure S5: Multi-model training curve comparison.

    For each model, attempts to extract training history from
    ``model.evals_result_`` (XGBoost / LightGBM).  If not available,
    records R² at linearly-spaced training-set sizes as a proxy.
    Falls back to a single-point scatter when neither is possible.

    Parameters
    ----------
    models_dict : dict of {str: model}
        Up to 4 fitted model objects.
    X_train, y_train : np.ndarray
        Training data.
    X_val, y_val : np.ndarray
        Validation data.
    save_path : Path, optional
        File path to save the figure.
    max_iter : int
        Maximum iteration count for the manual fallback.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from sklearn.metrics import r2_score, mean_squared_error

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    model_names = list(models_dict.keys())[:4]

    for i, name in enumerate(model_names):
        ax = axes[i]
        model = models_dict[name]

        train_scores = []
        val_scores = []

        # Attempt to retrieve training history (XGBoost / LightGBM)
        if hasattr(model, 'evals_result_'):
            results = model.evals_result_
            train_scores = [1 - v for v in results['validation_0']['rmse']]
            val_scores = [1 - v for v in results['validation_1']['rmse']]
        else:
            # Manual fallback: record R² at increasing training sizes
            for n in range(10, min(max_iter, len(X_train)), max(1, max_iter // 50)):
                if hasattr(model, 'n_estimators'):
                    model_temp = type(model)(n_estimators=n, random_state=42)
                else:
                    model_temp = model
                model_temp.fit(X_train[:n], y_train[:n])
                train_pred = model_temp.predict(X_train[:n])
                val_pred = model_temp.predict(X_val)
                train_scores.append(r2_score(y_train[:n], train_pred))
                val_scores.append(r2_score(y_val, val_pred))

        if len(train_scores) == 0:
            # Single-point fallback
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            ax.scatter([1], [train_r2], s=100, label='Train R²', marker='o')
            ax.scatter([1], [val_r2], s=100, label='Val R²', marker='s')
            ax.text(0.5, 0.5, f'Train R²={train_r2:.3f}\nVal R²={val_r2:.3f}',
                    transform=ax.transAxes, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        else:
            iterations = range(len(train_scores))
            ax.plot(iterations, train_scores, 'o-', label='Train',
                    linewidth=2, markersize=4)
            ax.plot(iterations, val_scores, 's-', label='Validation',
                    linewidth=2, markersize=4)

        ax.set_xlabel('Iteration', fontsize=20)
        ax.set_ylabel('R² Score', fontsize=20)
        ax.set_title(f'Figure S5({chr(97 + i)}): {name} Training Curve',
                     fontsize=22, fontweight='bold')
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure S5: Model Training Curves Comparison',
                 fontsize=22, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S6: Residual Comparison Across Models
# ============================================================

def plot_s6_residual_comparison(
    models_dict: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Figure S6: Residual analysis for multiple models.

    For each model in `models_dict`, two panels are generated:
      - Left: histogram of residuals overlaid with a normal fit.
      - Right: Q-Q plot against the standard normal distribution.

    Parameters
    ----------
    models_dict : dict of {str: model}
        Mapping from model name to a fitted model object with a
        `predict(X)` method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.
    save_path : Path, optional
        File path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 6 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    colors = ['steelblue', 'coral', 'forestgreen', 'darkorange',
              'mediumpurple', 'goldenrod', 'teal', 'tomato']

    y_true = np.array(y_test).ravel()

    for row, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test).ravel()
        residuals = y_true - y_pred
        color = colors[row % len(colors)]

        # --- Left panel: Residual histogram ---
        ax_hist = axes[row, 0]
        ax_hist.hist(residuals, bins=30, density=True, alpha=0.6,
                     color=color, edgecolor='white', linewidth=0.8)

        # Normal fit overlay
        mu, sigma = stats.norm.fit(residuals)
        x_range = np.linspace(residuals.min() - 0.15 * np.ptp(residuals),
                              residuals.max() + 0.15 * np.ptp(residuals), 300)
        ax_hist.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                     '--', color='black', linewidth=2.5, label='Normal fit')

        # Shapiro-Wilk test for normality
        if len(residuals) <= 5000:
            sw_stat, sw_p = stats.shapiro(residuals)
        else:
            sw_stat, sw_p = np.nan, np.nan

        stat_text = (
            f'$\\mu$ = {mu:.3f}\n'
            f'$\\sigma$ = {sigma:.3f}\n'
            f'Shapiro p = {sw_p:.3e}'
        )
        ax_hist.text(
            0.97, 0.97, stat_text,
            transform=ax_hist.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85)
        )

        ax_hist.set_xlabel('Residual', fontsize=20)
        ax_hist.set_ylabel('Density', fontsize=20)
        ax_hist.set_title(f'{name} -- Residual Distribution',
                          fontsize=22, fontweight='bold')
        ax_hist.legend(fontsize=16, loc='upper left')
        ax_hist.grid(True, alpha=0.25)

        # --- Right panel: Q-Q plot ---
        ax_qq = axes[row, 1]
        theoretical_q = stats.norm.ppf(
            (np.arange(1, len(residuals) + 1) - 0.5) / len(residuals)
        )
        sorted_residuals = np.sort((residuals - mu) / sigma)

        ax_qq.scatter(theoretical_q, sorted_residuals, alpha=0.5, s=30,
                      color=color, edgecolor='white', linewidth=0.3, zorder=3)

        # Reference line
        qq_min = min(theoretical_q.min(), sorted_residuals.min())
        qq_max = max(theoretical_q.max(), sorted_residuals.max())
        ax_qq.plot([qq_min, qq_max], [qq_min, qq_max], '--', color='crimson',
                   linewidth=2.5, zorder=4, label='y = x')

        ax_qq.set_xlabel('Theoretical Quantiles', fontsize=20)
        ax_qq.set_ylabel('Standardized Residuals', fontsize=20)
        ax_qq.set_title(f'{name} -- Q-Q Plot', fontsize=22, fontweight='bold')
        ax_qq.legend(fontsize=16, loc='upper left')
        ax_qq.grid(True, alpha=0.25)

    fig.tight_layout(pad=3.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S7: Uncertainty by Feature Interval
# ============================================================

def plot_s7_uncertainty_by_feature_interval(
    model,
    uncertainty_func,
    feature_names: List[str],
    X_train: np.ndarray,
    intervals: List[Tuple[str, str]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Figure S7: Prediction uncertainty distribution within feature intervals.

    For each (feature, range_str) pair, samples whose feature value falls
    inside the specified range are selected.  The `uncertainty_func` is
    called on those samples and the resulting uncertainty distribution is
    displayed as a violin + strip plot.

    Parameters
    ----------
    model : estimator
        Fitted model (used internally by `uncertainty_func` if needed).
    uncertainty_func : callable
        A function with signature ``uncertainty_func(model, X) -> np.ndarray``
        that returns a 1-D array of per-sample uncertainty estimates.
    feature_names : list of str
        Column names matching the columns of X_train.
    X_train : np.ndarray
        Training feature matrix.
    intervals : list of (str, str)
        Each element is (feature_name, range_string) where range_string is
        "low-high" (e.g., "0.8-1.2").
    save_path : Path, optional
        File path to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_intervals = len(intervals)
    ncols = min(n_intervals, 3)
    nrows = int(np.ceil(n_intervals / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n_intervals == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    feature_idx_map = {name: i for i, name in enumerate(feature_names)}
    colors = ['steelblue', 'coral', 'forestgreen', 'darkorange',
              'mediumpurple', 'goldenrod', 'teal', 'tomato']
    panel_labels = [f'({chr(97 + i)})' for i in range(n_intervals)]

    for idx, (feat, range_str) in enumerate(intervals):
        ax = axes[idx]
        col_idx = feature_idx_map.get(feat)

        if col_idx is None:
            ax.text(0.5, 0.5, f'Feature "{feat}" not found',
                    transform=ax.transAxes, ha='center', va='center', fontsize=18)
            ax.set_title(f'{panel_labels[idx]}  {feat}', fontsize=20, fontweight='bold')
            continue

        # Parse range
        parts = range_str.replace(' ', '').split('-')
        # Handle negative numbers: re-join if first part is empty (negative lower bound)
        if parts[0] == '':
            low = -float(parts[1])
            high = float(parts[2]) if len(parts) > 2 else float(parts[1])
        else:
            low, high = float(parts[0]), float(parts[1])

        # Filter samples
        feat_vals = X_train[:, col_idx]
        mask = (feat_vals >= low) & (feat_vals <= high)
        X_subset = X_train[mask]

        if X_subset.shape[0] < 3:
            ax.text(0.5, 0.5, f'Too few samples (n={X_subset.shape[0]})',
                    transform=ax.transAxes, ha='center', va='center', fontsize=18)
            ax.set_title(f'{panel_labels[idx]}  {get_display_name(feat)}',
                         fontsize=20, fontweight='bold')
            continue

        # Compute uncertainties
        uncertainties = uncertainty_func(model, X_subset)

        color = colors[idx % len(colors)]

        # Violin plot
        parts_vp = ax.violinplot(uncertainties, positions=[0], showmeans=True,
                                 showmedians=True, widths=0.7)
        for pc in parts_vp['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        for key in ['cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars']:
            if key in parts_vp:
                parts_vp[key].set_color('black')
                parts_vp[key].set_linewidth(2)

        # Strip (jitter) overlay
        jitter = np.random.normal(0, 0.04, size=len(uncertainties))
        ax.scatter(jitter, uncertainties, alpha=0.35, s=25,
                   color=color, edgecolor='white', linewidth=0.3, zorder=3)

        # Statistics annotation
        stat_text = (
            f'Mean = {np.mean(uncertainties):.3f}\n'
            f'Median = {np.median(uncertainties):.3f}\n'
            f'Std = {np.std(uncertainties):.3f}\n'
            f'n = {len(uncertainties)}'
        )
        ax.text(
            0.97, 0.97, stat_text,
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85)
        )

        feat_disp = get_display_name(feat)
        ax.set_title(
            f'{panel_labels[idx]}  {feat_disp}  [{low}, {high}]',
            fontsize=20, fontweight='bold', loc='left'
        )
        ax.set_ylabel('Prediction Uncertainty', fontsize=20)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.25, linewidth=0.8)

    # Hide unused axes
    for idx in range(n_intervals, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(pad=3.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S8: Prediction Interval Width vs. Sample Size
# ============================================================

def plot_s8_prediction_interval_vs_sample_size(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
    alpha: float = 0.1,
    n_sizes: int = 10,
    n_repeats: int = 20,
    min_fraction: float = 0.1
) -> plt.Figure:
    """
    Plot Figure S8: Prediction interval width vs. training set size.

    Uses split conformal prediction to compute prediction intervals at
    varying fractions of the training data.  For each size, the experiment
    is repeated ``n_repeats`` times with random sub-sampling, and the mean
    interval width and empirical coverage are reported.

    Parameters
    ----------
    model : estimator
        A model class or fitted model with `fit` and `predict` methods.
        If the model has already been fitted it will be re-fitted on each
        subset.
    X_train, y_train : np.ndarray
        Full training set.
    X_test, y_test : np.ndarray
        Held-out test set for coverage evaluation.
    save_path : Path, optional
        File path to save the figure.
    alpha : float
        Mis-coverage rate for conformal prediction (default 0.1 = 90% PI).
    n_sizes : int
        Number of sample-size fractions to evaluate.
    n_repeats : int
        Repetitions per size for robust estimation.
    min_fraction : float
        Smallest training fraction to evaluate.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from sklearn.base import clone

    y_train_flat = np.array(y_train).ravel()
    y_test_flat = np.array(y_test).ravel()
    n_train = X_train.shape[0]

    fractions = np.linspace(min_fraction, 1.0, n_sizes)
    sizes = np.unique(np.clip((fractions * n_train).astype(int), 10, n_train))

    mean_widths = []
    std_widths = []
    mean_coverages = []
    std_coverages = []
    actual_sizes = []

    for size in sizes:
        widths_rep = []
        coverages_rep = []

        for _ in range(n_repeats):
            # Random sub-sample
            idx = np.random.choice(n_train, size=size, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train_flat[idx]

            # Split into proper training and calibration sets
            cal_size = max(int(0.25 * size), 5)
            train_size = size - cal_size
            if train_size < 5:
                continue

            X_proper = X_sub[:train_size]
            y_proper = y_sub[:train_size]
            X_cal = X_sub[train_size:]
            y_cal = y_sub[train_size:]

            # Fit model on proper training set
            try:
                model_clone = clone(model)
            except Exception:
                # If clone fails (e.g., wrapped model), use the model as-is
                model_clone = model
            try:
                model_clone.fit(X_proper, y_proper)
            except Exception:
                continue

            # Calibration residuals
            cal_pred = model_clone.predict(X_cal).ravel()
            cal_residuals = np.abs(y_cal - cal_pred)

            # Conformal quantile
            q_level = np.ceil((1 - alpha) * (len(cal_residuals) + 1)) / len(cal_residuals)
            q_level = min(q_level, 1.0)
            q_hat = np.quantile(cal_residuals, q_level)

            # Prediction intervals on test set
            test_pred = model_clone.predict(X_test).ravel()
            pi_lower = test_pred - q_hat
            pi_upper = test_pred + q_hat

            width = np.mean(pi_upper - pi_lower)
            coverage = np.mean((y_test_flat >= pi_lower) & (y_test_flat <= pi_upper))
            widths_rep.append(width)
            coverages_rep.append(coverage)

        if widths_rep:
            actual_sizes.append(size)
            mean_widths.append(np.mean(widths_rep))
            std_widths.append(np.std(widths_rep))
            mean_coverages.append(np.mean(coverages_rep))
            std_coverages.append(np.std(coverages_rep))

    actual_sizes = np.array(actual_sizes)
    mean_widths = np.array(mean_widths)
    std_widths = np.array(std_widths)
    mean_coverages = np.array(mean_coverages)
    std_coverages = np.array(std_coverages)

    # --- Dual-axis plot ---
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Interval width (left axis)
    color_width = 'steelblue'
    ax1.errorbar(actual_sizes, mean_widths, yerr=std_widths,
                 fmt='o-', color=color_width, linewidth=3, markersize=10,
                 capsize=5, capthick=2, label='Mean PI Width', zorder=3)
    ax1.fill_between(actual_sizes, mean_widths - std_widths, mean_widths + std_widths,
                     alpha=0.15, color=color_width)
    ax1.set_xlabel('Training Set Size', fontsize=22)
    ax1.set_ylabel('Mean Prediction Interval Width', fontsize=22, color=color_width)
    ax1.tick_params(axis='y', labelcolor=color_width)

    # Coverage (right axis)
    ax2 = ax1.twinx()
    color_cov = 'coral'
    ax2.errorbar(actual_sizes, mean_coverages, yerr=std_coverages,
                 fmt='s--', color=color_cov, linewidth=3, markersize=10,
                 capsize=5, capthick=2, label='Empirical Coverage', zorder=3)
    ax2.axhline(y=1 - alpha, color='gray', linestyle=':', linewidth=2,
                label=f'Target Coverage ({(1-alpha)*100:.0f}%)')
    ax2.set_ylabel('Empirical Coverage', fontsize=22, color=color_cov)
    ax2.tick_params(axis='y', labelcolor=color_cov)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=18,
               loc='upper right', framealpha=0.85)

    ax1.set_title('Prediction Interval Width and Coverage vs. Training Set Size',
                  fontsize=24, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.25, linewidth=0.8)

    fig.tight_layout(pad=2.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# S11: 3D Structure-Performance Surface
# ============================================================

def plot_s11_3d_structure_performance_surface(
    model,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    pairs: List[Tuple[str, str, str]],
    save_path: Optional[Path] = None,
    grid_resolution: int = 50
) -> plt.Figure:
    """
    Plot Figure S11: 3D surface plots of predicted performance.

    For each (feat1, feat2, target_name) triple, a 3D surface is generated
    by varying feat1 and feat2 over their training-set ranges while holding
    all other features at their median values.

    Parameters
    ----------
    model : estimator
        A fitted model with a `predict` method.
    feature_names : list of str
        Feature names matching the columns of X_train.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector (used only for color-bar range hints).
    pairs : list of (str, str, str)
        Each element is (feature_1_name, feature_2_name, target_display_name).
    save_path : Path, optional
        File path to save the figure.
    grid_resolution : int
        Number of grid points per axis (default 50).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    feature_idx = {name: i for i, name in enumerate(feature_names)}
    medians = np.median(X_train, axis=0)
    n_pairs = len(pairs)

    ncols = min(n_pairs, 2)
    nrows = int(np.ceil(n_pairs / ncols))
    fig = plt.figure(figsize=(12 * ncols, 10 * nrows))

    panel_labels = [f'({chr(97 + i)})' for i in range(n_pairs)]

    for idx, (feat1, feat2, target_name) in enumerate(pairs):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

        i1 = feature_idx.get(feat1)
        i2 = feature_idx.get(feat2)
        if i1 is None or i2 is None:
            ax.text2D(0.5, 0.5, f'Feature(s) not found:\n{feat1}, {feat2}',
                      transform=ax.transAxes, ha='center', va='center', fontsize=16)
            continue

        # Grid over feat1 x feat2
        x1_range = np.linspace(X_train[:, i1].min(), X_train[:, i1].max(), grid_resolution)
        x2_range = np.linspace(X_train[:, i2].min(), X_train[:, i2].max(), grid_resolution)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Build prediction matrix: all features at median, vary feat1 and feat2
        n_grid = grid_resolution * grid_resolution
        X_grid = np.tile(medians, (n_grid, 1))
        X_grid[:, i1] = X1.ravel()
        X_grid[:, i2] = X2.ravel()

        Z = model.predict(X_grid).ravel().reshape(X1.shape)

        # Surface plot
        surf = ax.plot_surface(
            X1, X2, Z, cmap='viridis', alpha=0.85,
            edgecolor='none', antialiased=True, rstride=2, cstride=2
        )

        # Scatter actual training data on top
        ax.scatter(
            X_train[:, i1], X_train[:, i2],
            np.array(y_train).ravel(),
            c='crimson', s=12, alpha=0.4, depthshade=True, zorder=5
        )

        feat1_disp = get_display_name(feat1)
        feat2_disp = get_display_name(feat2)
        ax.set_xlabel(feat1_disp, fontsize=18, labelpad=14)
        ax.set_ylabel(feat2_disp, fontsize=18, labelpad=14)
        ax.set_zlabel(target_name, fontsize=18, labelpad=14)
        ax.set_title(
            f'{panel_labels[idx]}  {feat1_disp} vs {feat2_disp}',
            fontsize=22, fontweight='bold', pad=20
        )

        # Adjust viewing angle for clarity
        ax.view_init(elev=25, azim=135)

        # Color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.1)
        cbar.set_label(target_name, fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    fig.tight_layout(pad=4.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {save_path.name}")
    return fig


# ============================================================
# Convenience: Generate all supplementary figures
# ============================================================

def generate_all_supplementary(
    df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model,
    models_dict: Optional[Dict[str, object]] = None,
    uncertainty_func=None,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Generate all supplementary figures and tables in one call.

    This is a convenience wrapper that calls each plot_s* / create_s*
    function with sensible defaults.  Individual functions should be
    called directly when custom parameters are needed.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (used for distribution and correlation plots).
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data.
    feature_names : list of str
        Feature names matching columns of X_train.
    model : estimator
        Primary fitted model (e.g., XGBoost).
    models_dict : dict, optional
        Multiple models for residual comparison.  If None, only the
        primary model is used.
    uncertainty_func : callable, optional
        Uncertainty estimation function for S7.
    output_dir : Path, optional
        Directory to save all outputs.  Defaults to ``outputs/figures/``.

    Returns
    -------
    saved : dict of {str: Path}
        Mapping from figure ID to the saved file path.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # S1: Key Feature Distributions
    print("[S1] Key feature distributions...")
    s1_path = output_dir / 'Figure_S1_key_feature_distributions.png'
    try:
        plot_s1_key_feature_distributions(df, save_path=s1_path)
        saved['S1'] = s1_path
    except Exception as e:
        print(f"  [WARN] S1 failed: {e}")

    # S2: VIF Table
    print("[S2] VIF table...")
    s2_path = output_dir / 'Table_S2_VIF.csv'
    try:
        X_df = df[feature_names] if isinstance(df, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_names)
        create_s2_vif_table(X_df, save_path=s2_path)
        saved['S2'] = s2_path
    except Exception as e:
        print(f"  [WARN] S2 failed: {e}")

    # S3: Interaction Importance
    print("[S3] Interaction importance...")
    s3_path = output_dir / 'Figure_S3_interaction_importance.png'
    try:
        plot_s3_interaction_importance(model, feature_names, X_train, y_train, save_path=s3_path)
        saved['S3'] = s3_path
    except Exception as e:
        print(f"  [WARN] S3 failed: {e}")

    # S4: Local Correlation
    print("[S4] Local correlation analysis...")
    s4_path = output_dir / 'Figure_S4_local_correlation.png'
    default_pairs = [
        ('ID/IG', 'Degradation efficiency (%)', '0.8-1.2'),
        ('C=O (%)', 'Degradation efficiency (%)', '5-25'),
        ('Catalyst dosage (g/L)', 'Degradation efficiency (%)', '0.1-1.0'),
    ]
    try:
        plot_s4_local_correlation(df, default_pairs, save_path=s4_path)
        saved['S4'] = s4_path
    except Exception as e:
        print(f"  [WARN] S4 failed: {e}")

    # S5: Training Curves
    print("[S5] Model training curves...")
    s5_path = output_dir / 'Figure_S5_training_curves.png'
    try:
        if models_dict is None:
            models_dict = {'Primary Model': model}
        # Split training set for validation proxy
        from sklearn.model_selection import train_test_split
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        plot_s5_training_curves(models_dict, X_tr, y_tr, X_vl, y_vl, save_path=s5_path)
        saved['S5'] = s5_path
    except Exception as e:
        print(f"  [WARN] S5 failed: {e}")

    # S6: Residual Comparison
    if models_dict is None:
        models_dict = {'Primary Model': model}
    print("[S6] Residual comparison...")
    s6_path = output_dir / 'Figure_S6_residual_comparison.png'
    try:
        plot_s6_residual_comparison(models_dict, X_test, y_test, save_path=s6_path)
        saved['S6'] = s6_path
    except Exception as e:
        print(f"  [WARN] S6 failed: {e}")

    # S7: Uncertainty by Feature Interval
    if uncertainty_func is not None:
        print("[S7] Uncertainty by feature interval...")
        s7_path = output_dir / 'Figure_S7_uncertainty_by_interval.png'
        default_intervals = [
            ('ID/IG', '0.8-1.2'),
            ('C=O (%)', '5-25'),
            ('Catalyst dosage (g/L)', '0.1-1.0'),
        ]
        try:
            plot_s7_uncertainty_by_feature_interval(
                model, uncertainty_func, feature_names, X_train,
                default_intervals, save_path=s7_path
            )
            saved['S7'] = s7_path
        except Exception as e:
            print(f"  [WARN] S7 failed: {e}")

    # S8: Prediction Interval vs Sample Size
    print("[S8] Prediction interval vs sample size...")
    s8_path = output_dir / 'Figure_S8_PI_vs_sample_size.png'
    try:
        plot_s8_prediction_interval_vs_sample_size(
            model, X_train, y_train, X_test, y_test, save_path=s8_path
        )
        saved['S8'] = s8_path
    except Exception as e:
        print(f"  [WARN] S8 failed: {e}")

    # S11: 3D Surface
    print("[S11] 3D structure-performance surface...")
    s11_path = output_dir / 'Figure_S11_3d_surface.png'
    default_3d_pairs = [
        ('ID/IG', 'C=O (%)', 'Degradation efficiency (%)'),
        ('Catalyst dosage (g/L)', 'Oxidiser dosage (mM)', 'Degradation efficiency (%)'),
    ]
    try:
        plot_s11_3d_structure_performance_surface(
            model, feature_names, X_train, y_train,
            default_3d_pairs, save_path=s11_path
        )
        saved['S11'] = s11_path
    except Exception as e:
        print(f"  [WARN] S11 failed: {e}")

    print(f"\n[DONE] Generated {len(saved)} supplementary items.")
    return saved
