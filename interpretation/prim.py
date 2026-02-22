"""
PRIM (Patient Rule Induction Method) Module

Identifies high-performance process parameter intervals with
Bootstrap stability analysis and FDR-corrected significance testing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Statistical tests and FDR correction
try:
    from scipy.stats import binomtest  # scipy >= 1.7
except ImportError:
    from scipy.stats import binom_test as binomtest  # scipy < 1.7
from statsmodels.stats.multitest import multipletests

RANDOM_SEED = 42


class PRIMConfig:
    """PRIM configuration."""
    alpha = 0.05                           # Peeling fraction per iteration
    min_support = 0.05                     # Minimum support
    efficiency_threshold_percentile = 85   # Percentile for high-performance threshold
    reuse_threshold_percentile = 80
    bootstrap_n_iterations = 50
    bootstrap_sample_ratio = 0.8
    stability_threshold = 0.85             # Frequency threshold for stable features
    stability_alpha = 0.05                 # FDR significance level


class SimplePRIM:
    """
    Simplified PRIM algorithm.

    Iteratively peels low-performance samples to discover
    high-performance parameter subspaces (boxes).
    """

    def __init__(self, config: Optional[PRIMConfig] = None):
        self.config = config or PRIMConfig()
        self.boxes: List[Dict] = []
        self.peeling_history: List[Dict] = []
        self.population_mean: Optional[float] = None
        self.threshold: Optional[float] = None

    def fit(self,
            X: pd.DataFrame,
            y: np.ndarray,
            threshold: Optional[float] = None,
            verbose: bool = True) -> 'SimplePRIM':
        """
        Fit the PRIM model.

        Data leakage protection:
        - X and y should be training-set data only.
        - If threshold is None, it is computed from y percentiles.

        Args:
            X: Feature matrix (DataFrame, training set only).
            y: Target variable (numpy array, training set only).
            threshold: High-performance threshold (explicit value recommended).
            verbose: Whether to print progress.

        Returns:
            self
        """
        y = np.asarray(y)

        if threshold is None:
            threshold = np.percentile(y, self.config.efficiency_threshold_percentile)

        self.threshold = threshold
        self.population_mean = y.mean()

        if verbose:
            print(f"PRIM analysis (threshold={threshold:.2f}, "
                  f"population mean={self.population_mean:.2f})...")

        # Initialize
        X_box = X.copy().reset_index(drop=True)
        y_box = y.copy()

        current_box = {}
        for col in X.columns:
            current_box[col] = {'min': X[col].min(), 'max': X[col].max()}

        alpha = self.config.alpha
        min_support = self.config.min_support
        min_samples = int(len(X) * min_support)

        iteration = 0
        while len(X_box) > min_samples:
            # Find the best peeling direction
            best_peel = None
            best_mean = (y_box >= threshold).mean() if len(y_box) > 0 else 0

            for col in X.columns:
                col_values = X_box[col].values
                n_peel = max(1, int(len(X_box) * alpha))

                # Try peeling from below
                sorted_idx = np.argsort(col_values)
                peel_idx = sorted_idx[:n_peel]
                remain_idx = sorted_idx[n_peel:]

                if len(remain_idx) >= min_samples:
                    new_mean = (y_box[remain_idx] >= threshold).mean()
                    if new_mean > best_mean:
                        best_mean = new_mean
                        best_peel = (col, 'low', peel_idx.tolist())

                # Try peeling from above
                peel_idx = sorted_idx[-n_peel:]
                remain_idx = sorted_idx[:-n_peel]

                if len(remain_idx) >= min_samples:
                    new_mean = (y_box[remain_idx] >= threshold).mean()
                    if new_mean > best_mean:
                        best_mean = new_mean
                        best_peel = (col, 'high', peel_idx.tolist())

            if best_peel is None:
                break

            col, direction, peel_idx = best_peel
            keep_idx = np.array([i for i in range(len(X_box)) if i not in peel_idx])

            # Update boundaries
            peeled_values = X_box.iloc[peel_idx][col]
            if direction == 'low':
                current_box[col]['min'] = peeled_values.max()
            else:
                current_box[col]['max'] = peeled_values.min()

            # Update data
            X_box = X_box.iloc[keep_idx].reset_index(drop=True)
            y_box = y_box[keep_idx]

            # Record history
            self.peeling_history.append({
                'iteration': iteration,
                'feature': col,
                'direction': direction,
                'support': len(X_box) / len(X),
                'precision': best_mean
            })

            iteration += 1
            if iteration > 100:  # Safety limit
                break

        # Compute box statistics
        box_mean = y_box.mean() if len(y_box) > 0 else 0
        lift = box_mean / self.population_mean if self.population_mean > 0 else 0

        self.boxes.append({
            'bounds': current_box.copy(),
            'support': len(X_box) / len(X),
            'precision': (y_box >= threshold).mean() if len(y_box) > 0 else 0,
            'n_samples': len(X_box),
            'box_mean': box_mean,
            'lift': lift
        })

        if verbose:
            self.print_summary()

        return self

    def get_windows(self) -> List[Dict]:
        """Get identified process windows (with Lift metric)."""
        windows = []
        for box in self.boxes:
            window = {}
            for feat, bounds in box['bounds'].items():
                if bounds['min'] != bounds['max']:
                    window[feat] = [bounds['min'], bounds['max']]
            windows.append({
                'bounds': window,
                'support': box['support'],
                'precision': box['precision'],
                'box_mean': box.get('box_mean', 0),
                'lift': box.get('lift', 0)
            })
        return windows

    def print_summary(self) -> None:
        """Print a summary of identified boxes."""
        if not self.boxes:
            print("No boxes identified.")
            return

        print("\nPRIM Results:")
        print("=" * 50)

        for i, box in enumerate(self.boxes):
            print(f"\nBox {i + 1}:")
            print(f"  Support: {box['support'] * 100:.1f}%")
            print(f"  Precision: {box['precision'] * 100:.1f}%")
            print(f"  Samples: {box['n_samples']}")
            print(f"  Box mean: {box.get('box_mean', 0):.2f}")
            lift = box.get('lift', 0)
            label = '[Excellent]' if lift >= 2.0 else '[Effective]' if lift >= 1.5 else ''
            print(f"  Lift: {lift:.2f} {label}")

            print("  Boundaries:")
            for feat, bounds in box['bounds'].items():
                if bounds['min'] != bounds['max']:
                    print(f"    {feat}: [{bounds['min']:.3f}, {bounds['max']:.3f}]")


def run_sensitivity_analysis(X: pd.DataFrame,
                             y: np.ndarray,
                             thresholds: List[float],
                             config: Optional[PRIMConfig] = None) -> pd.DataFrame:
    """
    Threshold sensitivity analysis for PRIM.

    Args:
        X: Feature matrix.
        y: Target variable.
        thresholds: List of thresholds to evaluate.
        config: PRIM configuration.

    Returns:
        DataFrame with sensitivity results.
    """
    results = []
    for threshold in thresholds:
        prim = SimplePRIM(config)
        prim.fit(X, y, threshold=threshold, verbose=False)
        if prim.boxes:
            box = prim.boxes[0]
            results.append({
                'threshold': threshold,
                'support': box['support'],
                'precision': box['precision'],
                'n_features': sum(1 for b in box['bounds'].values()
                                  if b['min'] != b['max'])
            })
    return pd.DataFrame(results)


class PRIMBootstrapAnalyzer:
    """
    PRIM Bootstrap Stability Analyzer.

    Evaluates the stability of PRIM interval boundaries via repeated
    bootstrap sampling and identifies statistically significant features.
    """

    def __init__(self, config: Optional[PRIMConfig] = None):
        self.config = config or PRIMConfig()
        self.bootstrap_results: List[Dict] = []
        self.feature_stability: Dict[str, Dict] = {}
        self.stable_bounds: Dict[str, Dict] = {}

    def run_bootstrap_analysis(self,
                               X: pd.DataFrame,
                               y: np.ndarray,
                               n_iterations: Optional[int] = None,
                               sample_ratio: Optional[float] = None,
                               threshold: Optional[float] = None,
                               verbose: bool = True,
                               n_jobs: int = -1) -> Dict:
        """
        Run bootstrap stability analysis.

        Args:
            X: Feature matrix.
            y: Target variable.
            n_iterations: Number of bootstrap iterations.
            sample_ratio: Fraction of data per bootstrap sample.
            threshold: High-performance threshold.
            verbose: Print progress.
            n_jobs: Parallel jobs (-1 = all CPUs).

        Returns:
            Dict with stability analysis results.
        """
        n_iterations = n_iterations or self.config.bootstrap_n_iterations
        sample_ratio = sample_ratio or self.config.bootstrap_sample_ratio

        if threshold is None:
            threshold = np.percentile(y, self.config.efficiency_threshold_percentile)

        if verbose:
            print(f"PRIM Bootstrap stability analysis "
                  f"(n={n_iterations}, threshold={threshold:.2f})...")

        all_bounds = defaultdict(list)
        feature_appearances = defaultdict(int)

        np.random.seed(RANDOM_SEED)
        n_samples = len(X)
        sample_size = int(n_samples * sample_ratio)

        if n_jobs != 1:
            from joblib import Parallel, delayed

            def single_bootstrap(i, X, y, n_samples, sample_size, threshold, config):
                """Single bootstrap iteration."""
                np.random.seed(RANDOM_SEED + i)
                idx = np.random.choice(n_samples, sample_size, replace=True)
                X_boot = X.iloc[idx].reset_index(drop=True)
                y_boot = y[idx]

                prim = SimplePRIM(config)
                prim.fit(X_boot, y_boot, threshold=threshold, verbose=False)

                result = {'iteration': i, 'box': None}
                if prim.boxes:
                    box = prim.boxes[0]
                    result['box'] = {
                        'bounds': box['bounds'],
                        'support': box['support'],
                        'precision': box['precision'],
                        'n_features': sum(1 for b in box['bounds'].values()
                                          if b['min'] != b['max'])
                    }
                return result

            results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(single_bootstrap)(
                    i, X, y, n_samples, sample_size, threshold, self.config)
                for i in range(n_iterations)
            )

            for result in results:
                if result['box']:
                    box = result['box']
                    for feat, bounds in box['bounds'].items():
                        if bounds['min'] != bounds['max']:
                            feature_appearances[feat] += 1
                            all_bounds[feat].append({
                                'min': bounds['min'], 'max': bounds['max']
                            })
                    self.bootstrap_results.append({
                        'iteration': result['iteration'],
                        'support': box['support'],
                        'precision': box['precision'],
                        'n_features': box['n_features']
                    })
        else:
            # Serial execution
            for i in range(n_iterations):
                idx = np.random.choice(n_samples, sample_size, replace=True)
                X_boot = X.iloc[idx].reset_index(drop=True)
                y_boot = y[idx]

                prim = SimplePRIM(self.config)
                prim.fit(X_boot, y_boot, threshold=threshold, verbose=False)

                if prim.boxes:
                    box = prim.boxes[0]
                    for feat, bounds in box['bounds'].items():
                        if bounds['min'] != bounds['max']:
                            feature_appearances[feat] += 1
                            all_bounds[feat].append({
                                'min': bounds['min'], 'max': bounds['max']
                            })
                    self.bootstrap_results.append({
                        'iteration': i,
                        'support': box['support'],
                        'precision': box['precision'],
                        'n_features': sum(1 for b in box['bounds'].values()
                                          if b['min'] != b['max'])
                    })

                if verbose and (i + 1) % 10 == 0:
                    print(f"  Iteration {i + 1}/{n_iterations} done")

        # ============================================================
        # Feature stability + FDR-corrected significance testing
        # ============================================================
        stability_threshold = self.config.stability_threshold

        features = []
        p_values = []

        for feat, count in feature_appearances.items():
            frequency = count / n_iterations

            # Binomial test: H0: feature frequency = random (0.5)
            if hasattr(binomtest, '__call__'):
                result = binomtest(count, n_iterations, 0.5, alternative='greater')
                p_value = result.pvalue if hasattr(result, 'pvalue') else result
            else:
                p_value = binomtest(count, n_iterations, 0.5, alternative='greater')

            features.append(feat)
            p_values.append(p_value)

            bounds_list = all_bounds[feat]
            if bounds_list:
                min_values = [b['min'] for b in bounds_list]
                max_values = [b['max'] for b in bounds_list]

                self.feature_stability[feat] = {
                    'frequency': frequency,
                    'p_value': p_value,
                    'min_mean': np.mean(min_values),
                    'min_std': np.std(min_values),
                    'min_ci_lower': np.percentile(min_values, 2.5),
                    'min_ci_upper': np.percentile(min_values, 97.5),
                    'max_mean': np.mean(max_values),
                    'max_std': np.std(max_values),
                    'max_ci_lower': np.percentile(max_values, 2.5),
                    'max_ci_upper': np.percentile(max_values, 97.5)
                }

        # FDR multiple comparison correction (Benjamini-Hochberg)
        if len(p_values) > 0:
            reject, p_adjusted, _, _ = multipletests(
                p_values,
                alpha=self.config.stability_alpha,
                method='fdr_bh'
            )

            for i, feat in enumerate(features):
                frequency = self.feature_stability[feat]['frequency']
                is_significant = reject[i]
                is_stable = (frequency >= stability_threshold) and is_significant

                self.feature_stability[feat]['p_value_fdr'] = p_adjusted[i]
                self.feature_stability[feat]['is_significant'] = is_significant
                self.feature_stability[feat]['is_stable'] = is_stable

                if is_stable:
                    bounds_list = all_bounds[feat]
                    min_values = [b['min'] for b in bounds_list]
                    max_values = [b['max'] for b in bounds_list]

                    self.stable_bounds[feat] = {
                        'min': np.percentile(min_values, 97.5),   # Conservative lower
                        'max': np.percentile(max_values, 2.5),    # Conservative upper
                        'min_mean': np.mean(min_values),
                        'max_mean': np.mean(max_values)
                    }

        if verbose:
            self.print_stability_summary()

        return {
            'feature_stability': self.feature_stability,
            'stable_bounds': self.stable_bounds,
            'bootstrap_results': self.bootstrap_results
        }

    def get_stable_features(self) -> List[str]:
        """Get list of stable features."""
        return [feat for feat, info in self.feature_stability.items()
                if info['is_stable']]

    def get_stable_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get stable bounds (conservative estimates)."""
        return {feat: (bounds['min'], bounds['max'])
                for feat, bounds in self.stable_bounds.items()}

    def print_stability_summary(self) -> None:
        """Print stability analysis summary."""
        print("\n" + "=" * 60)
        print("PRIM Bootstrap Stability Analysis Results")
        print("=" * 60)

        sorted_features = sorted(self.feature_stability.items(),
                                 key=lambda x: x[1]['frequency'],
                                 reverse=True)

        print(f"\nStable features (freq >= {self.config.stability_threshold * 100:.0f}% "
              f"AND FDR-adjusted p < {self.config.stability_alpha}):")
        print("-" * 80)
        print(f"{'Status':<10} {'Feature':<30} {'Freq':<10} "
              f"{'p-value':<12} {'FDR-adj p':<12}")
        print("-" * 80)

        stable_count = 0
        for feat, info in sorted_features:
            p_val = info.get('p_value', 1.0)
            p_fdr = info.get('p_value_fdr', 1.0)
            is_sig = info.get('is_significant', False)

            if info['is_stable']:
                status = "[Stable]"
                stable_count += 1
            elif is_sig:
                status = "[Signif]"
            else:
                status = "[Normal]"

            print(f"{status:<10} {feat[:30]:30s} {info['frequency']*100:5.1f}%    "
                  f"{p_val:>10.4e}  {p_fdr:>10.4e}")

            if info['is_stable']:
                bounds = self.stable_bounds[feat]
                print(f"           -> Bounds: [{bounds['min']:.3f}, {bounds['max']:.3f}] "
                      f"(mean: [{bounds['min_mean']:.3f}, {bounds['max_mean']:.3f}])")

        print("-" * 60)
        print(f"Stable features: {stable_count} / {len(self.feature_stability)}")

        if self.bootstrap_results:
            supports = [r['support'] for r in self.bootstrap_results]
            precisions = [r['precision'] for r in self.bootstrap_results]
            print(f"\nBootstrap statistics:")
            print(f"  Support:   {np.mean(supports)*100:.1f}% "
                  f"+/- {np.std(supports)*100:.1f}%")
            print(f"  Precision: {np.mean(precisions)*100:.1f}% "
                  f"+/- {np.std(precisions)*100:.1f}%")

    def to_dataframe(self, include_all: bool = False) -> pd.DataFrame:
        """
        Convert stability results to DataFrame.

        Args:
            include_all: If True, include all features; if False, only stable ones.
        """
        records = []
        for feat, info in self.feature_stability.items():
            if not include_all and not info.get('is_stable', False):
                continue
            record = {'feature': feat}
            record.update(info)
            if feat in self.stable_bounds:
                record['stable_min'] = self.stable_bounds[feat]['min']
                record['stable_max'] = self.stable_bounds[feat]['max']
            records.append(record)
        return pd.DataFrame(records).sort_values('frequency', ascending=False)


def plot_support_precision_tradeoff(prim: SimplePRIM,
                                    save_path: Optional[str] = None,
                                    title: str = 'PRIM Support-Precision Trade-off') -> None:
    """
    Plot the support-precision trade-off during PRIM peeling.

    Args:
        prim: Fitted SimplePRIM object.
        save_path: File path to save the figure.
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    if not prim.peeling_history:
        print("No peeling history available.")
        return

    iterations = [h['iteration'] for h in prim.peeling_history]
    supports = [h['support'] * 100 for h in prim.peeling_history]
    precisions = [h['precision'] * 100 for h in prim.peeling_history]

    # Prepend initial point (full dataset)
    iterations = [0] + iterations
    supports = [100.0] + supports
    initial_precision = precisions[0] if precisions else 0
    precisions = [initial_precision] + precisions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Support & Precision vs Iteration
    ax1.plot(iterations, supports, 'o-', label='Support', color='steelblue', linewidth=2)
    ax1.plot(iterations, precisions, 's-', label='Precision', color='coral', linewidth=2)
    ax1.axhline(85, color='gray', linestyle='--', alpha=0.5, label='Target Precision (85%)')
    ax1.set_xlabel('Peeling Iteration', fontsize=22)
    ax1.set_ylabel('Percentage (%)', fontsize=22)
    ax1.set_title('Support & Precision vs Iteration', fontsize=22, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Support-Precision Pareto front
    ax2.plot(supports, precisions, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax2.scatter(supports[0], precisions[0], color='green', s=100, zorder=5, label='Start')
    ax2.scatter(supports[-1], precisions[-1], color='red', s=100, zorder=5, label='End')

    if prim.boxes:
        box = prim.boxes[0]
        ax2.text(0.03, 0.97,
                 f"Support: {box['support']*100:.1f}%\n"
                 f"Precision: {box['precision']*100:.1f}%\n"
                 f"Lift: {box.get('lift', 0):.2f}",
                 fontsize=18, ha='left', va='top',
                 transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    ax2.axhline(85, color='gray', linestyle='--', alpha=0.5, label='Target (85%)')
    ax2.set_xlabel('Support (%)', fontsize=22)
    ax2.set_ylabel('Precision (%)', fontsize=22)
    ax2.set_title('Support-Precision Pareto Front', fontsize=22, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Support-precision curve saved: {save_path}")
    else:
        plt.show()
    plt.close()


def run_multi_objective_prim(X: pd.DataFrame,
                             y_eff: np.ndarray,
                             y_reuse: np.ndarray,
                             eff_min: float = 85.0,
                             reuse_min: float = 12.0,
                             objective: str = 'product',
                             config: Optional[PRIMConfig] = None,
                             verbose: bool = True) -> Tuple[SimplePRIM, Dict]:
    """
    Multi-objective PRIM analysis (efficiency + reuse cycles).

    Args:
        X: Feature matrix.
        y_eff: Degradation efficiency array.
        y_reuse: Reuse cycles array.
        eff_min: Minimum efficiency requirement.
        reuse_min: Minimum reuse cycles requirement.
        objective: Composite objective type
            - 'product': efficiency * reuse (default)
            - 'weighted_sum': 0.5*efficiency + 0.5*(reuse/max)*100
            - 'harmonic': harmonic mean
        config: PRIM configuration.
        verbose: Print progress.

    Returns:
        (prim, stats): Fitted PRIM object and statistics dict.
    """
    mask = (y_eff >= eff_min) & (y_reuse >= reuse_min)

    if verbose:
        print(f"\n>>> Multi-objective PRIM analysis")
        print(f"  Constraints: efficiency >= {eff_min}%, cycles >= {reuse_min}")
        print(f"  Feasible samples: {mask.sum()}/{len(mask)} "
              f"({mask.sum()/len(mask)*100:.1f}%)")

    # Relax constraints if no samples qualify
    if mask.sum() == 0:
        print(f"  [WARNING] No samples meet constraints, relaxing...")
        for reuse_try in [10, 8, 6, 4]:
            mask = (y_eff >= eff_min) & (y_reuse >= reuse_try)
            if mask.sum() > 0:
                print(f"  [ADJUSTED] Relaxed cycles to >= {reuse_try}, "
                      f"{mask.sum()} samples qualify")
                reuse_min = reuse_try
                break

        if mask.sum() == 0:
            for eff_try in [80, 75, 70]:
                mask = (y_eff >= eff_try) & (y_reuse >= 4)
                if mask.sum() > 0:
                    print(f"  [ADJUSTED] Relaxed efficiency to >= {eff_try}%, "
                          f"cycles >= 4, {mask.sum()} samples qualify")
                    eff_min = eff_try
                    reuse_min = 4
                    break

        if mask.sum() == 0:
            print(f"  [WARNING] Constraints too strict, using all data")
            mask = np.ones(len(y_eff), dtype=bool)

    X_filtered = X[mask].copy().reset_index(drop=True)
    y_eff_filtered = y_eff[mask]
    y_reuse_filtered = y_reuse[mask]

    # Compute composite objective
    if objective == 'product':
        reuse_normalized = (y_reuse_filtered / y_reuse_filtered.max()) * 100
        y_composite = y_eff_filtered * reuse_normalized
        obj_name = "efficiency * reuse"
    elif objective == 'weighted_sum':
        reuse_normalized = (y_reuse_filtered / y_reuse_filtered.max()) * 100
        y_composite = 0.5 * y_eff_filtered + 0.5 * reuse_normalized
        obj_name = "weighted sum (0.5*eff + 0.5*reuse)"
    elif objective == 'harmonic':
        reuse_normalized = (y_reuse_filtered / y_reuse_filtered.max()) * 100
        y_composite = 2 / (1 / y_eff_filtered + 1 / reuse_normalized)
        obj_name = "harmonic mean"
    else:
        raise ValueError(f"Unknown objective: {objective}")

    if verbose:
        print(f"  Objective function: {obj_name}")
        print(f"  Composite score range: "
              f"[{y_composite.min():.2f}, {y_composite.max():.2f}]")

    prim = SimplePRIM(config)
    prim.fit(X_filtered, y_composite, threshold=None, verbose=verbose)

    stats = {
        'n_total': len(mask),
        'n_feasible': int(mask.sum()),
        'feasible_ratio': float(mask.sum() / len(mask)),
        'objective_type': objective,
        'eff_constraint': eff_min,
        'reuse_constraint': reuse_min,
        'composite_min': float(y_composite.min()),
        'composite_max': float(y_composite.max()),
        'composite_mean': float(y_composite.mean())
    }

    return prim, stats


def scan_prim_parameters(X: pd.DataFrame,
                         y: np.ndarray,
                         alpha_values: List[float] = [0.05, 0.10, 0.15, 0.20],
                         min_support_values: List[float] = [0.03, 0.05, 0.08, 0.10],
                         threshold_percentile: int = 85,
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Grid-search PRIM parameters and generate support-precision heatmaps.

    Args:
        X: Feature matrix.
        y: Target variable.
        alpha_values: Peeling fractions to evaluate.
        min_support_values: Minimum supports to evaluate.
        threshold_percentile: Percentile for the threshold.
        save_path: Path to save the heatmap figure.

    Returns:
        DataFrame with grid-search results.
    """
    import matplotlib.pyplot as plt

    threshold = np.percentile(y, threshold_percentile)
    results = []

    print(f"\n>>> Parameter scan (threshold={threshold:.2f})")
    print(f"  alpha: {alpha_values}")
    print(f"  min_support: {min_support_values}")

    for alpha in alpha_values:
        for min_support in min_support_values:
            config = PRIMConfig()
            config.alpha = alpha
            config.min_support = min_support

            prim = SimplePRIM(config)
            prim.fit(X, y, threshold=threshold, verbose=False)

            if prim.boxes:
                box = prim.boxes[0]
                results.append({
                    'alpha': alpha,
                    'min_support': min_support,
                    'support': box['support'] * 100,
                    'precision': box['precision'] * 100,
                    'n_samples': box['n_samples'],
                    'lift': box.get('lift', 0),
                    'box_mean': box.get('box_mean', 0)
                })

    df = pd.DataFrame(results)

    if save_path and len(df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(26, 9))

        df['composite_score'] = df['precision'] * np.sqrt(df['support'])

        pivot_support = df.pivot(index='alpha', columns='min_support', values='support')
        pivot_precision = df.pivot(index='alpha', columns='min_support', values='precision')
        pivot_composite = df.pivot(index='alpha', columns='min_support',
                                   values='composite_score')

        # Support heatmap
        im1 = axes[0].imshow(pivot_support.values, cmap='YlOrRd', aspect='auto',
                             vmin=0, vmax=15)
        axes[0].set_xticks(range(len(min_support_values)))
        axes[0].set_xticklabels([f'{x:.2f}' for x in min_support_values], fontsize=20)
        axes[0].set_yticks(range(len(alpha_values)))
        axes[0].set_yticklabels([f'{x:.2f}' for x in alpha_values], fontsize=20)
        axes[0].set_xlabel('min_support', fontsize=22, fontweight='bold')
        axes[0].set_ylabel('alpha (peeling fraction)', fontsize=22, fontweight='bold')
        axes[0].set_title('Support (%)', fontsize=22, fontweight='bold')
        for i in range(len(alpha_values)):
            for j in range(len(min_support_values)):
                val = pivot_support.values[i, j]
                color = 'white' if val > 7.5 else 'black'
                axes[0].text(j, i, f'{val:.1f}', ha="center", va="center",
                             color=color, fontsize=18, fontweight='bold')
        plt.colorbar(im1, ax=axes[0]).set_label('Support (%)', fontsize=20)

        # Precision heatmap
        im2 = axes[1].imshow(pivot_precision.values, cmap='RdYlGn', aspect='auto',
                             vmin=50, vmax=100)
        axes[1].set_xticks(range(len(min_support_values)))
        axes[1].set_xticklabels([f'{x:.2f}' for x in min_support_values], fontsize=20)
        axes[1].set_yticks(range(len(alpha_values)))
        axes[1].set_yticklabels([f'{x:.2f}' for x in alpha_values], fontsize=20)
        axes[1].set_xlabel('min_support', fontsize=22, fontweight='bold')
        axes[1].set_ylabel('alpha (peeling fraction)', fontsize=22, fontweight='bold')
        axes[1].set_title('Precision (%)', fontsize=22, fontweight='bold')
        for i in range(len(alpha_values)):
            for j in range(len(min_support_values)):
                val = pivot_precision.values[i, j]
                color = 'white' if val < 75 else 'black'
                axes[1].text(j, i, f'{val:.1f}', ha="center", va="center",
                             color=color, fontsize=18, fontweight='bold')
        plt.colorbar(im2, ax=axes[1]).set_label('Precision (%)', fontsize=20)

        # Composite score heatmap
        im3 = axes[2].imshow(pivot_composite.values, cmap='viridis', aspect='auto')
        axes[2].set_xticks(range(len(min_support_values)))
        axes[2].set_xticklabels([f'{x:.2f}' for x in min_support_values], fontsize=20)
        axes[2].set_yticks(range(len(alpha_values)))
        axes[2].set_yticklabels([f'{x:.2f}' for x in alpha_values], fontsize=20)
        axes[2].set_xlabel('min_support', fontsize=22, fontweight='bold')
        axes[2].set_ylabel('alpha (peeling fraction)', fontsize=22, fontweight='bold')
        axes[2].set_title('Composite Score (Recommended)', fontsize=22, fontweight='bold')

        best_idx = df['composite_score'].idxmax()
        best_alpha = df.loc[best_idx, 'alpha']
        best_min_support = df.loc[best_idx, 'min_support']
        best_i = alpha_values.index(best_alpha)
        best_j = min_support_values.index(best_min_support)

        for i in range(len(alpha_values)):
            for j in range(len(min_support_values)):
                val = pivot_composite.values[i, j]
                if i == best_i and j == best_j:
                    axes[2].text(j, i, f'{val:.1f}\nBEST', ha="center", va="center",
                                 color='yellow', fontsize=20, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
                else:
                    axes[2].text(j, i, f'{val:.1f}', ha="center", va="center",
                                 color='white', fontsize=18, fontweight='bold')
        plt.colorbar(im3, ax=axes[2]).set_label('Score = Precision * sqrt(Support)',
                                                 fontsize=20)

        plt.suptitle(f'PRIM Parameter Grid Search '
                     f'(Threshold={threshold:.2f}, n={len(df)} combinations)',
                     fontsize=22, fontweight='bold', y=1.00)
        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Parameter scan heatmap saved: {save_path}")
        print(f"  [Recommended] Best: alpha={best_alpha:.2f}, "
              f"min_support={best_min_support:.2f}")
        print(f"         -> Support={df.loc[best_idx, 'support']:.1f}%, "
              f"Precision={df.loc[best_idx, 'precision']:.1f}%, "
              f"n_samples={df.loc[best_idx, 'n_samples']}")
        plt.close()

    return df
