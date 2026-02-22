#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Pipeline — IM-BO-UQ Framework

Generates all figures and result files identical to the full analysis.

Outputs:
- ~23 figures (PNG)
- ~4 CSV files
- ~4 JSON files
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import json
from scipy import stats

from sklearn.model_selection import (
    train_test_split, RepeatedKFold, cross_val_score, learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import partial_dependence
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import shap

warnings.filterwarnings('ignore')

# ============================================================
# Global configuration — imported from config module
# ============================================================
from config.settings import (
    RANDOM_SEED, TARGET_EFFICIENCY, TARGET_REUSE,
    DataConfig, FeatureConfig
)
from config.output_structure import output_structure
from config.feature_names import get_display_name, get_filename_safe

np.random.seed(RANDOM_SEED)


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
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
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True


def construct_features(df_input, epsilon_dict=None, fill_values=None,
                       feature_config=None):
    """
    Feature construction (configurable).

    Args:
        df_input: Input DataFrame.
        epsilon_dict: Division-by-zero protection parameters.
        fill_values: Missing-value fill dictionary.
        feature_config: FeatureConfig instance (None → default = no construction).

    Returns:
        (transformed DataFrame, epsilon_dict, fill_values)
    """
    from config.settings import FeatureConfig

    df_out = df_input.copy()

    if feature_config is None:
        feature_config = FeatureConfig()

    # If constructed features are disabled, just fill missing values and return
    if not feature_config.use_constructed_features:
        from utils.helpers import fill_missing_values
        df_out, fill_values = fill_missing_values(df_out, fill_values)
        return df_out, {}, fill_values

    # Compute epsilon for division-by-zero protection
    if epsilon_dict is None:
        epsilon_dict = {
            'N': max(abs(df_out['N (Wt%)'].std() * 0.01), 1e-6,
                     abs(df_out['N (Wt%)'].min() * 0.01)),
            'O': max(abs(df_out['O (Wt%)'].std() * 0.01), 1e-6,
                     abs(df_out['O (Wt%)'].min() * 0.01)),
            'diameter': max(abs(df_out['Surface particle diameter'].std() * 0.01),
                            1e-6,
                            abs(df_out['Surface particle diameter'].min() * 0.01)),
            'catalyst': max(abs(df_out['Catalyst dosage (g/L)'].std() * 0.01),
                            1e-6,
                            abs(df_out['Catalyst dosage (g/L)'].min() * 0.01)),
            'deepth': max(abs(df_out['deepth'].std() * 0.01), 1e-6,
                          abs(df_out['deepth'].min() * 0.01)),
        }
        for key in epsilon_dict:
            epsilon_dict[key] = max(epsilon_dict[key], 1e-6)

    # 1. Element ratio features
    if feature_config.use_ratio_features:
        denom_N = np.maximum(df_out['N (Wt%)'] + epsilon_dict['N'], epsilon_dict['N'])
        denom_O = np.maximum(df_out['O (Wt%)'] + epsilon_dict['O'], epsilon_dict['O'])
        df_out['C_N_ratio'] = df_out['C (Wt%)'] / denom_N
        df_out['C_O_ratio'] = df_out['C (Wt%)'] / denom_O
        df_out['N_O_ratio'] = df_out['N (Wt%)'] / denom_O

    # 2. Structural features
    if feature_config.use_structure_features:
        denom_diam = np.maximum(
            df_out['Surface particle diameter'] + epsilon_dict['diameter'],
            epsilon_dict['diameter'])
        df_out['aperture_diameter_ratio'] = df_out['aperture'] / denom_diam
        denom_depth = np.maximum(
            df_out['deepth'] + epsilon_dict['deepth'], epsilon_dict['deepth'])
        df_out['pore_efficiency'] = df_out['aperture'] / denom_depth

    # 3. Functional group features
    if feature_config.use_functional_features:
        df_out['active_N_total'] = (df_out['graphitic N (%)']
                                    + df_out['N-oxide/ Nitrate N(%)'])

    # 4. Process interaction features
    if feature_config.use_interaction_features:
        df_out['catalyst_time_interaction'] = (
            df_out['Catalyst dosage (g/L)'] * df_out['reaction time'])
        df_out['oxidiser_pH_synergy'] = (
            df_out['Oxidiser dosage (mM)'] * df_out['pH'])
        denom_cat = np.maximum(
            df_out['Catalyst dosage (g/L)'] + epsilon_dict['catalyst'],
            epsilon_dict['catalyst'])
        df_out['treatment_capacity'] = (
            df_out['Pollutant concentration (mg/L)'] / denom_cat)
        df_out['oxidiser_strength'] = (
            df_out['Oxidiser dosage (mM)'] * df_out['reaction time'])

    # 5. Nonlinear transforms
    if feature_config.use_nonlinear_features:
        for col in ['Catalyst dosage (g/L)', 'reaction time',
                     'Oxidiser dosage (mM)']:
            df_out[f'{col}_log'] = np.log1p(np.maximum(df_out[col], -0.999))
            df_out[f'{col}_sqrt'] = np.sqrt(np.maximum(df_out[col], 0))

    # Clean inf/nan
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = {}
        for col in numeric_cols:
            median_val = df_out[col].median()
            fill_values[col] = float(median_val) if not np.isnan(median_val) else 0

    for col in numeric_cols:
        df_out[col] = df_out[col].fillna(fill_values.get(col, 0))

    return df_out, epsilon_dict, fill_values


def calculate_vif(X_df):
    """Compute Variance Inflation Factor for all features."""
    vif_data = []
    for i, col in enumerate(X_df.columns):
        try:
            vif = variance_inflation_factor(X_df.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except:
            vif_data.append({'feature': col, 'VIF': np.inf})
    return pd.DataFrame(vif_data)


def smart_vif_removal(X_df, y, vif_threshold=10, max_iter=30, verbose=True):
    """
    Iterative VIF-based multicollinearity removal.

    Data leakage protection:
    - Must be called on training set only.
    - Returned feature list is applied to both train and test.
    """
    from config.settings import DataConfig
    mandatory_features = DataConfig().mandatory_features

    X_current = X_df.copy()
    removal_log = []

    for iteration in range(max_iter):
        rf = xgb.XGBRegressor(n_estimators=50, max_depth=4,
                               random_state=RANDOM_SEED, verbosity=0)
        rf.fit(X_current, y)
        importance = pd.Series(rf.feature_importances_, index=X_current.columns)
        importance = ((importance - importance.min())
                      / (importance.max() - importance.min() + 1e-10))

        vif_df = calculate_vif(X_current)
        max_vif = vif_df['VIF'].replace([np.inf], 1000).max()

        if max_vif <= vif_threshold:
            if verbose:
                print(f"  Iter {iteration}: all VIF <= {vif_threshold}")
            break

        high_vif = vif_df[
            vif_df['VIF'].replace([np.inf], 1000) >= vif_threshold].copy()
        if len(high_vif) == 0:
            break

        high_vif = high_vif[~high_vif['feature'].isin(mandatory_features)]
        if len(high_vif) == 0:
            if verbose:
                print("  All high-VIF features are mandatory, stopping removal")
            break

        high_vif['importance'] = high_vif['feature'].map(importance).fillna(0)
        vif_vals = high_vif['VIF'].replace([np.inf], 1000)
        high_vif['vif_norm'] = ((vif_vals - vif_vals.min())
                                / (vif_vals.max() - vif_vals.min() + 1e-10))
        high_vif['removal_score'] = (high_vif['vif_norm'] * 0.5
                                     - high_vif['importance'] * 0.5)

        remove_idx = high_vif['removal_score'].idxmax()
        remove_feature = high_vif.loc[remove_idx, 'feature']

        removal_log.append(remove_feature)
        X_current = X_current.drop(columns=[remove_feature])

        if verbose and (iteration < 5 or iteration % 5 == 0):
            print(f"  Iter {iteration}: removed {remove_feature}")

    for feat in mandatory_features:
        if feat in X_df.columns and feat not in X_current.columns:
            X_current[feat] = X_df[feat]

    return X_current.columns.tolist(), removal_log


class CompletePipeline:
    """Complete IM-BO-UQ pipeline — produces all outputs."""

    def __init__(self, data_path: str = 'data/sample_data.csv',
                 use_constructed_features: bool = False,
                 n_jobs: int = -1):
        """
        Initialize the pipeline.

        Args:
            data_path: Path to the CSV data file.
            use_constructed_features: Whether to engineer extra features
                                      (default False = raw features only).
            n_jobs: Parallel jobs (-1 = all CPUs).
        """
        self.data_path = data_path
        self.use_constructed_features = use_constructed_features
        self.n_jobs = n_jobs

        output_structure.create_all_directories()
        setup_matplotlib()

    def _make_pipeline(self, estimator, need_scale=False):
        """Build an sklearn Pipeline; linear models require scaling."""
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if need_scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', estimator))
        return Pipeline(steps)

    def _create_xgb_model(self, params=None, **kwargs):
        """Create an XGBoost regressor with unified defaults."""
        default_params = {
            'random_state': RANDOM_SEED,
            'verbosity': 0,
            'n_jobs': self.n_jobs
        }
        if params:
            default_params.update(params)
        default_params.update(kwargs)
        return xgb.XGBRegressor(**default_params)

    def _save_figure(self, fig, save_path, dpi=300, bbox_inches='tight'):
        """Save and close a figure."""
        from visualization.utils_viz import save_figure
        save_figure(fig, save_path, dpi, bbox_inches)

    # ==================================================================
    # Main entry point
    # ==================================================================
    def run_all(self, n_trials: int = 50):
        """Run the complete 13-step pipeline."""
        print("=" * 60)
        print("IM-BO-UQ Complete Pipeline")
        print("=" * 60)

        self._step1_load_data()
        self._step2_feature_engineering()
        self._step3_baseline_comparison()
        self._step4_bayesian_optimization(n_trials)
        self._step5_evaluation()
        self._step6_uncertainty()
        self._step7_prim()
        self._step8_shap()
        self._step9_pdp()
        self._step10_advanced()
        self._step12_supplementary()
        self._step11_save_results()

        print("\n" + "=" * 60)
        print("[Done] Pipeline completed!")
        figures_count = sum(
            1 for _ in output_structure.figures['base'].rglob('*.png'))
        results_count = sum(
            1 for _ in output_structure.results['base'].rglob('*') if _.is_file())
        print(f"  Figures: {figures_count}")
        print(f"  Result files: {results_count}")
        print("=" * 60)

    # ==================================================================
    # Step 1: Data Loading & Exploration
    # ==================================================================
    def _step1_load_data(self):
        """Step 1: Load data and produce exploratory plots."""
        print("\n>>> Step 1: Data loading")

        self.df = pd.read_csv(self.data_path, encoding='utf-8')
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        idx_all = np.arange(len(self.df))
        self.train_idx, self.test_idx = train_test_split(
            idx_all, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
        self.df_train = self.df.iloc[self.train_idx].copy().reset_index(drop=True)
        self.df_test = self.df.iloc[self.test_idx].copy().reset_index(drop=True)

        print(f"  Train: {len(self.df_train)}, Test: {len(self.df_test)}")

        from visualization.plots import (
            plot_target_distributions,
            plot_feature_distributions,
            plot_correlation_with_mi
        )

        # Figure: Target variable distributions
        plot_target_distributions(
            self.df, TARGET_EFFICIENCY, TARGET_REUSE,
            output_structure.get_figure_path(
                'data_exploration', 'fig_target_distributions.png'))

        # Figure: Feature distributions (raincloud style)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols
                        if c not in [TARGET_EFFICIENCY, TARGET_REUSE]]
        plot_feature_distributions(
            self.df, feature_cols,
            output_structure.get_figure_path(
                'data_exploration', 'fig_feature_distributions.png'))

        # Figure: Spearman correlation + mutual information
        spearman_corr = plot_correlation_with_mi(
            self.df, numeric_cols.tolist(),
            output_structure.get_figure_path(
                'data_exploration', 'fig_spearman_correlation.png'))
        spearman_corr.to_csv(output_structure.get_result_path(
            'features', 'spearman_correlation.csv'))

    # ==================================================================
    # Step 2: Feature Engineering
    # ==================================================================
    def _step2_feature_engineering(self):
        """Step 2: Feature engineering and VIF control."""
        print("\n>>> Step 2: Feature engineering")

        from config.settings import FeatureConfig
        feature_config = FeatureConfig(
            use_constructed_features=self.use_constructed_features)

        if self.use_constructed_features:
            print("  Mode: constructed features enabled")
        else:
            print("  Mode: raw features only (recommended)")

        df_train_eng, self.epsilon_dict, self.fill_values = construct_features(
            self.df_train, feature_config=feature_config)
        df_test_eng, _, _ = construct_features(
            self.df_test, self.epsilon_dict, self.fill_values,
            feature_config=feature_config)

        all_features = [c for c in df_train_eng.columns
                        if c not in [TARGET_EFFICIENCY, TARGET_REUSE]]
        numeric_features = (df_train_eng[all_features]
                            .select_dtypes(include=[np.number]).columns.tolist())
        X_train_num = df_train_eng[numeric_features].replace(
            [np.inf, -np.inf], np.nan)
        self.train_medians = X_train_num.median()

        if self.use_constructed_features:
            print("  Running VIF multicollinearity control...")
            self.final_features, _ = smart_vif_removal(
                X_train_num, df_train_eng[TARGET_EFFICIENCY])
            print(f"  Final feature count: {len(self.final_features)}")
        else:
            self.final_features = numeric_features
            print(f"  Skipping VIF (raw features only)")
            print(f"  Final feature count: {len(self.final_features)}")

        # Prepare final data matrices
        self.df_train_final = df_train_eng[
            self.final_features + [TARGET_EFFICIENCY, TARGET_REUSE]].copy()
        self.df_test_final = df_test_eng[
            self.final_features + [TARGET_EFFICIENCY, TARGET_REUSE]].copy()

        for col in self.final_features:
            self.df_train_final[col] = (
                self.df_train_final[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.train_medians.get(col, 0)))
            self.df_test_final[col] = (
                self.df_test_final[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.train_medians.get(col, 0)))

        self.X_train = self.df_train_final[self.final_features].values
        self.X_test = self.df_test_final[self.final_features].values
        self.y_eff_train = self.df_train_final[TARGET_EFFICIENCY].values
        self.y_eff_test = self.df_test_final[TARGET_EFFICIENCY].values
        self.y_reuse_train = self.df_train_final[TARGET_REUSE].values
        self.y_reuse_test = self.df_test_final[TARGET_REUSE].values

    # ==================================================================
    # Step 3: Baseline Model Comparison
    # ==================================================================
    def _step3_baseline_comparison(self):
        """Step 3: Compare 6 baseline models + learning curves."""
        print("\n>>> Step 3: Baseline model comparison")

        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

        models = {
            'Ridge': self._make_pipeline(
                Ridge(alpha=1.0, max_iter=10000), need_scale=True),
            'Lasso': self._make_pipeline(
                Lasso(alpha=0.1, max_iter=10000, tol=1e-3), need_scale=True),
            'SVR': self._make_pipeline(
                SVR(kernel='rbf', C=1.0), need_scale=True),
            'RandomForest': self._make_pipeline(
                RandomForestRegressor(
                    n_estimators=200, max_depth=10,
                    random_state=42, n_jobs=-1)),
            'XGBoost': self._make_pipeline(
                self._create_xgb_model(n_estimators=300, max_depth=6)),
            'LightGBM': self._make_pipeline(
                lgb.LGBMRegressor(
                    n_estimators=300, max_depth=6,
                    random_state=42, verbose=-1, n_jobs=-1))
        }

        self.cv_scores = {}
        for name, model in models.items():
            print(f"  Training {name}...", end=' ', flush=True)
            scores = cross_val_score(
                model, self.X_train, self.y_eff_train,
                cv=cv, scoring='r2', n_jobs=1)
            self.cv_scores[name] = scores
            print(f"R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

        # Learning curves
        print("  Computing learning curves...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        self.learning_curve_data = {}

        for i, (y, name) in enumerate([
            (self.y_eff_train, 'Efficiency'),
            (self.y_reuse_train, 'Reuse')
        ]):
            model = self._make_pipeline(
                self._create_xgb_model(n_estimators=200, max_depth=5))
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring='r2', n_jobs=1)

            self.learning_curve_data[name.lower()] = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }

            axes[i].fill_between(
                train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1),
                alpha=0.2, color='blue')
            axes[i].plot(train_sizes, train_scores.mean(1), 'o-',
                         label='Training', linewidth=2, markersize=6)
            axes[i].fill_between(
                train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1),
                alpha=0.2, color='orange')
            axes[i].plot(train_sizes, val_scores.mean(1), 's-',
                         label='Validation', linewidth=2, markersize=6)

            best_idx = np.argmax(val_scores.mean(1))
            best_size = train_sizes[best_idx]
            best_score = val_scores.mean(1)[best_idx]
            axes[i].axvline(best_size, color='green', linestyle=':',
                            linewidth=2, label=f'Best: {int(best_size)} samples')
            axes[i].plot(best_size, best_score, 'g*', markersize=15, zorder=5)

            axes[i].text(0.05, 0.95,
                         f'Max Val R²: {val_scores.mean(1).max():.3f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.85), fontsize=20)
            axes[i].set_xlabel('Training Set Size', fontsize=22)
            axes[i].set_ylabel('R² Score', fontsize=22)
            axes[i].set_title(f'{name} Learning Curve',
                              fontsize=22, fontweight='bold')
            axes[i].legend(loc='best', fontsize=20)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'model_training', 'fig_learning_curves.png'))

    # ==================================================================
    # Step 4: Bayesian Optimization
    # ==================================================================
    def _step4_bayesian_optimization(self, n_trials):
        """Step 4: Bayesian hyperparameter optimization with Optuna."""
        print("\n>>> Step 4: Bayesian optimization")

        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

        def create_objective(X, y):
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 0.01, 0.2, log=True),
                    'min_child_weight': trial.suggest_int(
                        'min_child_weight', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float(
                        'colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 1e-3, 10, log=True),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 1e-2, 50, log=True),
                }
                model = self._make_pipeline(self._create_xgb_model(params))
                return cross_val_score(
                    model, X, y, cv=cv, scoring='r2', n_jobs=-1).mean()
            return objective

        # Efficiency model
        sampler = TPESampler(seed=RANDOM_SEED)
        self.study_eff = optuna.create_study(
            direction='maximize', sampler=sampler)
        self.study_eff.optimize(
            create_objective(self.X_train, self.y_eff_train),
            n_trials=n_trials, show_progress_bar=True)
        self.best_params_eff = self.study_eff.best_params
        print(f"  Efficiency best CV R2: {self.study_eff.best_value:.4f}")

        # Reuse model
        sampler = TPESampler(seed=RANDOM_SEED)
        self.study_reuse = optuna.create_study(
            direction='maximize', sampler=sampler)
        self.study_reuse.optimize(
            create_objective(self.X_train, self.y_reuse_train),
            n_trials=n_trials, show_progress_bar=True)
        self.best_params_reuse = self.study_reuse.best_params
        print(f"  Reuse best CV R2: {self.study_reuse.best_value:.4f}")

        # Save BO history
        self.bo_history = {
            'efficiency': {
                'trial_values': [t.value for t in self.study_eff.trials
                                 if t.value is not None],
                'best_value': self.study_eff.best_value,
                'best_trial': (self.study_eff.best_trial.number
                               if self.study_eff.best_trial else 0)
            },
            'reuse': {
                'trial_values': [t.value for t in self.study_reuse.trials
                                 if t.value is not None],
                'best_value': self.study_reuse.best_value,
                'best_trial': (self.study_reuse.best_trial.number
                               if self.study_reuse.best_trial else 0)
            }
        }

        # BO convergence figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for i, (study, name) in enumerate([
            (self.study_eff, 'Efficiency'), (self.study_reuse, 'Reuse')
        ]):
            values = [t.value for t in study.trials if t.value is not None]
            window = max(5, len(values) // 10)
            if len(values) > window:
                moving_avg = pd.Series(values).rolling(window=window).mean()
                axes[i].plot(moving_avg.values, 'g-', linewidth=2, alpha=0.8,
                             label='Moving Average')
            axes[i].plot(values, 'o-', alpha=0.4, markersize=3,
                         label='Trial Values', linewidth=1)
            axes[i].axhline(study.best_value, color='red', linestyle='--',
                            linewidth=2, label=f'Best: {study.best_value:.4f}')
            best_trial_idx = np.argmax(values)
            axes[i].plot(best_trial_idx, study.best_value, 'r*',
                         markersize=15, label='Best Trial', zorder=5)
            improvement = ((study.best_value - values[0]) / abs(values[0]) * 100
                           if values[0] != 0 else 0)
            axes[i].text(0.05, 0.95, f'Improvement: {improvement:+.1f}%',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.85), fontsize=22)
            axes[i].set_xlabel('Trial', fontsize=22, fontweight='bold')
            axes[i].set_ylabel('CV R²', fontsize=22, fontweight='bold')
            axes[i].set_title(f'{name} BO Convergence',
                              fontsize=24, fontweight='bold')
            axes[i].legend(loc='best', fontsize=22)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'model_training', 'fig_bo_convergence.png'))

        # Hyperparameter importance figure
        try:
            from optuna.importance import get_param_importances
            self.hyperparam_importance = {}

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            for i, (study, name) in enumerate([
                (self.study_eff, 'Efficiency'), (self.study_reuse, 'Reuse')
            ]):
                imp = get_param_importances(study)
                sorted_items = sorted(imp.items(), key=lambda x: x[1],
                                      reverse=True)
                params, importances = zip(*sorted_items)

                self.hyperparam_importance[name.lower()] = {
                    'params': list(params),
                    'importances': list(importances)
                }
                colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
                axes[i].barh(range(len(params)), importances, color=colors,
                             edgecolor='black', linewidth=0.5)
                for j, (param, imp_val) in enumerate(zip(params, importances)):
                    axes[i].text(imp_val, j, f' {imp_val:.3f}',
                                 va='center', fontsize=22)
                axes[i].set_yticks(range(len(params)))
                axes[i].set_yticklabels(params, fontsize=22)
                axes[i].set_xlabel('Importance Score',
                                   fontsize=22, fontweight='bold')
                axes[i].set_title(f'{name} Hyperparameter Importance',
                                  fontsize=24, fontweight='bold')
                axes[i].grid(True, alpha=0.3, axis='x')
            plt.tight_layout(pad=1.5)
            self._save_figure(fig, output_structure.get_figure_path(
                'model_training', 'fig_hyperparam_importance.png'))
        except:
            pass

    # ==================================================================
    # Step 5: Test-Set Evaluation
    # ==================================================================
    def _step5_evaluation(self):
        """Step 5: Final model evaluation on the held-out test set."""
        print("\n>>> Step 5: Test-set evaluation")

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(self.X_train)
        X_test_imp = imputer.transform(self.X_test)

        self.model_eff = self._create_xgb_model(self.best_params_eff)
        self.model_eff.fit(X_train_imp, self.y_eff_train)
        self.y_eff_pred = self.model_eff.predict(X_test_imp)

        self.model_reuse = self._create_xgb_model(self.best_params_reuse)
        self.model_reuse.fit(X_train_imp, self.y_reuse_train)
        self.y_reuse_pred = self.model_reuse.predict(X_test_imp)

        self.X_train_imp = X_train_imp
        self.X_test_imp = X_test_imp

        print(f"  Efficiency R2: "
              f"{r2_score(self.y_eff_test, self.y_eff_pred):.4f}")
        print(f"  Reuse R2: "
              f"{r2_score(self.y_reuse_test, self.y_reuse_pred):.4f}")

        # Predicted vs Actual scatter
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for i, (y_true, y_pred, name) in enumerate([
            (self.y_eff_test, self.y_eff_pred, 'Efficiency'),
            (self.y_reuse_test, self.y_reuse_pred, 'Reuse')
        ]):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            from scipy.stats import linregress
            slope, intercept, r_val, p_val, std_err = linregress(y_true, y_pred)

            lims = [min(y_true.min(), y_pred.min()),
                    max(y_true.max(), y_pred.max())]
            x_line = np.linspace(lims[0], lims[1], 100)
            y_line = slope * x_line + intercept
            y_err = std_err * 1.96
            axes[i].fill_between(x_line, y_line - y_err, y_line + y_err,
                                 alpha=0.2, color='blue', label='95% CI')
            axes[i].plot(lims, lims, 'r--', linewidth=2, label='Perfect')
            axes[i].plot(x_line, y_line, 'b-', linewidth=1.5, alpha=0.7,
                         label='Regression')

            residuals = y_true - y_pred
            outlier_mask = np.abs(residuals) > 2 * residuals.std()
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=30,
                            edgecolors='black', linewidths=0.3)
            if outlier_mask.sum() > 0:
                axes[i].scatter(
                    y_true[outlier_mask], y_pred[outlier_mask],
                    s=80, facecolors='none', edgecolors='red',
                    linewidths=2, marker='o', label='Outliers (2σ)', zorder=5)

            textstr = (f'R² = {r2:.3f}\nMAE = {mae:.2f}\n'
                       f'RMSE = {rmse:.2f}\np = {p_val:.2e}')
            axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes,
                         verticalalignment='top',
                         bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.9), fontsize=20)
            axes[i].set_xlabel('Actual', fontsize=22)
            axes[i].set_ylabel('Predicted', fontsize=22)
            axes[i].set_title(f'{name}', fontsize=24, fontweight='bold')
            axes[i].legend(loc='lower right', fontsize=18)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'model_evaluation', 'fig_predicted_vs_actual.png'))

        # Residual analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for i, (y_true, y_pred, name) in enumerate([
            (self.y_eff_test, self.y_eff_pred, 'Efficiency'),
            (self.y_reuse_test, self.y_reuse_pred, 'Reuse')
        ]):
            residuals = y_true - y_pred

            # Residual histogram + normal fit
            axes[i, 0].hist(residuals, bins=30, edgecolor='white',
                            density=True, alpha=0.7)
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
            axes[i, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal')
            from scipy.stats import shapiro
            _, shapiro_p = shapiro(
                residuals[:5000] if len(residuals) > 5000 else residuals)
            axes[i, 0].text(
                0.05, 0.95, f'Shapiro-Wilk: p={shapiro_p:.3f}',
                transform=axes[i, 0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85),
                fontsize=18)
            axes[i, 0].set_ylabel('Density', fontsize=20)
            axes[i, 0].set_title(f'{name} Residual Histogram',
                                 fontsize=22, fontweight='bold')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Residuals vs Predicted + trend
            axes[i, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
            axes[i, 1].axhline(0, color='red', linestyle='--', linewidth=2)
            sorted_idx = np.argsort(y_pred)
            if len(y_pred) > 10:
                from scipy.interpolate import UnivariateSpline
                spline = UnivariateSpline(
                    y_pred[sorted_idx], residuals[sorted_idx],
                    s=len(y_pred) // 5)
                axes[i, 1].plot(y_pred[sorted_idx],
                                spline(y_pred[sorted_idx]),
                                'g-', linewidth=2, alpha=0.7, label='Trend')
            axes[i, 1].set_xlabel('Predicted', fontsize=20)
            axes[i, 1].set_ylabel('Residuals', fontsize=20)
            axes[i, 1].set_title(f'{name} Residuals vs Predicted',
                                 fontsize=22, fontweight='bold')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

            # Q-Q plot
            stats.probplot(residuals, dist='norm', plot=axes[i, 2])
            axes[i, 2].set_title(f'{name} Q-Q Plot',
                                 fontsize=22, fontweight='bold')
            axes[i, 2].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'model_evaluation', 'fig_residual_analysis.png'))

    # ==================================================================
    # Step 6: Uncertainty Quantification (Split Conformal Prediction)
    # ==================================================================
    def _step6_uncertainty(self):
        """Step 6: Uncertainty quantification via split conformal prediction."""
        print("\n>>> Step 6: Uncertainty quantification")

        # Split conformal
        X_proper, X_calib, y_proper, y_calib = train_test_split(
            self.X_train_imp, self.y_eff_train,
            test_size=0.25, random_state=RANDOM_SEED)

        model_proper = self._create_xgb_model(self.best_params_eff, n_jobs=1)
        model_proper.fit(X_proper, y_proper)

        calib_scores = np.abs(y_calib - model_proper.predict(X_calib))
        n = len(calib_scores)
        q_level = np.ceil((n + 1) * 0.9) / n
        self.quantile_eff = np.quantile(calib_scores, min(q_level, 1.0))

        self.y_eff_lower = self.y_eff_pred - self.quantile_eff
        self.y_eff_upper = self.y_eff_pred + self.quantile_eff

        coverage = np.mean((self.y_eff_test >= self.y_eff_lower)
                           & (self.y_eff_test <= self.y_eff_upper))
        print(f"  Coverage: {coverage*100:.1f}%")

        # Prediction interval figure
        fig, ax = plt.subplots(figsize=(14, 9))
        sorted_idx = np.argsort(self.y_eff_pred)
        x_axis = np.arange(len(sorted_idx))
        interval_widths = self.y_eff_upper - self.y_eff_lower
        in_interval = ((self.y_eff_test >= self.y_eff_lower)
                       & (self.y_eff_test <= self.y_eff_upper))

        ax.fill_between(x_axis,
                         self.y_eff_lower[sorted_idx],
                         self.y_eff_upper[sorted_idx],
                         alpha=0.3, color='blue',
                         label='90% Prediction Interval')
        ax.plot(x_axis, self.y_eff_pred[sorted_idx], 'b-',
                linewidth=1.5, label='Prediction')
        ax.scatter(x_axis, self.y_eff_test[sorted_idx],
                   c='gray', s=20, alpha=0.5, label='Actual', zorder=2)
        out_of_interval = ~in_interval
        if out_of_interval.sum() > 0:
            ax.scatter(
                x_axis[out_of_interval[sorted_idx]],
                self.y_eff_test[sorted_idx][out_of_interval[sorted_idx]],
                c='red', s=50, marker='x', linewidths=2,
                label='Out of Interval', zorder=5)

        textstr = (f'Coverage: {coverage*100:.1f}%\n'
                   f'Mean Width: {interval_widths.mean():.2f}\n'
                   f'Median Width: {np.median(interval_widths):.2f}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                fontsize=20)
        ax.set_xlabel('Test Sample Index (Sorted by Prediction)', fontsize=22)
        ax.set_ylabel('Efficiency (%)', fontsize=22)
        ax.set_title(f'Prediction Intervals (Coverage={coverage*100:.1f}%)',
                     fontsize=22, fontweight='bold')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'uncertainty', 'fig_prediction_intervals.png'))

        # Calibration curve
        fig, ax = plt.subplots(figsize=(10, 10))
        alphas = [0.5, 0.3, 0.2, 0.1, 0.05]
        actuals = []
        for alpha in alphas:
            q = np.ceil((n + 1) * (1 - alpha)) / n
            quantile = np.quantile(calib_scores, min(q, 1.0))
            lower = self.y_eff_pred - quantile
            upper = self.y_eff_pred + quantile
            cov = np.mean((self.y_eff_test >= lower)
                          & (self.y_eff_test <= upper))
            actuals.append(cov)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
                label='Perfect Calibration')
        ax.plot([1-a for a in alphas], actuals, 'o-',
                linewidth=2, markersize=8, label='Observed')
        for alpha, actual in zip(alphas, actuals):
            ax.annotate(f'{actual:.2f}', xy=(1-alpha, actual),
                        xytext=(5, 5), textcoords='offset points', fontsize=20)
        calibration_error = np.mean(
            [abs(1-a - act) for a, act in zip(alphas, actuals)])
        ax.text(0.05, 0.95,
                f'Mean Calibration Error: {calibration_error:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                fontsize=20)
        ax.set_xlabel('Nominal Coverage', fontsize=22)
        ax.set_ylabel('Actual Coverage', fontsize=22)
        ax.set_title('Calibration Curve', fontsize=22, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'uncertainty', 'fig_calibration_curve.png'))

        # .632+ Bootstrap
        n_boot = 100
        oob_scores = []
        for _ in range(n_boot):
            idx = np.random.choice(len(self.X_train_imp),
                                   len(self.X_train_imp), replace=True)
            oob_idx = np.array(
                [j for j in range(len(self.X_train_imp)) if j not in idx])
            if len(oob_idx) < 5:
                continue
            model = self._create_xgb_model(self.best_params_eff, n_jobs=1)
            model.fit(self.X_train_imp[idx], self.y_eff_train[idx])
            oob_scores.append(r2_score(
                self.y_eff_train[oob_idx],
                model.predict(self.X_train_imp[oob_idx])))

        oob_scores = np.array(oob_scores)
        ci_lower = np.percentile(oob_scores, 2.5)
        ci_upper = np.percentile(oob_scores, 97.5)
        original_r2 = r2_score(self.y_eff_test, self.y_eff_pred)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(oob_scores, bins=20, edgecolor='white', alpha=0.7, density=True)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue', label='95% CI')
        ax.axvline(ci_lower, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(ci_upper, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(np.mean(oob_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(oob_scores):.4f}')
        ax.axvline(original_r2, color='green', linestyle='-', linewidth=2,
                   label=f'Test R²: {original_r2:.4f}')
        textstr = (f'Mean: {np.mean(oob_scores):.4f}\n'
                   f'Std: {np.std(oob_scores):.4f}\n'
                   f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                fontsize=20)
        ax.set_xlabel('OOB R²', fontsize=22)
        ax.set_ylabel('Density', fontsize=22)
        ax.set_title('.632+ Bootstrap Distribution',
                     fontsize=22, fontweight='bold')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'uncertainty', 'fig_632plus_bootstrap.png'))

    # ==================================================================
    # Step 7: PRIM Process Windows + Bootstrap Stability
    # ==================================================================
    def _step7_prim(self):
        """Step 7: PRIM interval mining with bootstrap stability analysis."""
        print("\n>>> Step 7: PRIM process windows")

        from interpretation.prim import (
            SimplePRIM, run_sensitivity_analysis, PRIMBootstrapAnalyzer,
            plot_support_precision_tradeoff, run_multi_objective_prim,
            scan_prim_parameters)

        X_prim = self.df_train_final[self.final_features]
        y_prim = self.y_eff_train

        # 7.1 Basic single-objective PRIM (efficiency)
        prim = SimplePRIM()
        prim.fit(X_prim, y_prim)
        self.prim_windows = prim.get_windows()
        self.prim_peeling_history = prim.peeling_history.copy()

        plot_support_precision_tradeoff(
            prim,
            save_path=output_structure.get_figure_path(
                'process_optimization', 'fig_prim_support_precision.png'),
            title='PRIM Support-Precision Trade-off (Efficiency)')

        # 7.2 Bootstrap stability
        print("\n>>> Step 7.2: PRIM bootstrap stability analysis")
        bootstrap_analyzer = PRIMBootstrapAnalyzer()
        self.bootstrap_results = bootstrap_analyzer.run_bootstrap_analysis(
            X_prim, y_prim, n_iterations=50, verbose=True,
            n_jobs=self.n_jobs)

        stability_df = bootstrap_analyzer.to_dataframe()
        stability_df.to_csv(output_structure.get_result_path(
            'prim', 'prim_bootstrap_stability.csv'), index=False)

        # 7.3 Parameter scan
        print("\n>>> Step 7.3: PRIM parameter scan")
        scan_df = scan_prim_parameters(
            X_prim, y_prim,
            alpha_values=[0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20],
            min_support_values=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12],
            threshold_percentile=85,
            save_path=output_structure.get_figure_path(
                'process_optimization', 'fig_prim_parameter_scan.png'))
        scan_df.to_csv(output_structure.get_result_path(
            'prim', 'prim_parameter_scan.csv'), index=False)

        # 7.4 Multi-objective PRIM (efficiency + reuse)
        print("\n>>> Step 7.4: Multi-objective PRIM")
        prim_multi, multi_stats = run_multi_objective_prim(
            X_prim, self.y_eff_train, self.y_reuse_train,
            eff_min=85.0, reuse_min=8.0, objective='product', verbose=True)

        plot_support_precision_tradeoff(
            prim_multi,
            save_path=output_structure.get_figure_path(
                'process_optimization',
                'fig_prim_multi_objective_support_precision.png'),
            title='Multi-Objective PRIM (Efficiency × Reuse)')

        multi_windows = prim_multi.get_windows()
        if multi_windows:
            from utils.helpers import convert_numpy_types
            data_to_save = {
                'windows': convert_numpy_types(multi_windows),
                'stats': convert_numpy_types(multi_stats),
                'constraints': {'efficiency_min': 85.0, 'reuse_min': 8.0}
            }
            with open(output_structure.get_result_path(
                    'prim', 'prim_multi_objective_windows.json'),
                    'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        # Sensitivity analysis
        thresholds = [np.percentile(y_prim, p) for p in [70, 75, 80, 85, 90]]
        sens_df = run_sensitivity_analysis(X_prim, y_prim, thresholds)

        fig, ax = plt.subplots(figsize=(13, 9))
        ax.plot(sens_df['threshold'], sens_df['support'], 'o-', linewidth=2,
                markersize=8, label='Support', color='blue')
        ax.plot(sens_df['threshold'], sens_df['precision'], 's-', linewidth=2,
                markersize=8, label='Precision', color='orange')
        f1_scores = (2 * (sens_df['support'] * sens_df['precision'])
                     / (sens_df['support'] + sens_df['precision'] + 1e-10))
        best_idx = f1_scores.idxmax()
        best_threshold = sens_df.loc[best_idx, 'threshold']
        ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal: {best_threshold:.2f}')
        ax.fill_between(sens_df['threshold'], sens_df['support'],
                         sens_df['precision'], alpha=0.2, color='gray',
                         label='Trade-off Region')
        ax.set_xlabel('Threshold', fontsize=22)
        ax.set_ylabel('Support / Precision', fontsize=22)
        ax.set_title('PRIM Threshold Sensitivity',
                     fontsize=22, fontweight='bold')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'process_optimization', 'fig_prim_sensitivity.png'))

        # PRIM split violin
        if self.prim_windows:
            from visualization.plots import plot_prim_split_violin
            bounds = self.prim_windows[0]['bounds']
            plot_prim_split_violin(
                self.df_train_final, bounds, None,
                output_structure.get_figure_path(
                    'process_optimization', 'fig_prim_split_violin.png'))

        # Bootstrap stability plots
        self._plot_bootstrap_stability(bootstrap_analyzer)

    def _plot_bootstrap_stability(self, analyzer):
        """Plot bootstrap stability analysis results."""
        stability_df = analyzer.to_dataframe()
        if len(stability_df) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left: Feature frequency bar chart
        top_n = min(15, len(stability_df))
        top_features = stability_df.head(top_n)

        colors = ['#2ecc71' if s else '#95a5a6'
                  for s in top_features['is_stable']]
        axes[0].barh(range(top_n), top_features['frequency'],
                     color=colors, edgecolor='black', linewidth=0.5)
        for j, (idx, row) in enumerate(top_features.iterrows()):
            axes[0].text(row['frequency'], j, f' {row["frequency"]:.2f}',
                         va='center', fontsize=18)
        axes[0].set_yticks(range(top_n))
        feature_labels = [get_display_name(f, use_latex=True)
                          for f in top_features['feature']]
        axes[0].set_yticklabels(feature_labels, fontsize=18)
        axes[0].axvline(0.7, color='red', linestyle='--', linewidth=2,
                        label='Stability Threshold (70%)')
        axes[0].set_xlabel('Appearance Frequency', fontsize=22)
        axes[0].set_title('Feature Stability in PRIM Bootstrap',
                          fontsize=22, fontweight='bold')
        axes[0].legend(loc='best', fontsize=20)
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')

        # Right: Bootstrap support/precision distribution
        if analyzer.bootstrap_results:
            supports = [r['support'] for r in analyzer.bootstrap_results]
            precisions = [r['precision'] for r in analyzer.bootstrap_results]
            axes[1].hist(supports, bins=20, alpha=0.6, label='Support',
                         color='blue', edgecolor='black', density=True)
            axes[1].hist(precisions, bins=20, alpha=0.6, label='Precision',
                         color='orange', edgecolor='black', density=True)
            if len(supports) > 10:
                from scipy.stats import gaussian_kde
                kde_s = gaussian_kde(supports)
                kde_p = gaussian_kde(precisions)
                x_s = np.linspace(min(supports), max(supports), 100)
                x_p = np.linspace(min(precisions), max(precisions), 100)
                axes[1].plot(x_s, kde_s(x_s), 'b-', linewidth=2, alpha=0.8)
                axes[1].plot(x_p, kde_p(x_p), 'orange', linewidth=2, alpha=0.8)
            axes[1].set_xlabel('Value', fontsize=22)
            axes[1].set_ylabel('Density', fontsize=22)
            axes[1].set_title('Bootstrap Distribution',
                              fontsize=22, fontweight='bold')
            axes[1].legend(loc='best', fontsize=20)
            axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'process_optimization', 'fig_prim_bootstrap_stability.png'))

        # Stable bounds confidence intervals
        stable_features = stability_df[stability_df['is_stable']]
        if len(stable_features) > 0:
            n_stable = min(8, len(stable_features))
            fig, axes = plt.subplots(1, n_stable, figsize=(5*n_stable, 6))
            if n_stable == 1:
                axes = [axes]
            for i, (_, row) in enumerate(
                    stable_features.head(n_stable).iterrows()):
                ax = axes[i]
                feat = row['feature']
                ax.fill_between([0, 1],
                                [row['min_ci_lower'], row['min_ci_lower']],
                                [row['max_ci_upper'], row['max_ci_upper']],
                                alpha=0.2, color='blue', label='95% CI')
                ax.axhline(row['min_mean'], color='green', linestyle='-',
                           label='Mean Min')
                ax.axhline(row['max_mean'], color='red', linestyle='-',
                           label='Mean Max')
                if 'stable_min' in row and 'stable_max' in row:
                    ax.axhline(row['stable_min'], color='green',
                               linestyle='--', alpha=0.7)
                    ax.axhline(row['stable_max'], color='red',
                               linestyle='--', alpha=0.7)
                ax.set_title(get_display_name(feat, use_latex=True),
                             fontsize=20)
                ax.set_xticks([])
                if i == 0:
                    ax.legend(fontsize=18)
            plt.tight_layout(pad=1.5)
            self._save_figure(fig, output_structure.get_figure_path(
                'process_optimization', 'fig_prim_stable_bounds.png'))

    # ==================================================================
    # Step 8: SHAP Interpretability
    # ==================================================================
    def _step8_shap(self):
        """Step 8: SHAP feature importance analysis."""
        print("\n>>> Step 8: SHAP interpretability")

        explainer = shap.TreeExplainer(self.model_eff)
        self.shap_values_eff = explainer.shap_values(self.X_train_imp)

        explainer_reuse = shap.TreeExplainer(self.model_reuse)
        self.shap_values_reuse = explainer_reuse.shap_values(self.X_train_imp)

        # SHAP summary plot
        latex_features = [get_display_name(f, use_latex=True)
                          for f in self.final_features]
        fig = plt.figure(figsize=(14, 11))
        shap.summary_plot(
            self.shap_values_eff,
            pd.DataFrame(self.X_train_imp, columns=latex_features),
            max_display=15, show=False)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'interpretability', 'fig_shap_summary.png'))

        # Save SHAP importance CSVs
        importance_eff = pd.DataFrame({
            'feature': self.final_features,
            'importance': np.abs(self.shap_values_eff).mean(axis=0)
        }).sort_values('importance', ascending=False)
        importance_eff.to_csv(output_structure.get_result_path(
            'features', 'shap_efficiency.csv'), index=False)

        importance_reuse = pd.DataFrame({
            'feature': self.final_features,
            'importance': np.abs(self.shap_values_reuse).mean(axis=0)
        }).sort_values('importance', ascending=False)
        importance_reuse.to_csv(output_structure.get_result_path(
            'features', 'shap_reuse.csv'), index=False)

        # SHAP bootstrap importance
        n_boot = 50
        all_imp = []
        for _ in range(n_boot):
            idx = np.random.choice(len(self.X_train_imp),
                                   len(self.X_train_imp), replace=True)
            model = self._create_xgb_model(self.best_params_eff, n_jobs=1)
            model.fit(self.X_train_imp[idx], self.y_eff_train[idx])
            exp = shap.TreeExplainer(model)
            sv = exp.shap_values(self.X_train_imp[idx])
            all_imp.append(np.abs(sv).mean(axis=0))

        all_imp = np.array(all_imp)
        mean_imp = all_imp.mean(axis=0)
        std_imp = all_imp.std(axis=0)

        fig, ax = plt.subplots(figsize=(14, 9))
        sorted_idx = np.argsort(mean_imp)[::-1][:10]
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        ax.barh(np.arange(10), mean_imp[sorted_idx],
                xerr=std_imp[sorted_idx],
                color=colors, edgecolor='black', linewidth=0.5, capsize=3)
        for j, (si, mv, sv) in enumerate(
                zip(sorted_idx, mean_imp[sorted_idx], std_imp[sorted_idx])):
            ax.text(mv, j, f' {mv:.3f}±{sv:.3f}', va='center', fontsize=18)
        ax.set_yticks(np.arange(10))
        feature_labels = [get_display_name(self.final_features[i],
                                           use_latex=True)
                          for i in sorted_idx]
        ax.set_yticklabels(feature_labels, fontsize=20)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Importance (Mean ± Std)', fontsize=22)
        ax.set_title('SHAP Bootstrap Importance',
                     fontsize=22, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'interpretability', 'fig_shap_bootstrap.png'))

    # ==================================================================
    # Step 9: PDP Analysis
    # ==================================================================
    def _step9_pdp(self):
        """Step 9: Partial Dependence Plot analysis."""
        print("\n>>> Step 9: PDP analysis")

        top_features = np.argsort(
            np.abs(self.shap_values_eff).mean(axis=0))[::-1][:4]

        # PDP for Efficiency
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        for i, feat_idx in enumerate(top_features):
            pdp = partial_dependence(self.model_eff, self.X_train_imp,
                                     [feat_idx], grid_resolution=50)
            axes[i].plot(pdp['grid_values'][0], pdp['average'][0],
                         linewidth=2, color='blue')
            feature_vals = self.X_train_imp[:, feat_idx]
            ax2 = axes[i].twinx()
            ax2.hist(feature_vals, bins=30, alpha=0.2, color='gray',
                     density=True)
            ax2.set_ylabel('Density', color='gray', fontsize=18)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            effect_range = pdp['average'][0].max() - pdp['average'][0].min()
            axes[i].text(0.05, 0.95, f'Range: {effect_range:.2f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.85), fontsize=18)
            axes[i].set_xlabel(
                get_display_name(self.final_features[feat_idx],
                                 use_latex=True), fontsize=20)
            axes[i].set_ylabel('Partial Dependence', fontsize=20)
            axes[i].set_title('PDP', fontsize=20, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'interpretability', 'fig_pdp_efficiency.png'))

        # PDP for Reuse
        top_reuse = np.argsort(
            np.abs(self.shap_values_reuse).mean(axis=0))[::-1][:4]
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        for i, feat_idx in enumerate(top_reuse):
            pdp = partial_dependence(self.model_reuse, self.X_train_imp,
                                     [feat_idx], grid_resolution=50)
            axes[i].plot(pdp['grid_values'][0], pdp['average'][0],
                         linewidth=2, color='orange')
            feature_vals = self.X_train_imp[:, feat_idx]
            ax2 = axes[i].twinx()
            ax2.hist(feature_vals, bins=30, alpha=0.2, color='gray',
                     density=True)
            ax2.set_ylabel('Density', color='gray', fontsize=18)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            effect_range = pdp['average'][0].max() - pdp['average'][0].min()
            axes[i].text(0.05, 0.95, f'Range: {effect_range:.2f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(facecolor='white', edgecolor='none',
                                   alpha=0.85), fontsize=18)
            axes[i].set_xlabel(
                get_display_name(self.final_features[feat_idx],
                                 use_latex=True), fontsize=20)
            axes[i].set_ylabel('Partial Dependence', fontsize=20)
            axes[i].set_title('PDP', fontsize=20, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'interpretability', 'fig_pdp_reuse.png'))

        # 2D PDP interaction heatmap
        if len(top_features) >= 2:
            fig, ax = plt.subplots(figsize=(13, 10))
            pdp_2d = partial_dependence(
                self.model_eff, self.X_train_imp,
                [(top_features[0], top_features[1])], grid_resolution=20)
            x_grid = pdp_2d['grid_values'][0]
            y_grid = pdp_2d['grid_values'][1]
            z_values = pdp_2d['average'][0]
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

            contour = ax.contourf(X_grid, Y_grid, z_values,
                                  cmap='RdYlBu_r', levels=20, extend='both')
            contour_lines = ax.contour(X_grid, Y_grid, z_values,
                                       colors='black', alpha=0.3,
                                       linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=18)
            cbar = plt.colorbar(contour, ax=ax, aspect=20, pad=0.02)
            cbar.set_label('Predicted Efficiency', rotation=270,
                           labelpad=20, fontsize=20)
            interaction_strength = pdp_2d['average'][0].std()
            ax.text(0.05, 0.95,
                    f'Interaction: {interaction_strength:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.9), fontsize=20)
            ax.set_xlabel(get_display_name(
                self.final_features[top_features[0]], use_latex=True),
                fontsize=22)
            ax.set_ylabel(get_display_name(
                self.final_features[top_features[1]], use_latex=True),
                fontsize=22)
            ax.set_title('2D PDP Interaction',
                         fontsize=22, fontweight='bold')
            ax.set_xlim(x_grid.min(), x_grid.max())
            ax.set_ylim(y_grid.min(), y_grid.max())
            ax.set_aspect('auto')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout(pad=1.5)
            self._save_figure(fig, output_structure.get_figure_path(
                'interpretability', 'fig_feature_interaction_heatmap.png'))

    # ==================================================================
    # Step 10: Advanced Analysis (Pareto + Stacking + Ablation)
    # ==================================================================
    def _step10_advanced(self):
        """Step 10: Pareto front, stacking, and ablation study."""
        print("\n>>> Step 10: Advanced analysis")

        eff_pred = self.y_eff_pred
        reuse_pred = self.y_reuse_pred

        # Pareto front identification
        pareto_mask = np.zeros(len(eff_pred), dtype=bool)
        for i in range(len(eff_pred)):
            dominated = False
            for j in range(len(eff_pred)):
                if i != j:
                    if (eff_pred[j] >= eff_pred[i]
                            and reuse_pred[j] >= reuse_pred[i]):
                        if (eff_pred[j] > eff_pred[i]
                                or reuse_pred[j] > reuse_pred[i]):
                            dominated = True
                            break
            pareto_mask[i] = not dominated

        fig, ax = plt.subplots(figsize=(14, 10))
        pareto_points = np.array(
            [[eff_pred[i], reuse_pred[i]]
             for i in range(len(eff_pred)) if pareto_mask[i]])
        if len(pareto_points) > 0:
            pareto_sorted = pareto_points[pareto_points[:, 0].argsort()]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r-',
                    linewidth=2.5, alpha=0.8, label='Pareto Front', zorder=4)

        ax.scatter(self.y_eff_test, self.y_reuse_test, c='green',
                   s=40, alpha=0.4, marker='^', label='Actual Values',
                   zorder=2, edgecolors='black', linewidths=0.5)
        ax.scatter(eff_pred[~pareto_mask], reuse_pred[~pareto_mask],
                   alpha=0.3, c='gray', s=30, label='Non-Pareto', zorder=1)
        ax.scatter(eff_pred[pareto_mask], reuse_pred[pareto_mask],
                   c='red', s=120, marker='*', edgecolors='black',
                   linewidths=1, label='Pareto Points', zorder=5)
        ax.axvline(85.0, color='blue', linestyle=':', alpha=0.5,
                   linewidth=1.5, label='Target Efficiency')
        ax.axhline(8.0, color='blue', linestyle=':', alpha=0.5,
                   linewidth=1.5, label='Target Reuse')

        pareto_ratio = pareto_mask.sum() / len(pareto_mask)
        textstr = (f'Pareto: {pareto_mask.sum()}/{len(pareto_mask)} '
                   f'({pareto_ratio*100:.1f}%)\n'
                   f'Mean Eff: {eff_pred[pareto_mask].mean():.1f}\n'
                   f'Mean Reuse: {reuse_pred[pareto_mask].mean():.1f}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9),
                fontsize=20)
        ax.set_xlabel('Predicted Efficiency (%)', fontsize=22)
        ax.set_ylabel('Predicted Reuse (cycles)', fontsize=22)
        ax.set_title('Pareto Front (Test Set Predictions)',
                     fontsize=22, fontweight='bold')
        ax.legend(loc='lower left', fontsize=20, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'advanced_analysis', 'fig_pareto_front.png'))

        # Stacking ensemble
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
        base_estimators = [
            ('xgb', self._create_xgb_model(self.best_params_eff)),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=42)),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, random_state=42, verbose=-1))
        ]
        stacking = StackingRegressor(
            estimators=base_estimators, final_estimator=RidgeCV(), cv=5)
        stacking_scores = cross_val_score(
            stacking, self.X_train_imp, self.y_eff_train,
            cv=cv, scoring='r2', n_jobs=-1)
        xgb_scores = cross_val_score(
            self._create_xgb_model(self.best_params_eff),
            self.X_train_imp, self.y_eff_train,
            cv=cv, scoring='r2', n_jobs=-1)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot([xgb_scores, stacking_scores],
                   labels=['XGBoost', 'Stacking'],
                   patch_artist=False,
                   boxprops=dict(linewidth=2.5),
                   whiskerprops=dict(linewidth=2.5),
                   capprops=dict(linewidth=2.5),
                   medianprops=dict(linewidth=2.5, color='darkorange'),
                   flierprops=dict(markersize=8, markeredgewidth=1.5))
        ax.set_ylabel('CV R²', fontsize=22, fontweight='bold')
        ax.set_title('Stacking vs XGBoost', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20, width=2, length=6)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'advanced_analysis', 'fig_stacking_comparison.png'))

        # Ablation: Default vs BO-Optimized
        default_scores = cross_val_score(
            self._create_xgb_model(), self.X_train_imp, self.y_eff_train,
            cv=cv, scoring='r2', n_jobs=-1)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot([default_scores, xgb_scores],
                   labels=['Default', 'BO-Optimized'],
                   patch_artist=False,
                   boxprops=dict(linewidth=2.5),
                   whiskerprops=dict(linewidth=2.5),
                   capprops=dict(linewidth=2.5),
                   medianprops=dict(linewidth=2.5, color='darkorange'),
                   flierprops=dict(markersize=8, markeredgewidth=1.5))
        ax.set_ylabel('CV R²', fontsize=22, fontweight='bold')
        ax.set_title('Ablation: Default vs BO-Optimized',
                     fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20, width=2, length=6)
        plt.tight_layout(pad=1.5)
        self._save_figure(fig, output_structure.get_figure_path(
            'advanced_analysis', 'fig_ablation_study.png'))

    # ==================================================================
    # Step 12: Supplementary Material Visualization
    # ==================================================================
    def _step12_supplementary(self):
        """Step 12: Generate supplementary figures."""
        print("\n>>> Step 12: Supplementary visualizations")

        from visualization.supplementary_plots import (
            plot_s1_key_feature_distributions,
            plot_s3_interaction_importance,
            plot_s4_local_correlation,
            create_s2_vif_table,
            plot_s6_residual_comparison,
            plot_s7_uncertainty_by_feature_interval,
            plot_s8_prediction_interval_vs_sample_size,
            plot_s11_3d_structure_performance_surface
        )
        from visualization.utils_viz import safe_plot

        # Figure S1: Key feature distributions
        key_features = ['ID/IG', 'Carbonyl_ratio',
                        'Catalyst_dosage', 'Degradation_efficiency']
        plot_s1_key_feature_distributions(
            self.df, key_features,
            output_structure.get_figure_path(
                'supplementary', 'fig_s1_key_feature_distributions.png'))

        # Table S2: VIF multicollinearity
        X_df = pd.DataFrame(self.X_train_imp, columns=self.final_features)
        create_s2_vif_table(
            X_df,
            output_structure.get_result_path(
                'features', 'table_s2_vif_multicollinearity.csv'))

        # Figure S3: Interaction importance
        safe_plot(plot_s3_interaction_importance,
                  self.model_eff, self.final_features,
                  self.X_train_imp, self.y_eff_train,
                  output_structure.get_figure_path(
                      'supplementary', 'fig_s3_interaction_importance.png'))

        # Figure S4: Local correlation
        local_corr_pairs = [
            ('ID/IG', 'Degradation efficiency (%)', '1.15~1.45'),
            ('C=O (%)', 'reuse-times', '42~55'),
            ('Oxidiser dosage (mM)', 'Degradation efficiency (%)', '4~7')
        ]
        corr_df = self.df_train_final.copy()
        plot_s4_local_correlation(
            corr_df, local_corr_pairs,
            output_structure.get_figure_path(
                'supplementary', 'fig_s4_local_correlation.png'))

        # Figure S6: Multi-model residual comparison
        safe_plot(self._plot_s6_residuals)

        # Figure S7–S8, S11
        safe_plot(self._plot_s7_uncertainty)
        safe_plot(plot_s8_prediction_interval_vs_sample_size,
                  self._create_xgb_model(self.best_params_eff),
                  self.X_train_imp, self.y_eff_train,
                  self.X_test_imp, self.y_eff_test,
                  output_structure.get_figure_path(
                      'supplementary',
                      'fig_s8_prediction_interval_vs_sample_size.png'))

        surface_pairs = [
            ('ID/IG', 'C=O (%)', 'Efficiency'),
            ('graphitic N (%)', 'Fe-O/C-O (%)', 'Reuse Cycles')]
        adjusted = [(f1, f2, t) for f1, f2, t in surface_pairs
                    if f1 in self.final_features
                    and f2 in self.final_features]
        if adjusted:
            safe_plot(plot_s11_3d_structure_performance_surface,
                      self.model_eff, self.final_features,
                      self.X_train_imp, self.y_eff_train, adjusted,
                      output_structure.get_figure_path(
                          'supplementary',
                          'fig_s11_3d_structure_performance.png'))

        # Additional supplementary plots
        from visualization.missing_charts import (
            plot_figureS_top6_scatter_matrix,
            plot_figureS2a_oxidiser_type_distribution)
        from visualization.utils_viz import get_top_features_by_importance

        safe_plot(self._plot_figure5_gpr)
        top6 = get_top_features_by_importance(
            self.shap_values_eff, self.final_features, 6)
        safe_plot(plot_figureS_top6_scatter_matrix,
                  self.df_train_final, top6, TARGET_EFFICIENCY,
                  output_structure.get_figure_path(
                      'supplementary',
                      'fig_s_top6_scatter_matrix.png'))
        safe_plot(plot_figureS2a_oxidiser_type_distribution,
                  self.df,
                  output_structure.get_figure_path(
                      'supplementary',
                      'fig_s2a_oxidant_type_distribution.png'))

        print("  [Done] Supplementary visualizations completed")

    def _plot_s6_residuals(self):
        """Figure S6: Multi-model residual comparison (helper)."""
        from visualization.supplementary_plots import plot_s6_residual_comparison

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train_imp)

        class ModelWrapper:
            def __init__(self, model, scaler=None):
                self.model, self.scaler = model, scaler
            def predict(self, X):
                if self.scaler:
                    X = self.scaler.transform(X)
                return self.model.predict(X)

        models = {
            'XGBoost': ModelWrapper(self.model_eff),
            'LightGBM': ModelWrapper(
                lgb.LGBMRegressor(**self.best_params_eff,
                                  random_state=42, verbose=-1))}
        models['LightGBM'].model.fit(self.X_train_imp, self.y_eff_train)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF
            gpr = GaussianProcessRegressor(kernel=RBF(), random_state=42)
            gpr.fit(X_train_scaled, self.y_eff_train)
            models['GPR'] = ModelWrapper(gpr, scaler)
        except:
            pass

        svr = SVR(kernel='rbf', C=1.0)
        svr.fit(X_train_scaled, self.y_eff_train)
        models['SVR'] = ModelWrapper(svr, scaler)

        plot_s6_residual_comparison(
            models, self.X_test_imp, self.y_eff_test,
            output_structure.get_figure_path(
                'supplementary', 'fig_s6_residual_comparison.png'))

    def _plot_s7_uncertainty(self):
        """Figure S7: Feature interval uncertainty (helper)."""
        from visualization.supplementary_plots import (
            plot_s7_uncertainty_by_feature_interval)

        residuals = self.y_eff_train - self.model_eff.predict(self.X_train_imp)
        uncertainty_func = lambda X: np.std(residuals) * np.ones(len(X))
        intervals = [
            ('ID/IG', '1.15~1.45'),
            ('C=O (%)', '42~55'),
            ('Catalyst dosage (g/L)', '0.6~1.0')]
        adjusted = []
        for f, i in intervals:
            if f in self.final_features:
                adjusted.append((f, i))
            elif f == 'C=O (%)' and 'C=O/C-O（%）' in self.final_features:
                adjusted.append(('C=O/C-O（%）', i))
        if adjusted:
            plot_s7_uncertainty_by_feature_interval(
                self.model_eff, uncertainty_func,
                self.final_features, self.X_train_imp, adjusted,
                output_structure.get_figure_path(
                    'supplementary',
                    'fig_s7_feature_interval_uncertainty.png'))

    def _plot_figure5_gpr(self):
        """Figure 5: GPR high-performance region (helper)."""
        from visualization.missing_charts import (
            plot_figure5_gpr_high_performance_region)
        from visualization.utils_viz import get_top_features_by_importance
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        scaler = StandardScaler()
        gpr = GaussianProcessRegressor(kernel=RBF(), random_state=42)
        gpr.fit(scaler.fit_transform(self.X_train_imp), self.y_eff_train)
        top = get_top_features_by_importance(
            self.shap_values_eff, self.final_features, 2)
        if len(top) >= 2:
            plot_figure5_gpr_high_performance_region(
                gpr, self.final_features,
                self.X_train_imp, self.y_eff_train, top[0], top[1],
                output_structure.get_figure_path(
                    'supplementary',
                    'fig5_gpr_high_performance_region.png'))

    # ==================================================================
    # Step 11: Save All Results
    # ==================================================================
    def _step11_save_results(self):
        """Step 11: Persist all results to disk."""
        print("\n>>> Step 11: Saving results")

        # Model performance
        results = {
            'efficiency': {
                'r2': float(r2_score(self.y_eff_test, self.y_eff_pred)),
                'mae': float(mean_absolute_error(
                    self.y_eff_test, self.y_eff_pred)),
                'rmse': float(np.sqrt(mean_squared_error(
                    self.y_eff_test, self.y_eff_pred)))
            },
            'reuse': {
                'r2': float(r2_score(self.y_reuse_test, self.y_reuse_pred)),
                'mae': float(mean_absolute_error(
                    self.y_reuse_test, self.y_reuse_pred)),
                'rmse': float(np.sqrt(mean_squared_error(
                    self.y_reuse_test, self.y_reuse_pred)))
            }
        }
        with open(output_structure.get_result_path(
                'metrics', 'model_performance.json'),
                'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Learning curve data
        if hasattr(self, 'learning_curve_data'):
            with open(output_structure.get_result_path(
                    'metrics', 'learning_curve_data.json'),
                    'w', encoding='utf-8') as f:
                json.dump(self.learning_curve_data, f, indent=2)

        # BO history
        if hasattr(self, 'bo_history'):
            with open(output_structure.get_result_path(
                    'metrics', 'bo_optimization_history.json'),
                    'w', encoding='utf-8') as f:
                json.dump(self.bo_history, f, indent=2)

        # Hyperparameter importance
        if hasattr(self, 'hyperparam_importance'):
            with open(output_structure.get_result_path(
                    'metrics', 'hyperparameter_importance.json'),
                    'w', encoding='utf-8') as f:
                json.dump(self.hyperparam_importance, f, indent=2)

        # PRIM peeling history
        if hasattr(self, 'prim_peeling_history'):
            with open(output_structure.get_result_path(
                    'prim', 'prim_peeling_history.json'),
                    'w', encoding='utf-8') as f:
                json.dump(self.prim_peeling_history, f, indent=2)

        # PRIM windows
        prim_json = []
        for w in self.prim_windows:
            prim_json.append({
                'bounds': {k: [float(v[0]), float(v[1])]
                           for k, v in w['bounds'].items()},
                'support': float(w['support']),
                'precision': float(w['precision'])
            })
        with open(output_structure.get_result_path(
                'prim', 'prim_process_windows.json'),
                'w', encoding='utf-8') as f:
            json.dump(prim_json, f, indent=2, ensure_ascii=False)

        # Bootstrap summary
        if hasattr(self, 'bootstrap_results') and self.bootstrap_results:
            bootstrap_summary = {
                'stable_features': list(
                    self.bootstrap_results.get('stable_bounds', {}).keys()),
                'stable_bounds': {
                    k: {'min': float(v['min']), 'max': float(v['max'])}
                    for k, v in self.bootstrap_results.get(
                        'stable_bounds', {}).items()
                },
                'feature_stability': {
                    k: {
                        'frequency': float(v['frequency']),
                        'is_stable': bool(v['is_stable'])
                    }
                    for k, v in self.bootstrap_results.get(
                        'feature_stability', {}).items()
                }
            }
            with open(output_structure.get_result_path(
                    'prim', 'prim_bootstrap_summary.json'), 'w') as f:
                json.dump(bootstrap_summary, f, indent=2, ensure_ascii=False)

        # Final features
        with open(output_structure.get_result_path(
                'features', 'final_features.json'),
                'w', encoding='utf-8') as f:
            json.dump({'features': self.final_features,
                       'count': len(self.final_features)},
                      f, indent=2, ensure_ascii=False)

        print("  [Done] All results saved")


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description='IM-BO-UQ Pipeline for N-C/PMS Catalyst Design')
    parser.add_argument('--data', default='data/sample_data.csv',
                        help='Path to data CSV file')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Bayesian optimization trials')
    args = parser.parse_args()

    pipeline = CompletePipeline(data_path=args.data)
    pipeline.run_all(n_trials=args.trials)


if __name__ == '__main__':
    main()
