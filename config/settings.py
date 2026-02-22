"""
IM-BO-UQ Project Configuration

Centralized configuration for all pipeline parameters.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# ============================================================
# Global Constants
# ============================================================

RANDOM_SEED = 42

# Target variable names
TARGET_EFFICIENCY = 'Degradation efficiency (%)'
TARGET_REUSE = 'reuse-times'

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'sample_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
RESULTS_DIR = OUTPUT_DIR / 'results'


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    data_path: str = str(DATA_PATH)
    encoding: str = 'utf-8'
    header_row: int = 0  # 0-indexed header row
    test_size: float = 0.2
    random_seed: int = RANDOM_SEED

    # Material features (for feature engineering)
    material_features: List[str] = field(default_factory=lambda: [
        'Surface particle diameter', 'aperture', 'deepth', 'ID/IG', 'C (Wt%)',
        'N (Wt%)', 'O (Wt%)', 'C=O (%)', 'C=O/C-O（%）', 'Fe-O/C-O (%)',
        'graphitic N (%)', 'N-oxide/ Nitrate N(%)'
    ])

    # Mandatory features protected from VIF removal
    mandatory_features: List[str] = field(default_factory=lambda: [
        'Catalyst dosage (g/L)',
        'reaction time',
        'Oxidiser dosage (mM)',
        'Pollutant concentration (mg/L)',
        'ID/IG',
        'C=O (%)',
        'C=O/C-O（%）',
        'graphitic N (%)',
        'Fe-O/C-O (%)'
    ])


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    vif_threshold: float = 10.0
    max_vif_iterations: int = 30
    epsilon_min: float = 1e-6

    # Constructed features toggle (default: off, use raw features only)
    use_constructed_features: bool = False

    # Individual toggles (only active when use_constructed_features=True)
    use_ratio_features: bool = True         # Element ratio features
    use_structure_features: bool = True     # Structural features
    use_functional_features: bool = True    # Functional group features
    use_interaction_features: bool = True   # Process interaction features
    use_nonlinear_features: bool = False    # Nonlinear transforms (default: off)


@dataclass
class ModelConfig:
    """Model training configuration."""
    cv_n_splits: int = 5
    cv_n_repeats: int = 3
    optuna_n_trials: int = 50

    # XGBoost hyperparameter search space
    xgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (100, 600),
        'max_depth': (2, 8),
        'learning_rate': (0.01, 0.2),
        'min_child_weight': (1, 20),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 5),
        'reg_alpha': (1e-3, 10),
        'reg_lambda': (1e-2, 50)
    })


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""
    conformal_alpha: float = 0.1  # 90% confidence interval
    conformal_calibration_size: float = 0.25
    calibration_alphas: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2, 0.1, 0.05])
    bootstrap_n_iterations: int = 100


@dataclass
class PRIMConfig:
    """PRIM process window configuration."""
    alpha: float = 0.05           # Peeling fraction per iteration
    min_support: float = 0.05     # Minimum support
    efficiency_threshold_percentile: int = 85
    reuse_threshold_percentile: int = 80

    # Bootstrap stability analysis
    bootstrap_n_iterations: int = 50
    bootstrap_sample_ratio: float = 0.8
    stability_threshold: float = 0.85
    stability_alpha: float = 0.05  # FDR significance level


@dataclass
class SHAPConfig:
    """SHAP interpretability configuration."""
    top_k_features: int = 10
    use_adaptive_threshold: bool = True
    adaptive_std_multiplier: float = 1.0
    quantile_threshold: float = 0.7
    bootstrap_n_iterations: int = 50
    n_samples: int = 100


@dataclass
class ConstraintConfig:
    """Industrial constraint configuration."""
    fe_leaching_enabled: bool = True
    fe_leaching_column: str = 'Fe_leaching'
    fe_leaching_max: float = 0.3       # mg/L (GB 25467-2010)
    min_degradation_efficiency: float = 85.0  # %
    min_cycle_number: int = 12


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    dpi: int = 300
    figure_format: str = 'png'
    style: str = 'seaborn-v0_8-whitegrid'
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': 'steelblue',
        'secondary': 'coral',
        'success': 'lightgreen',
        'warning': 'orange',
        'danger': 'lightcoral'
    })


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    prim: PRIMConfig = field(default_factory=PRIMConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    random_seed: int = RANDOM_SEED

    def __post_init__(self):
        """Ensure output directories exist."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()
