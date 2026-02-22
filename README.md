# IM-BO-UQ: Interval Mining – Bayesian Optimization – Uncertainty Quantification

**From Pointwise Optimization to Industrially Robust Operating Windows for Nitrogen-Doped Carbon Catalysts Activating Peroxymonosulfate**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

## Overview

This repository contains the complete code for the IM-BO-UQ framework. The framework shifts from conventional pointwise ML optimization to discovering **robust, multi-objective operating windows** with statistical coverage guarantees for industrial deployment of N-C/PMS catalysts in wastewater treatment.

### Three-Stage Pipeline

1. **PRIM** (Patient Rule Induction Method) — mines interpretable parameter intervals from XGBoost predictions
2. **Bayesian Optimization** — refines interval boundaries on the Pareto front (efficiency ≥85%, cycles ≥8, Fe leaching ≤0.3 mg/L)
3. **Split Conformal Prediction** — provides distribution-free coverage guarantees (>91% at 90% confidence) with risk stratification

### Key Results

| Metric | Value |
|--------|-------|
| Efficiency R² | 0.966 |
| Stability R² | 0.975 |
| PRIM Lift | 1.52 |
| PI Coverage | >91% |
| Pilot-Scale Error | <1% |
| Dataset | 826 samples, 18 features |

## Project Structure

```
├── main.py                    # Entry point
├── pipeline.py                # Complete 13-step pipeline
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
│
├── data/
│   └── sample_data.csv        # Representative sample (30 rows)
│
├── config/
│   ├── settings.py            # Global configuration
│   ├── feature_names.py       # Feature name → LaTeX mapping
│   └── output_structure.py    # Output directory layout
│
├── utils/
│   └── helpers.py             # Data cleaning utilities
│
├── interpretation/
│   └── prim.py                # PRIM algorithm + Bootstrap stability
│
├── visualization/
│   ├── plots.py               # Core figures (target dist., correlation, PRIM violin)
│   ├── supplementary_plots.py # Supplementary material figures (S1–S11)
│   ├── missing_charts.py      # GPR region, scatter matrix, oxidant distribution
│   ├── plot_3d.py             # 3D surface visualization
│   └── utils_viz.py           # Shared plotting helpers
│
└── outputs/                   # Generated after running (not tracked in git)
    ├── figures/               # All figures organized by analysis stage
    └── results/               # JSON and CSV result files
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# Default configuration (50 BO trials, raw features only)
python main.py

# Custom settings
python pipeline.py --data data/sample_data.csv --trials 100
```

### 3. View results

After completion, all outputs are saved under `outputs/`:
- **Figures**: `outputs/figures/01_data_exploration/` through `08_supplementary/`
- **Results**: `outputs/results/metrics/`, `feature_analysis/`, `prim_analysis/`

## Pipeline Steps

| Step | Description | Outputs |
|------|-------------|---------|
| 1 | Data loading & exploration | Target distributions, feature distributions, Spearman correlation |
| 2 | Feature engineering | VIF multicollinearity control, feature selection |
| 3 | Baseline comparison | 6-model comparison (Ridge, Lasso, SVR, RF, XGBoost, LightGBM), learning curves |
| 4 | Bayesian optimization | Optuna TPE hyperparameter tuning, convergence plots |
| 5 | Test-set evaluation | Predicted vs actual, residual analysis (histogram, Q-Q) |
| 6 | Uncertainty quantification | Split conformal prediction intervals, calibration curve, .632+ bootstrap |
| 7 | PRIM process windows | Single/multi-objective interval mining, bootstrap stability, parameter scan |
| 8 | SHAP interpretability | SHAP summary, bootstrap importance ranking |
| 9 | PDP analysis | Partial dependence plots, 2D interaction heatmaps |
| 10 | Advanced analysis | Pareto front, stacking ensemble, ablation study |
| 11 | Save results | JSON/CSV persistence of all metrics and PRIM windows |
| 12 | Supplementary figures | Supplementary figures (S1–S11)

## Data

The included `data/sample_data.csv` contains a representative 30-row sample from the full 826-sample dataset. Each row represents an N-C catalyst with:

- **11 microstructural features**: particle diameter, pore size, depth, I_D/I_G, elemental composition (C, N, O wt%), functional groups (C=O, C=O/C-O, Fe-O/C-O), nitrogen species (graphitic N, N-oxide)
- **7 process parameters**: catalyst dosage, pollutant concentration, oxidant type, oxidant dosage, pH, reaction time
- **2 targets**: degradation efficiency (%), reuse cycles

> **Note**: The sample dataset is provided for code verification. For full reproducibility, the complete dataset is available upon request from the corresponding author.

## Configuration

All parameters are centralized in `config/settings.py`:

- **DataConfig**: file path, encoding, test split ratio
- **FeatureConfig**: VIF threshold, constructed feature toggles
- **ModelConfig**: CV folds, BO trials, XGBoost search space
- **UncertaintyConfig**: conformal alpha, bootstrap iterations
- **PRIMConfig**: peeling fraction, minimum support, stability thresholds
- **SHAPConfig**: top-K features, bootstrap iterations

## Citation

If you use this code, please cite:

```
Xia, Z.; Ren, T.; Zuo, S. From Pointwise Optimization to Industrially Robust
Operating Windows: An IM-BO-UQ Framework for Nitrogen-Doped Carbon Catalysts
Activating Peroxymonosulfate.
```

## Authors

- **Zichen Xia** — School of Software, Northeastern University, Shenyang
- **Tao Ren** (Corresponding: chinarentao@163.com) — School of Software, Northeastern University
- **Shiyu Zuo** — School of Environmental Engineering, South China University of Technology

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
