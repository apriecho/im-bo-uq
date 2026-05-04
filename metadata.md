# Metadata

This document describes the data files, variables, preprocessing steps, and
reproducibility scope for the IM-BO-UQ repository.

## Repository Data Availability

The repository currently includes a representative sample dataset:

| File | Description | Rows | Columns | Format |
| --- | --- | ---: | ---: | --- |
| `data/sample_data.csv` | Representative N-C/PMS catalyst records used for code verification and demonstration | 30 | 20 | UTF-8 CSV |

The full study dataset contains 826 samples. Because the complete literature
collection includes curated records from published studies and may require
additional rights and provenance checks before redistribution, the full dataset
is not included directly in this public repository. It is available from the
corresponding author upon reasonable request. The included sample data preserves
the same column schema as the modeling pipeline.

## Data Source

The dataset was curated from literature reports on nitrogen-doped carbon
catalysts activating peroxymonosulfate for pollutant degradation. Each row
corresponds to one experimental catalyst-process record. The columns combine
catalyst microstructural descriptors, reaction/process conditions, and measured
performance targets.

## File Encoding and Structure

- File path: `data/sample_data.csv`
- Encoding: UTF-8
- Header row: first row
- Delimiter: comma
- Missing values: empty cells are treated as missing values by `pandas`
- Default loader: `pandas.read_csv(..., encoding="utf-8")`

The default path is configured in `config/settings.py` as
`data/sample_data.csv`.

## Variable Dictionary

### Catalyst Microstructural Features

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `Surface particle diameter` | Numeric | nm | Catalyst surface particle diameter. |
| `aperture` | Numeric | nm | Pore aperture or pore size. |
| `deepth` | Numeric | nm | Pore depth descriptor. The column name is kept as originally used by the code and dataset schema. |
| `ID/IG` | Numeric | unitless | Raman defect-to-graphitic band intensity ratio. |
| `C (Wt%)` | Numeric | wt% | Carbon elemental fraction. |
| `N (Wt%)` | Numeric | wt% | Nitrogen elemental fraction. |
| `O (Wt%)` | Numeric | wt% | Oxygen elemental fraction. |
| `C=O (%)` | Numeric | % | Carbonyl functional group percentage. |
| `C=O/C-O（%）` | Numeric | ratio or % | Relative abundance of C=O to C-O oxygen-containing groups. |
| `Fe-O/C-O (%)` | Numeric | ratio or % | Relative Fe-O/C-O functional group descriptor. |
| `graphitic N (%)` | Numeric | % | Graphitic nitrogen percentage. |
| `N-oxide/ Nitrate N(%)` | Numeric | % | N-oxide or nitrate nitrogen percentage. |

### Reaction and Process Features

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `Catalyst dosage (g/L)` | Numeric | g/L | Catalyst concentration in the reaction system. |
| `Pollutant concentration (mg/L)` | Numeric | mg/L | Initial pollutant concentration. |
| `Oxidiser type` | Numeric/categorical code | coded category | Oxidant type encoded as a numeric category in the current CSV schema. |
| `Oxidiser dosage (mM)` | Numeric | mM | Oxidant concentration. |
| `pH` | Numeric | pH unit | Initial reaction pH. |
| `reaction time` | Numeric | min | Reaction time. |

### Target Variables

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `Degradation efficiency (%)` | Numeric | % | Pollutant degradation efficiency. This is the primary regression target for efficiency modeling. |
| `reuse-times` | Numeric/integer | cycles | Catalyst reuse or stability cycles. This is the target for stability modeling. |

## Raw and Processed Data

The current public repository provides the sample data file directly under
`data/`. The modeling pipeline performs preprocessing in memory and writes
generated outputs under `outputs/` when executed.

The pipeline-generated outputs are not tracked by Git by default. They include:

- exploratory data figures
- model comparison metrics
- Bayesian optimization logs and plots
- test-set evaluation figures
- conformal uncertainty quantification outputs
- PRIM process-window analysis results
- SHAP and PDP interpretability figures
- supplementary figures and result tables

These outputs are created by running:

```bash
python main.py
```

or:

```bash
python pipeline.py --data data/sample_data.csv --trials 100
```

## Preprocessing and Cleaning

The pipeline preprocessing is implemented in `pipeline.py` and
`utils/helpers.py`. The main operations are:

- strip whitespace from column names
- remove duplicate rows
- split data into training and test subsets with `test_size=0.2`
- use `random_seed=42` for reproducibility
- retain raw numeric features by default
- optionally construct engineered features when enabled in `FeatureConfig`
- replace infinite values with missing values
- impute numeric missing values using training-set medians
- optionally apply VIF-based feature filtering when constructed features are enabled

By default, constructed features are disabled and the pipeline uses the raw
feature schema listed above.

## Modeling and Analysis Scope

The repository contains code for the full IM-BO-UQ workflow:

1. data loading and exploratory analysis
2. feature engineering and VIF control
3. baseline model comparison
4. Bayesian optimization using Optuna
5. XGBoost-based efficiency and reuse-cycle prediction
6. split conformal prediction for uncertainty quantification
7. PRIM process-window extraction and bootstrap stability analysis
8. SHAP and partial dependence interpretation
9. supplementary figure generation
10. result export to JSON and CSV files

The sample dataset is intended to verify that the code runs and that the
repository structure is reproducible. Numerical results reported in the
manuscript are based on the full 826-sample dataset.

## Reproducibility Notes

- Python version: Python 3.8 or later
- Dependencies: listed in `requirements.txt`
- Main entry point: `main.py`
- Full pipeline implementation: `pipeline.py`
- Random seed: `42`
- Default data file: `data/sample_data.csv`
- Default outputs directory: `outputs/`

To reproduce the repository demonstration:

```bash
pip install -r requirements.txt
python main.py
```

For manuscript-level reproduction, replace `data/sample_data.csv` with the full
826-sample dataset using the same column schema, then run the same pipeline
entry point.

## Data Use and Citation

If this repository or dataset schema is used, please cite the associated
manuscript:

```text
Xia, Z.; Ren, T.; Zuo, S. From Pointwise Optimization to Industrially Robust
Operating Windows: An IM-BO-UQ Framework for Nitrogen-Doped Carbon Catalysts
Activating Peroxymonosulfate.
```

For access to the complete dataset, contact the corresponding author listed in
`README.md`.
