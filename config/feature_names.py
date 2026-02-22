"""
Feature Name Mapping — LaTeX symbols for publication-quality figures.

Rules:
- Use LaTeX math symbols (concise, professional)
- Units in parentheses where applicable
- Render in matplotlib via r'$...$'
"""

# ============================================================
# Feature Name → LaTeX Display Mapping (for figures)
# ============================================================
FEATURE_LATEX = {
    # Material properties
    'Surface particle diameter': r'$d_p$ (nm)',
    'aperture': r'$d_{pore}$ (nm)',
    'deepth': r'Depth (nm)',
    'ID/IG': r'$I_D/I_G$',
    'C (Wt%)': r'C (wt%)',
    'N (Wt%)': r'N (wt%)',
    'O (Wt%)': r'O (wt%)',
    'C=O (%)': r'C=O (%)',
    'C=O/C-O（%）': r'C=O/C-O',
    'Fe-O/C-O (%)': r'Fe-O/C-O',
    'graphitic N (%)': r'$N_{graph}$ (%)',
    'N-oxide/ Nitrate N(%)': r'$N_{oxide}$ (%)',

    # Process parameters
    'Catalyst dosage (g/L)': r'$C_{cat}$ (g/L)',
    'Pollutant concentration (mg/L)': r'$C_{poll}$ (mg/L)',
    'Oxidiser type': r'Oxidant',
    'Oxidiser dosage (mM)': r'$C_{ox}$ (mM)',
    'pH': r'pH',
    'reaction time': r'$t$ (min)',

    # Target variables
    'Degradation efficiency (%)': r'$\eta$ (%)',
    'reuse-times': r'Reuse Cycles',
}

# ============================================================
# Feature Name → Short Abbreviation Mapping (for filenames)
# ============================================================
FEATURE_ABBREV = {
    # Material properties
    'Surface particle diameter': 'Particle_Size',
    'aperture': 'Pore_Size',
    'deepth': 'Depth',
    'ID/IG': 'ID_IG',
    'C (Wt%)': 'C_pct',
    'N (Wt%)': 'N_pct',
    'O (Wt%)': 'O_pct',
    'C=O (%)': 'C_O_pct',
    'C=O/C-O（%）': 'CO_ratio',
    'Fe-O/C-O (%)': 'FeO_ratio',
    'graphitic N (%)': 'Graphitic_N',
    'N-oxide/ Nitrate N(%)': 'N_oxide',

    # Process parameters
    'Catalyst dosage (g/L)': 'Cat_Dose',
    'Pollutant concentration (mg/L)': 'Poll_Conc',
    'Oxidiser type': 'Ox_Type',
    'Oxidiser dosage (mM)': 'Ox_Dose',
    'pH': 'pH',
    'reaction time': 'Time',

    # Target variables
    'Degradation efficiency (%)': 'Efficiency',
    'reuse-times': 'Reuse_Cycles',
}

# Reverse mappings
FEATURE_ABBREV_REVERSE = {v: k for k, v in FEATURE_ABBREV.items()}
FEATURE_LATEX_REVERSE = {v: k for k, v in FEATURE_LATEX.items()}


def get_display_name(feature_name: str, use_latex: bool = True) -> str:
    """
    Get the display name of a feature.

    Args:
        feature_name: Original feature name.
        use_latex: If True, return LaTeX format (for figures).
                   If False, return short abbreviation.

    Returns:
        Formatted feature name.
    """
    if use_latex:
        return FEATURE_LATEX.get(feature_name, feature_name)
    else:
        return FEATURE_ABBREV.get(feature_name, feature_name)


def get_filename_safe(feature_name: str) -> str:
    """Get a filename-safe version of the feature name (no special characters)."""
    abbrev = FEATURE_ABBREV.get(feature_name, feature_name)
    safe_name = abbrev.replace('/', '_').replace(' ', '_').replace('%', 'pct')
    safe_name = safe_name.replace('(', '').replace(')', '').replace('$', '')
    return safe_name


def abbreviate_dataframe(df):
    """Rename DataFrame columns to short abbreviations."""
    import pandas as pd
    return df.rename(columns=FEATURE_ABBREV)
