"""
Utility Functions

Provides data cleaning, type conversion, and missing-value handling.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf and NaN in numeric columns."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    return df_clean


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types (for JSON serialization)."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def fill_missing_values(df: pd.DataFrame, fill_values: Dict[str, float] = None) -> tuple:
    """
    Fill missing values with median (or provided values).

    Returns:
        (filled DataFrame, fill_values dict)
    """
    df_clean = clean_numeric_data(df)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    if fill_values is None:
        fill_values = {}
        for col in numeric_cols:
            median_val = df_clean[col].median()
            fill_values[col] = float(median_val) if not np.isnan(median_val) else 0.0

    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(fill_values.get(col, 0.0))

    return df_clean, fill_values
