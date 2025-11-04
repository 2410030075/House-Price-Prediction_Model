"""Preprocessing utilities for the Melbourne housing dataset.

This module provides a function `run_preprocessing()` that is used by the web
service to generate the cleaned dataset. It can also be run directly.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def run_preprocessing(input_csv: str = "Melbourne.csv", output_csv: str = "Melbourne_imputed.csv") -> str:
    """Load raw dataset, impute missing values, cap outliers and save.

    Args:
        input_csv: Path to the raw input CSV (must exist).
        output_csv: Path where the cleaned CSV will be saved.

    Returns:
        The path to the generated cleaned CSV file.
    """
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # Replace None with np.nan for consistency
    df = df.replace({None: np.nan})

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.columns.difference(numeric_cols)

    # Impute numeric columns with median
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Impute categorical columns with mode (most frequent)
    if len(categorical_cols) > 0:
        modes = df[categorical_cols].mode(dropna=True)
        if not modes.empty:
            df[categorical_cols] = df[categorical_cols].fillna(modes.iloc[0])

    # Cap outliers to median using IQR rule (numeric columns only)
    if len(numeric_cols) > 0:
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        for col in numeric_cols:
            lower = Q1[col] - 1.5 * IQR[col]
            upper = Q3[col] + 1.5 * IQR[col]
            med = df[col].median()
            mask = (df[col] < lower) | (df[col] > upper)
            if mask.any():
                df.loc[mask, col] = med

    # Save the cleaned dataset
    df.to_csv(output_csv, index=False)
    return str(output_csv)


if __name__ == "__main__":
    out = run_preprocessing()
    print(f"Saved cleaned dataset to {out}")
