"""Preprocessing utilities for the Melbourne housing dataset.

This module provides a function `run_preprocessing()` that is used by the web
service to generate the cleaned dataset. It can also be run directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

# Lightweight Flask app exposed as `app` so that misconfigured start commands
# like `gunicorn preprocess:app` still boot a working service on Render.
from flask import Flask, jsonify, render_template_string, send_from_directory, request


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


# ---------------------------
# Minimal Flask web app here
# ---------------------------
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
ADV_PLOTS_DIR = BASE_DIR / "advanced_plots"
DATA_FILE = BASE_DIR / "Melbourne_imputed.csv"
RAW_DATA_FILE = BASE_DIR / "Melbourne.csv"

app = Flask(__name__)

HOME_TEMPLATE = """
<!doctype html>
<html lang='en'>
    <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>House Price Service</title>
        <style>
            body { font-family: system-ui, Arial; margin: 2rem auto; max-width: 900px; line-height: 1.5; color:#222; }
            .card { border:1px solid #eee; border-radius:10px; padding:1rem 1.25rem; margin:1rem 0; }
            .btn { display:inline-block; padding:.5rem .9rem; border:1px solid #0d6efd; border-radius:6px; color:#0d6efd; text-decoration:none; }
            .btn:hover { background:#eef5ff; }
            code { background:#f7f7f7; padding:0 .25rem; border-radius:4px; }
            ul { padding-left:1.2rem; }
        </style>
    </head>
    <body>
        <h1>🏠 House Price Prediction</h1>
        <div class='card'>
            <h3>Dataset</h3>
            <p>Raw: <code>{{ raw_exists }}</code> • Cleaned: <code>{{ cleaned_exists }}</code></p>
            {% if not cleaned_exists and raw_exists %}
            <form method='post' action='/preprocess'><button class='btn' type='submit'>Run preprocessing</button></form>
            {% endif %}
        </div>
            <div class='card'>
                <h3>Plots</h3>
                {% if plot_links %}
                    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;">
                    {% for label, href in plot_links %}
                        <div style="border:1px solid #eee;border-radius:8px;padding:8px;">
                            <div style="font-size:14px;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{{ label }}</div>
                            {% if href.endswith('.png') %}
                                <a href='{{ href }}' target='_blank'><img src='{{ href }}' alt='{{ label }}' style='width:100%;height:140px;object-fit:cover;border-radius:6px;'/></a>
                            {% else %}
                                <a href='{{ href }}' target='_blank' class='btn'>Open</a>
                            {% endif %}
                        </div>
                    {% endfor %}
                    </div>
                {% else %}
                    <p>No plots found. Generate locally to the <code>plots/</code> folder.</p>
                {% endif %}
                <p>Health: <a href='/healthz'>/healthz</a></p>
            </div>
    </body>
</html>
"""


def _label_from_filename(name: str) -> str:
    base = Path(name).stem.replace('_', ' ').strip()
    return base[:1].upper() + base[1:]


def _discover_plot_links() -> List[Tuple[str, str]]:
    links: List[Tuple[str, str]] = []

    # All PNGs in plots/
    if PLOTS_DIR.exists():
        for f in sorted(PLOTS_DIR.glob('*.png')):
            links.append((_label_from_filename(f.name), f"/plots/{f.name}"))

    # PNG and HTML in advanced_plots/
    if ADV_PLOTS_DIR.exists():
        for ext in ("*.png", "*.html"):
            for f in sorted(ADV_PLOTS_DIR.glob(ext)):
                links.append((_label_from_filename(f.name), f"/advanced_plots/{f.name}"))

    # Prefer to show interactive plot first if present
    links.sort(key=lambda x: (0 if x[1].endswith('interactive_area_price.html') else 1, x[0]))
    return links


@app.get("/healthz")
def healthz():
        return jsonify(status="ok")


@app.route("/")
def index():
        return render_template_string(
                HOME_TEMPLATE,
                raw_exists=str(RAW_DATA_FILE.exists()),
                cleaned_exists=str(DATA_FILE.exists()),
                plot_links=_discover_plot_links(),
        )


@app.route("/preprocess", methods=["POST", "GET"])
def preprocess_endpoint():
        try:
                output = run_preprocessing(str(RAW_DATA_FILE), str(DATA_FILE))
                return jsonify(ok=True, output=output)
        except Exception as e:
                return jsonify(ok=False, error=str(e)), 500


@app.get("/plots/<path:filename>")
def serve_plots(filename: str):
        return send_from_directory(PLOTS_DIR, filename)


@app.get("/advanced_plots/<path:filename>")
def serve_advanced_plots(filename: str):
        return send_from_directory(ADV_PLOTS_DIR, filename)


if __name__ == "__main__":
        # Allow running this module directly as a server too
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
