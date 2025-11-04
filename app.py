"""Minimal Flask web service for Melbourne House Price project.

Endpoints
- GET /            : Home page with dataset status and links to plots (if present)
- POST/GET /preprocess : Run preprocessing to generate Melbourne_imputed.csv
- GET /healthz     : Liveness probe
- GET /plots/<path> and /advanced_plots/<path> : Serve generated plots

This app is intentionally lightweight for Render deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, render_template_string, request

from preprocess import run_preprocessing


BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
ADV_PLOTS_DIR = BASE_DIR / "advanced_plots"
DATA_FILE = BASE_DIR / "Melbourne_imputed.csv"
RAW_DATA_FILE = BASE_DIR / "Melbourne.csv"


app = Flask(__name__)


HOME_TEMPLATE = """
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<title>House Price Service</title>
		<style>
			body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 2rem auto; max-width: 900px; line-height: 1.5; color: #222; }
			h1 { margin-bottom: 0.25rem; }
			.muted { color: #666; }
			.card { border: 1px solid #eee; border-radius: 10px; padding: 1rem 1.25rem; margin: 1rem 0; }
			code { background: #f7f7f7; padding: 0.1rem 0.3rem; border-radius: 4px; }
			ul { padding-left: 1.2rem; }
			a { color: #0d6efd; text-decoration: none; }
			a:hover { text-decoration: underline; }
			.btn { display: inline-block; padding: .5rem .9rem; border: 1px solid #0d6efd; border-radius: 6px; color: #0d6efd; text-decoration: none; }
			.btn:hover { background: #eef5ff; }
		</style>
	</head>
	<body>
		<h1>🏠 House Price Prediction - Service</h1>
		<p class="muted">Deployed on Render • Flask + Gunicorn</p>

		<div class="card">
			<h3>Dataset</h3>
			<p>Raw CSV: <code>{{ raw_exists }}</code> • Cleaned CSV: <code>{{ cleaned_exists }}</code></p>
			{% if not cleaned_exists and raw_exists %}
			<form method="post" action="/preprocess"><button class="btn" type="submit">Run preprocessing</button></form>
			{% endif %}
		</div>

		<div class="card">
			<h3>Plots</h3>
			{% if plot_links %}
				<ul>
				{% for label, href in plot_links %}
					<li><a href="{{ href }}" target="_blank">{{ label }}</a></li>
				{% endfor %}
				</ul>
			{% else %}
				<p class="muted">No plots were found. Generate them locally using <code>visualization_dav.py</code> and deploy, or upload to the repository.</p>
			{% endif %}
		</div>

		<p class="muted">Health: <a href="/healthz">/healthz</a></p>
	</body>
	</html>
"""


def _discover_plot_links() -> list[tuple[str, str]]:
		links: list[tuple[str, str]] = []
		# Prefer interactive Plotly output if present
		candidates = [
				("Interactive: Area vs Price (plotly)", ADV_PLOTS_DIR / "interactive_area_price.html", "/advanced_plots/interactive_area_price.html"),
				("Price distribution", PLOTS_DIR / "price_distribution.png", "/plots/price_distribution.png"),
				("Area vs price", PLOTS_DIR / "area_vs_price.png", "/plots/area_vs_price.png"),
				("Price by location", PLOTS_DIR / "price_by_location.png", "/plots/price_by_location.png"),
				("Correlation heatmap", PLOTS_DIR / "correlation_heatmap.png", "/plots/correlation_heatmap.png"),
		]
		for label, file_path, href in candidates:
				if file_path.exists():
						links.append((label, href))
		return links


@app.get("/healthz")
def healthz():
		return jsonify(status="ok")


@app.route("/", methods=["GET"])
def index():
		raw_exists = RAW_DATA_FILE.exists()
		cleaned_exists = DATA_FILE.exists()
		return render_template_string(
				HOME_TEMPLATE,
				raw_exists=str(raw_exists),
				cleaned_exists=str(cleaned_exists),
				plot_links=_discover_plot_links(),
		)


@app.route("/preprocess", methods=["POST", "GET"])
def preprocess_endpoint():
		# Allow idempotent GET to make it easy to trigger
		try:
				output = run_preprocessing(str(RAW_DATA_FILE), str(DATA_FILE))
				return jsonify(ok=True, output=output)
		except Exception as e:
				return jsonify(ok=False, error=str(e)), 500


# Static file serving for plots created offline
@app.get("/plots/<path:filename>")
def serve_plots(filename: str):
		return send_from_directory(PLOTS_DIR, filename)


@app.get("/advanced_plots/<path:filename>")
def serve_advanced_plots(filename: str):
		return send_from_directory(ADV_PLOTS_DIR, filename)


if __name__ == "__main__":
		# Local run convenience (Render will use gunicorn)
		port = int(os.environ.get("PORT", "5000"))
		app.run(host="0.0.0.0", port=port, debug=True)

