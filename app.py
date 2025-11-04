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
				<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;">
				{% for label, href in plot_links %}
					<div style="border:1px solid #eee;border-radius:8px;padding:8px;">
						<div style="font-size:14px;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{{ label }}</div>
						{% if href.endswith('.png') %}
							<a href="{{ href }}" target="_blank"><img src="{{ href }}" alt="{{ label }}" style="width:100%;height:140px;object-fit:cover;border-radius:6px;"/></a>
						{% else %}
							<a href="{{ href }}" target="_blank" class="btn">Open</a>
						{% endif %}
					</div>
				{% endfor %}
				</div>
			{% else %}
				<p class="muted">No plots were found. Generate them locally using <code>visualization_dav.py</code> and deploy, or upload to the repository.</p>
			{% endif %}
		</div>

		<p class="muted">Health: <a href="/healthz">/healthz</a></p>
	</body>
	</html>
"""


def _label_from_filename(name: str) -> str:
		base = Path(name).stem.replace('_', ' ').strip()
		return base[:1].upper() + base[1:]


def _discover_plot_links() -> list[tuple[str, str]]:
		links: list[tuple[str, str]] = []
		# Collect all PNGs from plots/
		if PLOTS_DIR.exists():
				for f in sorted(PLOTS_DIR.glob('*.png')):
						links.append((_label_from_filename(f.name), f"/plots/{f.name}"))
		# Collect PNG/HTML from advanced_plots/
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
	import os
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port)

