"""Command-line client for the Shipment Delay Predictor API.

Usage examples (run from repo root):
  python scripts/predict_cli.py predict --weight 4200 --discount 45 --mode road \
      --warehouse F --calls 4 --rating 2 --prior 3 --importance high \
      --cost 180 --gender M --threshold 0.4

  python scripts/predict_cli.py predict --file shipments.json
  python scripts/predict_cli.py explain --file shipment.json
  python scripts/predict_cli.py worst-case
  python scripts/predict_cli.py best-case
  python scripts/predict_cli.py sensitivity --file shipment.json
  python scripts/predict_cli.py predict --file s.json --url https://your-app.onrender.com

If --url is omitted, falls back to $API_HOST or http://localhost:8000.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
DEFAULT_URL = os.getenv("API_HOST", "http://localhost:8000")


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _post(base: str, path: str, **kwargs) -> Any:
    try:
        r = requests.post(_api_url(base, path), timeout=30, **kwargs)
    except requests.RequestException as e:
        console.print(f"[red]Network error:[/red] {e}")
        sys.exit(2)
    if r.status_code >= 400:
        console.print(f"[red]HTTP {r.status_code}:[/red] {r.text}")
        sys.exit(1)
    return r.json()


def _build_shipment_from_args(a) -> dict:
    """Convert CLI flags into the JSON shape the API expects."""
    return {
        "warehouse_block": a.warehouse,
        "mode_of_shipment": a.mode,
        "customer_care_calls": a.calls,
        "customer_rating": a.rating,
        "cost_of_product": a.cost,
        "prior_purchases": a.prior,
        "product_importance": a.importance,
        "gender": a.gender,
        "discount_offered": a.discount,
        "weight_in_gms": a.weight,
        "threshold": a.threshold,
    }


def _load_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in {path}:[/red] {e}")
        sys.exit(1)


# ---- pretty printers --------------------------------------------------------

def _render_prediction(d: dict, *, title: str = "Prediction") -> None:
    delayed = d["delayed"]
    icon = "[red bold]⚠ DELAYED[/red bold]" if delayed else "[green bold]✓ ON TIME[/green bold]"
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Status", icon)
    table.add_row("Probability", f"[bold]{d['probability']*100:.1f}%[/bold]")
    table.add_row("Confidence", str(d["confidence"]).upper())
    table.add_row("Threshold", f"{d['threshold_used']:.2f}")
    if d.get("note"):
        table.add_row("Note", Text(d["note"], style="italic dim"))
    console.print(Panel(table, title=title, border_style="red" if delayed else "green"))


def _render_batch(items: list[dict]) -> None:
    t = Table(title=f"Batch prediction · {len(items)} shipments", title_style="bold")
    t.add_column("#", justify="right")
    t.add_column("Status")
    t.add_column("Probability", justify="right")
    t.add_column("Confidence")
    t.add_column("Threshold", justify="right")
    for i, d in enumerate(items, 1):
        status = "[red]⚠ DELAYED[/red]" if d["delayed"] else "[green]✓ ON TIME[/green]"
        t.add_row(str(i), status, f"{d['probability']*100:.1f}%",
                  str(d["confidence"]).upper(), f"{d['threshold_used']:.2f}")
    console.print(t)


def _render_explanation(d: dict) -> None:
    _render_prediction(d, title="Prediction")
    t = Table(title="Top SHAP factors", title_style="bold")
    t.add_column("Feature")
    t.add_column("Direction")
    t.add_column("Magnitude", justify="right")
    for f in d.get("top_factors", []):
        arrow = "[red]↑ raises risk[/red]" if f["direction"] == "increases_delay_risk" else "[green]↓ lowers risk[/green]"
        t.add_row(f["feature"], arrow, f"{f['magnitude']:.3f}")
    console.print(t)
    console.print(Panel(d.get("explanation", ""), title="Narrative", border_style="cyan"))
    actions = d.get("suggested_actions", [])
    if actions:
        a = Table.grid(padding=(0, 2))
        a.add_column(style="bold cyan", justify="right")
        a.add_column()
        for i, act in enumerate(actions, 1):
            a.add_row(f"{i}.", act)
        console.print(Panel(a, title="Suggested actions", border_style="cyan"))


def _render_sensitivity(d: dict) -> None:
    console.print(f"[bold]Base probability:[/bold] {d['base_probability']*100:.1f}%\n")
    items = [
        (feat, r["max_prob"] - r["min_prob"], r["min_prob"], r["max_prob"], r["most_impactful_value"])
        for feat, r in d["feature_ranges"].items()
    ]
    items.sort(key=lambda x: -x[1])
    t = Table(title="Sensitivity — per-feature probability range", title_style="bold")
    t.add_column("Feature")
    t.add_column("Range", justify="right")
    t.add_column("Min", justify="right")
    t.add_column("Max", justify="right")
    t.add_column("Max-impact value", justify="right")
    for feat, r, mn, mx, v in items:
        t.add_row(feat, f"±{r*100:.1f}%", f"{mn*100:.1f}%", f"{mx*100:.1f}%", f"{v:.1f}")
    console.print(t)


# ---- subcommand handlers ----------------------------------------------------

def cmd_predict(a) -> None:
    if a.file:
        payload = _load_file(a.file)
        if isinstance(payload, list):
            with open(a.file, "rb") as fh:
                resp = _post(a.url, "/predict", files={"file": (Path(a.file).name, fh, "application/json")})
            _render_batch(resp)
        else:
            d = _post(a.url, "/predict", json=payload)
            _render_prediction(d)
    else:
        d = _post(a.url, "/predict", json=_build_shipment_from_args(a))
        _render_prediction(d)


def cmd_explain(a) -> None:
    payload = _load_file(a.file) if a.file else _build_shipment_from_args(a)
    if isinstance(payload, list):
        console.print("[yellow]explain accepts a single shipment; got a list — using the first row.[/yellow]")
        payload = payload[0]
    d = _post(a.url, "/explain", json=payload)
    _render_explanation(d)


def cmd_worst_case(a) -> None:
    _render_prediction(_post(a.url, "/worst-case"), title="Worst case")


def cmd_best_case(a) -> None:
    _render_prediction(_post(a.url, "/best-case"), title="Best case")


def cmd_sensitivity(a) -> None:
    payload = _load_file(a.file) if a.file else _build_shipment_from_args(a)
    if isinstance(payload, list):
        payload = payload[0]
    _render_sensitivity(_post(a.url, "/sensitivity", json=payload))


# ---- argument parsing -------------------------------------------------------

def _add_inline_shipment_flags(p: argparse.ArgumentParser, required: bool = False) -> None:
    """Add the 10 shipment fields as flags. Marked optional so --file works."""
    p.add_argument("--warehouse", default=None, help="A/B/C/D/F")
    p.add_argument("--mode", default=None, help="Ship/Flight/Road")
    p.add_argument("--calls", type=int, default=None, help="customer_care_calls")
    p.add_argument("--rating", type=int, default=None, help="customer_rating 1-5")
    p.add_argument("--cost", type=float, default=None, help="cost_of_product")
    p.add_argument("--prior", type=int, default=None, help="prior_purchases")
    p.add_argument("--importance", default=None, help="Low/Medium/High")
    p.add_argument("--gender", default=None, help="M/F")
    p.add_argument("--discount", type=float, default=None, help="discount_offered (%)")
    p.add_argument("--weight", type=float, default=None, help="weight_in_gms")
    p.add_argument("--threshold", type=float, default=None, help="decision threshold 0-1")
    p.add_argument("--file", default=None, help="JSON file with shipment(s); overrides inline flags")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI for the Shipment Delay Predictor API")
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"API base URL (default: {DEFAULT_URL})")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in [("predict", cmd_predict), ("explain", cmd_explain),
                     ("sensitivity", cmd_sensitivity)]:
        sp = sub.add_parser(name)
        _add_inline_shipment_flags(sp)
        sp.set_defaults(func=fn)

    for name, fn in [("worst-case", cmd_worst_case), ("best-case", cmd_best_case)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)

    return p


def main() -> None:
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
