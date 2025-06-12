"""
modern_dashboard.py · v2.1
==========================

Dark-theme Dash dashboard for **RealSyntheticAnalyzer v1.0**.
Features
--------
• Column selector grouped by inferred roles (identifier, temporal, …)  
• Dtype-aware plot selector  
• Metric cards with adaptive labels + values  
• Correlation heat-maps (real & synthetic)  
• Full evaluation-log tab (downloadable CSV)  
• Summary-stats table (downloadable CSV)  
• PNG export for current chart  
• IsolationForest anomaly panel  
• “Refresh” button reloads data/roles on the fly

Run:
    python modern_dashboard.py
and open http://127.0.0.1:8050
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from real_synth_analyzer import AnalyzerConfig, RealSyntheticAnalyzer

# ───────────────────────── configuration ──────────────────────────────
REAL_PATH = Path("data/real_data.csv")
SYNTH_PATH = Path("data/synthetic_data.csv")
THEME = dbc.themes.CYBORG  # dark Bootstrap theme

AN_CFG = AnalyzerConfig()
_an = RealSyntheticAnalyzer(REAL_PATH, SYNTH_PATH, config=AN_CFG)

# ───────────── role → columns mapping & helpers ───────────────────────
def _build_role_cols():
    role_cols: dict[str, List[str]] = defaultdict(list)
    for col in _an.real.columns.intersection(_an.synth.columns):
        role_cols[_an.column_role(col)].append(col)
    return role_cols


ROLE_COLS = _build_role_cols()
ROLE_LABELS = {
    "identifier": "Identifier",
    "demographic": "Demographic",
    "temporal": "Temporal",
    "behavioral": "Behavioral",
    "predictive": "Predictive",
    "target": "Target",
}
BADGE_COLOR = {
    "identifier": "secondary",
    "demographic": "info",
    "temporal": "warning",
    "behavioral": "success",
    "predictive": "primary",
    "target": "danger",
}


def _column_options():
    """Build dropdown options with pseudo-optgroup headers."""
    opts = []
    for role, cols in ROLE_COLS.items():
        opts.append(
            {
                "label": f"— {ROLE_LABELS.get(role, role.capitalize())} —",
                "value": f"hdr_{role}",
                "disabled": True,
            }
        )
        opts.extend({"label": c, "value": c} for c in sorted(cols))
    return opts

# ────────── helper for PK / FK status ────────────────────────────────
def pkfk_status(col: str) -> str:
    """
    Return 'PK' if column is a primary-key candidate,
    'FK → <target>' if it points to a PK column,
    otherwise '–'.
    """
    df_keys = _an.identify_keys()
    row = df_keys.loc[df_keys["column"] == col].squeeze()
    if row["is_pk"]:
        return "PK"
    if pd.notna(row["fk_to"]):
        return f"FK → {row['fk_to']}"
    return "–"

def metric_card_pkfk(value: str) -> dbc.Col:
    return metric_card("Key type", value)


# ───────────── Dash helper components ─────────────────────────────────
def metric_card(title: str, value: str | float = "-") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            [dbc.CardHeader(title), dbc.CardBody(html.H5(value))],
            className="text-center shadow-sm",
        ),
        lg=2,
        md=4,
        sm=6,
    )


def role_badge(role: str) -> html.Span:
    return html.Span(
        ROLE_LABELS.get(role, role.capitalize()),
        className=f"badge bg-{BADGE_COLOR.get(role,'light')} ms-2",
    )


# ───────────── Dash app & layout ──────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[THEME])
server = app.server

app.layout = dbc.Container(
    [
        # Header ───────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.H3("📊 Real vs Synthetic Dashboard")),
                dbc.Col(
                    dbc.Button(
                        "Refresh",
                        id="btn-refresh",
                        color="primary",
                        className="ms-auto",
                    ),
                    width="auto",
                ),
            ],
            className="my-3 align-items-center",
        ),
        # Hidden stores / downloads ────────────────────────────────────
        dcc.Store(id="store-anoms"),
        dcc.Download(id="dl-chart"),
        dcc.Download(id="dl-summary"),
        dcc.Download(id="dl-log"),
        # Controls ─────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Column"),
                        dcc.Dropdown(id="dd-col", options=_column_options()),
                    ],
                    lg=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Visualisation"),
                        dcc.Dropdown(id="dd-plot"),
                    ],
                    lg=4,
                ),
                dbc.Col(
                    dbc.Checklist(
                        options=[
                            {
                                "label": "Highlight outliers (≥3σ)",
                                "value": "out",
                            }
                        ],
                        id="chk-outliers",
                        value=[],
                        inline=True,
                    ),
                    lg=4,
                    className="pt-lg-4",
                ),
            ],
            className="mb-3",
        ),
        # Heading + metrics ────────────────────────────────────────────
        dbc.Row(dbc.Col(html.H4(id="col-heading"))),
        dbc.Row(id="metric-row", className="gy-3 mb-3"),
        # Main graph ───────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="graph-main", config={"displaylogo": False}), width=12
            ),
            className="mb-4",
        ),
        # Action buttons ───────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Download Chart",
                        id="btn-dl-chart",
                        outline=True,
                        color="secondary",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Download Summary",
                        id="btn-dl-summary",
                        outline=True,
                        color="secondary",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Download Metrics",
                        id="btn-dl-log",
                        outline=True,
                        color="secondary",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("Detect Anomalies", id="btn-anom", color="warning"),
                    width="auto",
                ),
            ],
            className="gy-2 mb-4",
        ),
        # Tabs ─────────────────────────────────────────────────────────
        dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Graph(id="corr-real", config={"displaylogo": False}),
                    label="Real Correlation",
                ),
                dbc.Tab(
                    dcc.Graph(id="corr-synth", config={"displaylogo": False}),
                    label="Synthetic Correlation",
                ),
                dbc.Tab(html.Div(id="tbl-log"), label="Metrics Table"),
            ],
            className="mb-4",
        ),
        # Summary & anomalies ──────────────────────────────────────────
        dbc.Row(dbc.Col(html.Div(id="tbl-summary"))),
        dbc.Row(dbc.Col(html.Div(id="anom-output"))),
        html.Hr(),
        html.Footer("© 2025 – Modern Dashboard", className="text-muted text-end my-2"),
    ],
    fluid=True,
)

# ───────────────────────── callbacks ──────────────────────────────────
# Refresh button ─ reload data & roles
@app.callback(
    Output("dd-col", "options"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(_):
    global _an, ROLE_COLS
    _an = RealSyntheticAnalyzer(REAL_PATH, SYNTH_PATH, config=AN_CFG)
    ROLE_COLS = _build_role_cols()
    return _column_options()


# Plot-mode dropdown
@app.callback(
    Output("dd-plot", "options"),
    Output("dd-plot", "value"),
    Input("dd-col", "value"),
)
def set_plot_modes(col):
    if not col or col.startswith("hdr_"):
        return [], None
    modes = _an.suggest_plot_modes(col)
    return [{"label": m, "value": m} for m in modes], modes[0]


# Main dashboard update
@app.callback(
    Output("col-heading", "children"),
    Output("metric-row", "children"),
    Output("graph-main", "figure"),
    Output("corr-real", "figure"),
    Output("corr-synth", "figure"),
    Output("tbl-summary", "children"),
    Output("tbl-log", "children"),
    Input("dd-col", "value"),
    Input("dd-plot", "value"),
    Input("chk-outliers", "value"),
)
def update_dashboard(col, mode, out_opts):
    if not col or col.startswith("hdr_") or not mode:
        empty = go.Figure()
        return "", [], empty, empty, empty, None, None

    role = _an.column_role(col)
    heading = [html.Span(col), role_badge(role)]

    fig_main = _an.make_figure(col, mode=mode, highlight="out" in out_opts)

    names = _an.metric_names(col)
    vals = _an.metrics(col)

    key_tag = pkfk_status(col)          # NEW ────────────────

    cards = [metric_card_pkfk(key_tag)] + [
        metric_card(n, f"{v:.4f}" if pd.notna(v) else "NA")
        for n, v in zip(names, vals)
    ]

    fig_r, fig_s = _an.correlation_figs()

    summary_df = (
        _an.summary().reset_index().rename(columns={"index": "Statistic"})
    )
    tbl_summary = dbc.Table.from_dataframe(
        summary_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm",
    )

    log_df = _an.evaluation_log()
    tbl_log = dbc.Table.from_dataframe(
        log_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-sm",
    )

    return heading, dbc.Row(cards, className="gy-3"), fig_main, fig_r, fig_s, tbl_summary, tbl_log


# Anomaly detection
@app.callback(
    Output("store-anoms", "data"),
    Input("btn-anom", "n_clicks"),
    prevent_initial_call=True,
)
def detect_anoms(_):
    return _an.detect_anomalies(max_rows=200).to_dict("records")


@app.callback(Output("anom-output", "children"), Input("store-anoms", "data"))
def show_anoms(rows):
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    return dbc.Alert(
        [
            html.H6(
                f"IsolationForest detected {len(df)} anomalies",
                className="alert-heading",
            ),
            dbc.Table.from_dataframe(
                df.head(20),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="table-sm",
            ),
        ],
        color="danger",
        className="mt-3",
    )


# Download: PNG chart
@app.callback(
    Output("dl-chart", "data"),
    Input("btn-dl-chart", "n_clicks"),
    State("graph-main", "figure"),
    prevent_initial_call=True,
)
def download_chart(_, fig_dict):
    if not fig_dict:
        return no_update
    return dcc.send_bytes(_an.to_png(go.Figure(fig_dict), scale=3), "chart.png")


# Download: summary CSV
@app.callback(
    Output("dl-summary", "data"),
    Input("btn-dl-summary", "n_clicks"),
    State("tbl-summary", "children"),
    prevent_initial_call=True,
)
def download_summary(_, tbl):
    if tbl is None:
        return no_update
    df = pd.read_html(str(tbl))[0]
    return dcc.send_data_frame(df.to_csv, "summary.csv", index=False)


# Download: evaluation-log CSV
@app.callback(
    Output("dl-log", "data"),
    Input("btn-dl-log", "n_clicks"),
    prevent_initial_call=True,
)
def download_log(_):
    df = _an.evaluation_log()
    return dcc.send_data_frame(df.to_csv, "metrics_log.csv", index=False)


# ───────────────────────── run ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, dev_tools_silence_routes_logging=True)