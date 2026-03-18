"""
Dash component: Spectral Entropy Viewer
Fetches a real MS/MS spectrum from GNPS2 by USI and renders
the entropy interactive plot + per-peak breakdown table.

Install dependencies:
    pip install dash dash-bootstrap-components plotly requests

Run:
    python dash_spectral_entropy.py
    then open http://127.0.0.1:8050
"""

import math
from typing import Optional

import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ── reuse the core functions from spectral_entropy.py ────────────────────────
# (inline them here so this file is self-contained)

GNPS_USI_BASE = "https://metabolomics-usi.gnps2.org"

EXAMPLE_USIS = [
    {"label": "Quercetin 3-glucoside",    "value": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622"},
    {"label": "Caffeine",                 "value": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001548"},
    {"label": "Rutin",                    "value": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579358"},
    {"label": "Apigenin",                 "value": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579385"},
]


def fetch_spectrum(usi: str, timeout: int = 15) -> dict:
    url = f"{GNPS_USI_BASE}/json/"
    resp = requests.get(url, params={"usi1": usi}, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()
    annotations = raw.get("annotations", [{}])
    meta = annotations[0] if annotations else {}
    peaks = raw.get("peaks", [])
    return {
        "usi":           usi,
        "peaks":         peaks,
        "precursor_mz":  float(raw.get("precursor_mz") or meta.get("Precursor_MZ", 0)),
        "compound_name": meta.get("Compound_Name", "Unknown"),
        "adduct":        meta.get("Adduct", "Unknown"),
        "instrument":    meta.get("Instrument", "Unknown"),
        "n_peaks":       len(peaks),
    }


def preprocess_peaks(
    peaks: list,
    top_n: Optional[int] = None,
    min_intensity_frac: float = 0.01,
) -> tuple[list[float], list[float]]:
    if not peaks:
        return [], []
    mzs_raw  = [float(p[0]) for p in peaks]
    ints_raw = [float(p[1]) for p in peaks]
    max_int  = max(ints_raw)
    threshold = min_intensity_frac * max_int
    filtered  = [(mz, i) for mz, i in zip(mzs_raw, ints_raw) if i >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        filtered = filtered[:top_n]
    filtered.sort(key=lambda x: x[0])
    mzs        = [p[0] for p in filtered]
    intensities = [p[1] / max_int for p in filtered]
    return mzs, intensities


def spectral_entropy(intensities: list[float]) -> float:
    total = sum(intensities)
    if total == 0:
        return 0.0
    ps = [i / total for i in intensities]
    return -sum(p * math.log(p) for p in ps if p > 0)


def per_peak_contributions(intensities: list[float]) -> list[float]:
    total = sum(intensities)
    if total == 0:
        return [0.0] * len(intensities)
    ps = [i / total for i in intensities]
    return [-p * math.log(p) if p > 0 else 0.0 for p in ps]


def quality_info(S: float, n: int) -> dict:
    max_S = math.log(n) if n > 1 else 0.0
    frac  = S / max_S if max_S > 0 else 0.0
    if S < 0.5:
        return {"label": "Noisy / low quality",    "color": "#E24B4A", "frac": frac}
    if S < 1.0:
        return {"label": "Moderate quality",        "color": "#BA7517", "frac": frac}
    if S < 1.75:
        return {"label": "Clean / good quality",   "color": "#1D9E75", "frac": frac}
    return     {"label": "Very diffuse / uniform", "color": "#378ADD", "frac": frac}


# ── Plotly figure builders ────────────────────────────────────────────────────

def make_spectrum_figure(mzs, intensities, contributions, precursor_mz, compound_name):
    """
    Two-panel figure:
      top    : normalised MS/MS spectrum (bar chart)
      bottom : per-peak entropy contribution −pᵢ ln(pᵢ)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
    )

    total = sum(intensities)
    ps    = [i / total for i in intensities] if total else [0] * len(intensities)
    S     = spectral_entropy(intensities)
    n     = len(intensities)
    qi    = quality_info(S, n)

    # ── top panel: spectrum bars ──────────────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=mzs,
            y=intensities,
            width=[0.8] * len(mzs),
            marker_color=[
                f"rgba(55, 138, 221, {0.3 + 0.7 * i})" for i in intensities
            ],
            marker_line_width=0,
            customdata=list(zip(ps, contributions)),
            hovertemplate=(
                "<b>m/z %{x:.4f}</b><br>"
                "Norm. intensity: %{y:.4f}<br>"
                "pᵢ: %{customdata[0]:.4f}<br>"
                "−pᵢ ln pᵢ: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # precursor marker
    if precursor_mz:
        fig.add_vline(
            x=precursor_mz, line_dash="dash", line_color="rgba(180,180,180,0.5)",
            line_width=1, row=1, col=1,
        )

    # ── bottom panel: entropy contribution bars ───────────────────────────────
    fig.add_trace(
        go.Bar(
            x=mzs,
            y=contributions,
            width=[0.8] * len(mzs),
            marker_color=qi["color"],
            marker_opacity=0.7,
            marker_line_width=0,
            hovertemplate=(
                "<b>m/z %{x:.4f}</b><br>"
                "−pᵢ ln pᵢ: %{y:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2, col=1,
    )

    # entropy sum line
    fig.add_hline(
        y=S / n if n else 0,   # mean contribution level
        line_dash="dot", line_color=qi["color"], line_width=1,
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(
            text=f"<b>{compound_name}</b>   S = {S:.3f}  |  {qi['label']}",
            font=dict(size=14),
            x=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        bargap=0.15,
    )
    fig.update_yaxes(title_text="Norm. intensity", row=1, col=1, tickfont_size=11)
    fig.update_yaxes(title_text="−pᵢ ln pᵢ",       row=2, col=1, tickfont_size=11)
    fig.update_xaxes(title_text="m/z",               row=2, col=1, tickfont_size=11)

    return fig


def make_empty_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        annotations=[dict(
            text="Enter a USI above and click Fetch",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
        )],
    )
    return fig


# ── Dash layout ───────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Spectral Entropy Viewer",
)

app.layout = dbc.Container(
    fluid=True,
    style={"maxWidth": "1100px", "paddingTop": "1.5rem"},
    children=[

        html.H4("Spectral Entropy Viewer", style={"fontWeight": "500"}),
        html.P(
            "Fetch a GNPS2 library spectrum by USI, then compute and visualise its spectral entropy.",
            className="text-muted", style={"fontSize": "13px"},
        ),

        # ── USI input row ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    id="usi-input",
                    value=EXAMPLE_USIS[0]["value"],
                    placeholder="mzspec:GNPS:GNPS-LIBRARY:accession:...",
                    style={"fontFamily": "monospace", "fontSize": "12px"},
                ),
                width=8,
            ),
            dbc.Col(
                dbc.Button("Fetch spectrum", id="fetch-btn", color="primary", size="sm"),
                width=2,
            ),
            dbc.Col(
                dbc.Select(
                    id="example-select",
                    options=[{"label": "Examples…", "value": ""}] + EXAMPLE_USIS,
                    value="",
                    size="sm",
                ),
                width=2,
            ),
        ], className="mb-2"),

        # ── preprocessing controls ────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Small("Top N peaks:", className="text-muted"), width="auto"),
            dbc.Col(
                dcc.Slider(id="top-n", min=5, max=100, step=5, value=50,
                           marks={5:"5", 25:"25", 50:"50", 75:"75", 100:"100"},
                           tooltip={"placement":"bottom"}),
                width=3,
            ),
            dbc.Col(html.Small("Min intensity (% base peak):", className="text-muted"), width="auto"),
            dbc.Col(
                dcc.Slider(id="min-int", min=0, max=10, step=1, value=1,
                           marks={0:"0", 5:"5%", 10:"10%"},
                           tooltip={"placement":"bottom"}),
                width=3,
            ),
        ], align="center", className="mb-3"),

        # ── error / status ────────────────────────────────────────────────────
        html.Div(id="status-msg", style={"fontSize": "12px", "minHeight": "18px"}),

        # ── summary metric cards ──────────────────────────────────────────────
        dbc.Row(id="metric-cards", className="mb-3 g-2"),

        # ── main figure ───────────────────────────────────────────────────────
        dcc.Graph(id="entropy-chart", figure=make_empty_figure(), config={"displayModeBar": False}),

        # ── per-peak breakdown table ──────────────────────────────────────────
        html.Hr(),
        html.H6("Per-peak breakdown", style={"fontWeight": "500", "fontSize": "13px"}),
        html.Div(id="peak-table"),

        # hidden store for spectrum data
        dcc.Store(id="spectrum-store"),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("usi-input", "value"),
    Input("example-select", "value"),
    prevent_initial_call=True,
)
def fill_example(usi):
    return usi if usi else dash.no_update


@app.callback(
    Output("spectrum-store", "data"),
    Output("status-msg", "children"),
    Input("fetch-btn", "n_clicks"),
    State("usi-input", "value"),
    State("top-n", "value"),
    State("min-int", "value"),
    prevent_initial_call=True,
)
def fetch_and_store(n_clicks, usi, top_n, min_int_pct):
    if not usi:
        return dash.no_update, dbc.Alert("Please enter a USI.", color="warning", duration=3000)

    try:
        spectrum = fetch_spectrum(usi.strip())
    except requests.HTTPError as e:
        return dash.no_update, dbc.Alert(f"HTTP error: {e}", color="danger")
    except requests.Timeout:
        return dash.no_update, dbc.Alert("Request timed out. Try again.", color="danger")
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Error: {e}", color="danger")

    mzs, ints = preprocess_peaks(
        spectrum["peaks"],
        top_n=top_n,
        min_intensity_frac=(min_int_pct or 1) / 100,
    )

    if not mzs:
        return dash.no_update, dbc.Alert("No peaks after preprocessing — try lowering the min intensity filter.", color="warning")

    n          = len(mzs)
    S          = spectral_entropy(ints)
    max_S      = math.log(n) if n > 1 else 0.0
    contribs   = per_peak_contributions(ints)
    total      = sum(ints)
    ps         = [i / total for i in ints] if total else [0.0] * n
    qi         = quality_info(S, n)

    store = {
        "compound_name": spectrum["compound_name"],
        "usi":           spectrum["usi"],
        "precursor_mz":  spectrum["precursor_mz"],
        "adduct":        spectrum["adduct"],
        "instrument":    spectrum["instrument"],
        "n_peaks_raw":   spectrum["n_peaks"],
        "n_peaks_used":  n,
        "entropy":       round(S, 4),
        "entropy_max":   round(max_S, 4),
        "entropy_frac":  round(S / max_S, 4) if max_S > 0 else 0.0,
        "quality_label": qi["label"],
        "quality_color": qi["color"],
        "mzs":           mzs,
        "intensities":   ints,
        "probabilities": ps,
        "contributions": contribs,
    }
    return store, dbc.Alert(f"Loaded {n} peaks for {spectrum['compound_name']}", color="success", duration=3000)


@app.callback(
    Output("entropy-chart", "figure"),
    Output("metric-cards", "children"),
    Output("peak-table", "children"),
    Input("spectrum-store", "data"),
)
def update_display(data):
    if not data:
        return make_empty_figure(), [], []

    mzs    = data["mzs"]
    ints   = data["intensities"]
    contribs = data["contributions"]

    fig = make_spectrum_figure(
        mzs, ints, contribs,
        data["precursor_mz"],
        data["compound_name"],
    )

    # ── metric cards ─────────────────────────────────────────────────────────
    def card(label, value, color=None):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.P(label, className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(value, className="mb-0",
                            style={"fontFamily": "monospace", "color": color or "inherit"}),
                ], style={"padding": "10px 14px"}),
            ),
            xs=6, md=3,
        )

    metrics = [
        card("Spectral entropy S",    f"{data['entropy']:.4f}",   data["quality_color"]),
        card("Max entropy ln(N)",      f"{data['entropy_max']:.4f}"),
        card("S / ln(N)",              f"{data['entropy_frac']:.3f}"),
        card("Quality",                data["quality_label"],      data["quality_color"]),
    ]

    # ── peak table ────────────────────────────────────────────────────────────
    rows = [
        {
            "m/z":          f"{mz:.4f}",
            "norm. I":      f"{i:.4f}",
            "pᵢ":           f"{p:.4f}",
            "−pᵢ ln pᵢ":   f"{c:.4f}",
        }
        for mz, i, p, c in zip(
            mzs, ints, data["probabilities"], contribs
        )
    ]
    # totals row
    rows.append({
        "m/z":        "∑",
        "norm. I":    "",
        "pᵢ":         f"{sum(data['probabilities']):.4f}",
        "−pᵢ ln pᵢ": f"{data['entropy']:.4f}",
    })

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["m/z", "norm. I", "pᵢ", "−pᵢ ln pᵢ"]],
        style_table={"overflowX": "auto", "fontSize": "12px"},
        style_header={"fontWeight": "500", "backgroundColor": "#f8f9fa", "fontSize": "11px"},
        style_cell={"fontFamily": "monospace", "padding": "5px 10px", "textAlign": "right"},
        style_data_conditional=[
            {"if": {"row_index": len(rows) - 1},
             "fontWeight": "bold", "borderTop": "2px solid #dee2e6"},
        ],
        page_size=20,
        sort_action="native",
    )

    return fig, metrics, table


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
