"""Shared plotly figures and styling for the dashboard, static report and README images.

Palette: validated categorical slots (fixed channel -> colour mapping so a channel keeps
its colour on every chart), one-hue blue sequential ramp, blue/red diverging with a gray
midpoint. Three light slots are below 3:1 contrast, so charts carry direct labels and
every chart has a table next to it in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

CHANNEL_ORDER = [
    "Direct", "Organic Search", "Referral", "Paid Search", "Email", "Social", "Display", "Other/Unknown",
]
CHANNEL_COLORS = {
    "Direct": "#2a78d6",
    "Organic Search": "#eb6834",
    "Referral": "#1baf7a",
    "Paid Search": "#eda100",
    "Email": "#e87ba4",
    "Social": "#008300",
    "Display": "#4a3aa7",
    "Other/Unknown": "#8a8984",
}
MODEL_ORDER = [
    "first_touch", "last_touch", "last_non_direct", "linear", "time_decay", "position_based", "markov", "shapley",
]
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#c43a3a"], [0.5, "#f0efec"], [1.0, "#1c5cab"]]  # red = lost credit, blue = gained
ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def money(v: float, signed: bool = False, decimals: int = 0) -> str:
    """$1,234 / -$1,234 / +$1,234 (sign before the currency symbol)."""
    sign = "-" if v < 0 else ("+" if signed and v > 0 else "")
    return f"{sign}${abs(v):,.{decimals}f}"


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)) if title else None,
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=20, t=60 if title else 20, b=10),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#bdbcb7", linecolor=GRID)
    return fig


def _channels_in(df: pd.DataFrame, col: str = "channel") -> list[str]:
    present = set(df[col])
    return [c for c in CHANNEL_ORDER if c in present]


def credit_heatmap(comparison: pd.DataFrame, labels: dict[str, str], value: str = "share") -> go.Figure:
    """Channel x model heatmap of each channel's share of credited conversions."""
    piv = comparison.pivot(index="channel", columns="model", values=value)
    piv = piv.reindex(index=_channels_in(comparison), columns=[m for m in MODEL_ORDER if m in piv.columns])
    z = piv.to_numpy() * 100
    text = [[f"{v:.1f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[labels[m] for m in piv.columns], y=piv.index, text=text, texttemplate="%{text}",
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar=dict(title="Share of<br>conversions", ticksuffix="%"), xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{text} of credited conversions<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-25)
    return style(fig, "Share of conversion credit by channel and model", height=460)


def pct_change_heatmap(changes: pd.DataFrame) -> go.Figure:
    """Channel x data-driven model: % change in credited conversions vs last touch."""
    piv = changes.pivot(index="channel", columns="model", values="pct_change")
    piv = piv.reindex(index=_channels_in(changes))
    cols = [c for c in ["markov", "shapley"] if c in piv.columns]
    z = piv[cols].to_numpy()
    lim = float(np.nanmax(np.abs(np.clip(z, -100, 100)))) if np.isfinite(z).any() else 1.0
    text = [["n/a" if not np.isfinite(v) else f"{v:+.0f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=np.clip(z, -lim, lim), x=[{"markov": "Markov", "shapley": "Shapley"}[c] for c in cols], y=piv.index,
        text=text, texttemplate="%{text}", colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
        colorbar=dict(title="vs last touch", ticksuffix="%"), xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}: %{text} vs last touch<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return style(fig, "How much credit changes vs last touch", height=420)


def model_bars(results: pd.DataFrame, model: str, metric: str, label: str) -> go.Figure:
    """Credited conversions or revenue for one model, one bar per channel (channel colours)."""
    d = results[results.model == model].set_index("channel").reindex(_channels_in(results)).reset_index()
    fmt = "$%{x:,.0f}" if metric == "revenue" else "%{x:,.0f}"
    fig = go.Figure(go.Bar(
        y=d.channel, x=d[metric], orientation="h", marker_color=[CHANNEL_COLORS[c] for c in d.channel],
        text=[f"${v:,.0f}" if metric == "revenue" else f"{v:,.0f}" for v in d[metric]], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: " + fmt + "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    title = f"{label}: credited {'revenue' if metric == 'revenue' else 'conversions'}"
    return style(fig, title, height=380)


def ci_plot(ci: pd.DataFrame, metric: str = "conversions") -> go.Figure:
    """Point estimate + 95% bootstrap CI for Markov and Shapley, with last touch as a reference marker."""
    d = ci[ci.metric == metric]
    chans = _channels_in(d)
    fig = go.Figure()
    spec = {"last_touch": ("Last touch", "#52514e", "diamond-open", 0.0),
            "markov": ("Markov", "#2a78d6", "circle", -0.18), "shapley": ("Shapley", "#eb6834", "square", 0.18)}
    pos = {c: i for i, c in enumerate(chans)}
    for model, (name, color, symbol, off) in spec.items():
        m = d[d.model == model].set_index("channel").reindex(chans)
        fig.add_trace(go.Scatter(
            x=m["estimate"], y=[pos[c] + off for c in chans], mode="markers", name=name,
            marker=dict(color=color, symbol=symbol, size=10, line=dict(width=2, color=color)),
            error_x=dict(type="data", symmetric=False, array=m["ci_high"] - m["estimate"],
                         arrayminus=m["estimate"] - m["ci_low"], color=color, thickness=2, width=4),
            customdata=np.stack([chans, m["ci_low"], m["ci_high"]], axis=1),
            hovertemplate="%{customdata[0]} · " + name + "<br>%{x:,.1f} (95% CI %{customdata[1]:,.1f} to %{customdata[2]:,.1f})<extra></extra>",
        ))
    fig.update_yaxes(tickvals=list(pos.values()), ticktext=chans, autorange="reversed", showgrid=False, zeroline=False)
    fig.update_xaxes(title="Credited " + metric + (" (USD)" if metric == "revenue" else ""))
    return style(fig, "Data-driven credit with 95% bootstrap intervals (1,000 resamples)", height=460)


def diff_ci_plot(diffs: pd.DataFrame, last_touch: pd.Series, metric: str = "conversions") -> go.Figure:
    """% difference vs last touch for Markov and Shapley with paired 95% bootstrap CIs.
    Intervals that cross 0 are not statistically meaningful (drawn hollow)."""
    d = diffs[(diffs.metric == metric) & diffs.comparison.isin(["markov - last_touch", "shapley - last_touch"])]
    chans = [c for c in _channels_in(d) if last_touch.get(c, 0) > 0]
    pos = {c: i for i, c in enumerate(chans)}
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color="#bdbcb7", width=1.5))
    for comp, name, color, off in [("markov - last_touch", "Markov", "#2a78d6", -0.15),
                                   ("shapley - last_touch", "Shapley", "#eb6834", 0.15)]:
        m = d[d.comparison == comp].set_index("channel").reindex(chans)
        base = last_touch.reindex(chans).to_numpy()
        est, lo, hi = (m[c].to_numpy() / base * 100 for c in ("estimate", "ci_low", "ci_high"))
        sig = m["significant"].to_numpy(dtype=bool)
        fig.add_trace(go.Scatter(
            x=est, y=[pos[c] + off for c in chans], mode="markers", name=name,
            marker=dict(color=[color if s_ else SURFACE for s_ in sig], size=11, line=dict(width=2, color=color)),
            error_x=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo, color=color, thickness=2, width=4),
            customdata=np.stack([chans, lo, hi, np.where(sig, "significant", "not significant (CI crosses 0)")], axis=1),
            hovertemplate="%{customdata[0]} · " + name + ": %{x:+.1f}% vs last touch<br>95% CI %{customdata[1]:+.1f}% to %{customdata[2]:+.1f}%<br>%{customdata[3]}<extra></extra>",
        ))
    fig.update_yaxes(tickvals=list(pos.values()), ticktext=chans, autorange="reversed", showgrid=False, zeroline=False)
    fig.update_xaxes(title=f"Change in credited {metric} vs last touch (%). Hollow = CI crosses 0", ticksuffix="%", zeroline=False)
    return style(fig, f"Data-driven vs last touch: {metric}, 95% bootstrap CI (1,000 resamples)", height=440)


def funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    """Closed funnel: sessions reaching each step, with step-to-step conversion labels."""
    fig = go.Figure(go.Bar(
        x=funnel["sessions"], y=funnel["step_label"], orientation="h",
        marker_color=ORDINAL[: len(funnel)][::-1] if len(funnel) <= len(ORDINAL) else ORDINAL[-1],
        text=[f"{s:,.0f}" + ("" if pd.isna(r) else f"  ·  {r:.1%} of previous") for s, r in zip(funnel.sessions, funnel.step_rate)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} sessions<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(type="log", title="Sessions (log scale)", tickvals=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5],
                     ticktext=["1k", "3k", "10k", "30k", "100k", "300k"], range=[3, 5.75])
    return style(fig, "Session funnel (closed: each step requires all earlier steps)", height=380)


def step_rate_heatmap(rates: pd.DataFrame, dim: str) -> go.Figure:
    """Step-to-step conversion rate by segment (rows) and step (columns)."""
    piv = rates.pivot(index=dim, columns="step_label", values="step_rate")
    piv = piv[[c for c in rates.drop_duplicates("step_order").sort_values("step_order").step_label if c in piv.columns]]
    if dim == "channel":
        piv = piv.reindex(_channels_in(rates))
    z = piv.to_numpy() * 100
    text = [["–" if not np.isfinite(v) else f"{v:.1f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=piv.columns, y=piv.index, text=text, texttemplate="%{text}",
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar=dict(title="Step<br>conversion", ticksuffix="%"), xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}: %{text} of previous step<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style(fig, f"Step-to-step conversion by {dim.replace('_', ' ')}", height=120 + 40 * len(piv))


def path_length_chart(dist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=dist["bucket"], y=dist["share"] * 100, marker_color="#2a78d6",
        text=[f"{v:.0%}" for v in dist["share"]], textposition="outside", cliponaxis=False,
        customdata=dist["journeys"], hovertemplate="%{x} touches: %{customdata:,} journeys (%{y:.1f}%)<extra></extra>",
    ))
    fig.update_xaxes(title="Touches (sessions) in the 30-day journey", type="category", showgrid=False)
    fig.update_yaxes(title="Share of converting journeys", ticksuffix="%")
    return style(fig, "Path length of converting journeys", height=360)


def time_to_conversion_chart(dist: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=dist["bucket"], y=dist["share"] * 100, marker_color="#2a78d6",
        text=[f"{v:.0%}" for v in dist["share"]], textposition="outside", cliponaxis=False,
        customdata=dist["journeys"], hovertemplate="%{x}: %{customdata:,} journeys (%{y:.1f}%)<extra></extra>",
    ))
    fig.update_xaxes(title="First touch to purchase", type="category", showgrid=False)
    fig.update_yaxes(title="Share of converting journeys", ticksuffix="%")
    return style(fig, "Time to conversion", height=360)


def sankey(paths: pd.DataFrame, max_steps: int = 4, top_n: int = 15) -> go.Figure:
    """Sankey of the top converting paths: step-k channel -> step-k+1 channel -> Purchase."""
    top = paths.head(top_n)
    labels: list[str] = []
    colors: list[str] = []
    index: dict[tuple[int, str], int] = {}

    def node(step: int, name: str) -> int:
        key = (step, name)
        if key not in index:
            index[key] = len(labels)
            labels.append(name)
            colors.append(CHANNEL_COLORS.get(name, INK_2))
        return index[key]

    links: dict[tuple[int, int], float] = {}
    for path, n in zip(top["path"], top["journeys"]):
        chans = path.split(" > ")
        if len(chans) > max_steps:
            chans = chans[: max_steps - 1] + [f"… +{len(chans) - max_steps + 1} more"]
        seq = [node(i, c) for i, c in enumerate(chans)] + [node(99, "Purchase")]
        for a, b in zip(seq[:-1], seq[1:]):
            links[(a, b)] = links.get((a, b), 0) + n
    src, tgt, val = zip(*[(a, b, v) for (a, b), v in links.items()])
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=14, thickness=14, line=dict(width=0)),
        link=dict(source=src, target=tgt, value=val, color="rgba(138,137,132,0.25)",
                  hovertemplate="%{source.label} → %{target.label}: %{value:,} journeys<extra></extra>"),
    ))
    return style(fig, f"Top {top_n} converting paths (first {max_steps} touches)", height=520)


def budget_chart(sim: pd.DataFrame) -> go.Figure:
    """Expected revenue change under each returns assumption, with bootstrap 95% CI."""
    fig = go.Figure(go.Bar(
        x=sim["scenario"], y=sim["delta_revenue"],
        marker_color=["#1c5cab" if v >= 0 else "#c43a3a" for v in sim["delta_revenue"]],
        error_y=dict(type="data", symmetric=False, array=sim["ci_high"] - sim["delta_revenue"],
                     arrayminus=sim["delta_revenue"] - sim["ci_low"], color="#52514e", thickness=1.5, width=6),
        text=[money(v, signed=True) for v in sim["delta_revenue"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"),
        hovertemplate="%{x}: %{y:$,.0f}<extra></extra>",
    ))
    fig.update_yaxes(title="Change in data-driven revenue (USD, 92 days)", tickprefix="$")
    fig.update_xaxes(showgrid=False)
    return style(fig, "Directional impact of the reallocation", height=380)
