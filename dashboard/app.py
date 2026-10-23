"""GA4 Multi-Touch Attribution Lab: Streamlit dashboard.

Reads only files written by the pipeline (reports/*.csv, reports/metrics.json,
data/cache/*.parquet); no numbers are hard-coded here.
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mta import viz  # noqa: E402
from mta.analysis import ADDRESSABLE  # noqa: E402
from mta.attribution import MODEL_LABELS  # noqa: E402

REPORTS, CACHE = ROOT / "reports", ROOT / "data" / "cache"

st.set_page_config(page_title="GA4 Attribution Lab", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      [data-testid="stMetric"] {background: #f6f5f2; border: 1px solid #e6e5e1; border-radius: 10px; padding: 14px 16px;}
      .caption {color: #52514e; font-size: 0.92rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load() -> dict:
    """Load every pipeline output once."""
    if not (REPORTS / "metrics.json").exists():
        st.error("No pipeline outputs found. Run `make all` first.")
        st.stop()
    rd = lambda n: pd.read_csv(REPORTS / n)  # noqa: E731
    return {
        "m": json.loads((REPORTS / "metrics.json").read_text()),
        "results": rd("attribution_results.csv"),
        "changes": rd("model_comparison.csv"),
        "ci": rd("bootstrap_ci.csv"),
        "diffs": rd("bootstrap_differences.csv"),
        "bias": rd("last_touch_bias.csv"),
        "paths": rd("converting_paths.csv"),
        "path_length": rd("path_length.csv"),
        "ttc": rd("time_to_conversion.csv"),
        "funnel": rd("funnel_overall.csv"),
        "funnel_channel": rd("funnel_by_channel.csv"),
        "funnel_device": rd("funnel_by_device.csv"),
        "boot": pd.read_parquet(REPORTS / "bootstrap_replicates.parquet"),
        "daily": pd.read_parquet(CACHE / "sessions_daily.parquet"),
        "channel_map": pd.read_parquet(CACHE / "channel_map.parquet"),
    }


D = load()
M = D["m"]
CHART = {"use_container_width": True, "config": {"displaylogo": False}}


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- pages

def overview() -> None:
    o, p = M["overview"], M["profile"]
    st.title("GA4 Multi-Touch Attribution Lab")
    st.markdown(
        f"<p class='caption'>Google Merchandise Store, GA4 obfuscated sample · {p['min_date']} to {p['max_date']} "
        f"({p['n_days']} days) · {p['events']:,} events. Which channels drive purchases, and how much does "
        "the answer depend on the attribution model?</p>", unsafe_allow_html=True)

    c = st.columns(5)
    c[0].metric("Sessions", f"{o['sessions']:,}")
    c[1].metric("Users", f"{o['users']:,}")
    c[2].metric("Purchases", f"{o['purchases']:,}")
    c[3].metric("Revenue (deduplicated)", f"${o['revenue_usd']:,.0f}")
    c[4].metric("Session conversion rate", pct(o["session_cr"], 2))
    c = st.columns(5)
    c[0].metric("Average order value", f"${o['aov']:,.2f}")
    c[1].metric("Converting journeys", f"{M['journeys']['converting_journeys']:,}")
    c[2].metric("Multi-channel journeys", pct(M["journeys"]["multi_channel_share"]),
                help="Share of converting journeys touching 2 or more channels in the 30-day lookback")
    c[3].metric("Direct inferred (no source params)", pct(o["inferred_direct_share"]))
    c[4].metric("Other/Unknown channel", pct(o["other_unknown_share"]))

    st.subheader("Daily sessions and revenue by channel")
    daily = D["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    metric = st.segmented_control("Metric", ["sessions", "revenue_usd"], default="sessions",
                                  format_func=lambda x: "Sessions" if x == "sessions" else "Revenue (USD)")
    metric = metric or "sessions"
    g = daily.groupby(["date", "channel"], as_index=False)[metric].sum()
    fig = go.Figure()
    for ch in [c for c in viz.CHANNEL_ORDER if c in set(g.channel)]:
        s = g[g.channel == ch]
        fig.add_trace(go.Scatter(x=s.date, y=s[metric], name=ch, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=viz.CHANNEL_COLORS[ch]),
                                 hovertemplate=ch + ": %{y:,.0f}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%b %d")
    st.plotly_chart(viz.style(fig, None, 380), **CHART)

    st.subheader("Channel summary")
    ch = pd.DataFrame(o["channels"]).set_index("channel").reindex([c for c in viz.CHANNEL_ORDER if c in {x["channel"] for x in o["channels"]}])
    tbl = pd.DataFrame({
        "Sessions": ch.sessions.map("{:,.0f}".format), "Share of sessions": ch.share.map(pct),
        "Converting sessions": ch.converting_sessions.map("{:,.0f}".format),
        "Session CR": ch.session_cr.map(lambda v: pct(v, 2)), "Revenue": ch.revenue_usd.map("${:,.0f}".format),
        "Inferred Direct": ch.inferred_sessions.map("{:,.0f}".format),
    })
    st.dataframe(tbl, use_container_width=True)
    with st.expander("Channel grouping rules (dbt seed) and the source/medium values they matched"):
        st.dataframe(D["channel_map"], use_container_width=True, hide_index=True)


def funnel() -> None:
    f = M["funnel"]
    st.title("Funnel")
    st.markdown(
        f"<p class='caption'>Closed session funnel: a session counts at a step only if it also reached every "
        f"earlier step. The steepest drop after the product page is <b>{f['weakest_post_view_from']} → "
        f"{f['weakest_post_view_step']}</b>: only {pct(f['weakest_post_view_rate'])} continue "
        f"({f['weakest_post_view_lost']:,} of {f['weakest_post_view_prev']:,} sessions lost).</p>",
        unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Sessions", f"{f['sessions']:,}")
    c[1].metric("Closed-funnel purchases", f"{f['purchase_sessions_closed']:,}")
    c[2].metric("All purchase sessions", f"{f['purchase_sessions_open']:,}",
                help="Includes purchases whose earlier steps happened in a previous session")
    c[3].metric(f"Desktop vs mobile at {f['device_test_step']}",
                f"{pct(f['desktop_step_rate'])} vs {pct(f['mobile_step_rate'])}",
                help=f"Two-proportion z-test p = {f['device_p']:.2f}")
    st.plotly_chart(viz.funnel_chart(D["funnel"]), **CHART)

    dim = st.segmented_control("Break down by", ["channel", "device_category"], default="channel",
                               format_func=lambda x: "Channel" if x == "channel" else "Device")
    dim = dim or "channel"
    src = D["funnel_channel"] if dim == "channel" else D["funnel_device"]
    st.plotly_chart(viz.step_rate_heatmap(src[src.step_order > 1], dim), **CHART)
    with st.expander("Counts"):
        piv = src.pivot(index=dim, columns="step_label", values="sessions")
        st.dataframe(piv[[c for c in D["funnel"].step_label if c in piv.columns]].fillna(0).astype(int), use_container_width=True)


def journeys() -> None:
    j = M["journeys"]
    st.title("Journeys")
    st.markdown("<p class='caption'>A journey is a user's sessions in the 30 days up to a purchase (or up to their "
                "last session if they never bought). Sessions after a purchase start a new journey.</p>",
                unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Converting journeys", f"{j['converting_journeys']:,}")
    c[1].metric("Single-touch", pct(j["single_touch_share"]))
    c[2].metric("Median time to purchase", f"{j['median_days_to_conversion'] * 24:,.1f} h")
    c[3].metric("90th percentile", f"{j['p90_days_to_conversion']:,.1f} days")
    top_n = st.slider("Paths in Sankey", 5, 30, 15)
    st.plotly_chart(viz.sankey(D["paths"], top_n=top_n), **CHART)
    a, b = st.columns(2)
    a.plotly_chart(viz.path_length_chart(D["path_length"]), **CHART)
    b.plotly_chart(viz.time_to_conversion_chart(D["ttc"]), **CHART)
    st.subheader("Top converting paths")
    t = D["paths"].head(25).copy()
    t["share"] = t["share"].map(pct)
    t["revenue"] = t["revenue"].map("${:,.0f}".format)
    st.dataframe(t, use_container_width=True, hide_index=True)


def attribution() -> None:
    st.title("Attribution comparison")
    st.markdown("<p class='caption'>Eight models, all implemented from scratch in <code>src/mta/attribution.py</code> "
                "and unit-tested against hand-calculated examples.</p>", unsafe_allow_html=True)
    res = D["results"]
    a, b = st.columns([1, 1])
    model = a.selectbox("Model", list(MODEL_LABELS), index=6, format_func=MODEL_LABELS.get)
    metric = b.segmented_control("Metric", ["conversions", "revenue"], default="conversions",
                                 format_func=str.capitalize) or "conversions"
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.model_bars(res, model, metric, MODEL_LABELS[model]), **CHART)
    c2.plotly_chart(viz.model_bars(res, "last_touch", metric, "Last touch (baseline)"), **CHART)

    st.plotly_chart(viz.credit_heatmap(res, MODEL_LABELS), **CHART)
    st.plotly_chart(viz.pct_change_heatmap(D["changes"][D["changes"].model.isin(["markov", "shapley"])]), **CHART)

    st.subheader("Is each channel over- or under-credited by last touch?")
    st.markdown("<p class='caption'>A verdict needs <b>both</b> data-driven models to differ from last touch "
                "in the same direction, with 95% bootstrap CIs excluding zero.</p>", unsafe_allow_html=True)
    bias = D["bias"][D["bias"].metric == metric].copy()
    show = pd.DataFrame({
        "Channel": bias.channel, "Last touch": bias.last_touch, "Markov": bias.markov, "Shapley": bias.shapley,
        "Markov vs LT": bias.markov_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
        "Shapley vs LT": bias.shapley_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
        "Verdict": bias.verdict,
    })
    fmt = "${:,.0f}" if metric == "revenue" else "{:,.1f}"
    for col in ["Last touch", "Markov", "Shapley"]:
        show[col] = show[col].map(fmt.format)
    st.dataframe(show, use_container_width=True, hide_index=True)
    with st.expander("Full results table (all models)"):
        st.dataframe(res.pivot(index="channel", columns="model_label", values=metric).round(1), use_container_width=True)


def intervals() -> None:
    st.title("Confidence intervals")
    st.markdown(f"<p class='caption'>{M['attribution']['n_boot']:,} bootstrap resamples of user journeys. Each "
                "resample re-fits Markov and Shapley <i>and</i> last touch, so the differences have paired "
                "intervals. Hollow markers: the 95% CI crosses zero, so the difference is not statistically "
                "meaningful.</p>", unsafe_allow_html=True)
    metric = st.segmented_control("Metric", ["conversions", "revenue"], default="revenue",
                                  format_func=str.capitalize) or "revenue"
    lt = D["results"][D["results"].model == "last_touch"].set_index("channel")[metric]
    st.plotly_chart(viz.diff_ci_plot(D["diffs"], lt, metric), **CHART)
    st.plotly_chart(viz.ci_plot(D["ci"], metric), **CHART)
    d = D["diffs"][D["diffs"].metric == metric].copy()
    d["95% CI"] = [f"{lo:,.1f} to {hi:,.1f}" for lo, hi in zip(d.ci_low, d.ci_high)]
    st.dataframe(d[["comparison", "channel", "estimate", "95% CI", "significant"]].round(1),
                 use_container_width=True, hide_index=True)


def budget() -> None:
    b = M["budget"]
    st.title("Budget simulator")
    st.markdown(
        "<p class='caption'><b>Directional, not a forecast.</b> GA4 has no spend data, so “effort” is measured in "
        "sessions, and each session is valued at its channel's data-driven credited revenue per session. "
        "Returns factor k scales the value of the <i>added</i> sessions (k = 1 means constant returns; lower "
        "values mean diminishing returns). Correlational credit, not incrementality.</p>", unsafe_allow_html=True)
    st.info(f"Pipeline default: shift {pct(b['shift_pct'], 0)} of {b['from_channel']} sessions "
            f"(most over-credited by last touch on revenue, mean {b['from_mean_pct_change']:+.1f}%) to "
            f"{b['to_channel']} (most under-credited, mean {b['to_mean_pct_change']:+.1f}%).")
    chans = [c for c in ADDRESSABLE if c in b["rev_per_session_markov"]]
    a, c, d = st.columns(3)
    src = a.selectbox("Move effort from", chans, index=chans.index(b["from_channel"]))
    dst = c.selectbox("to", [x for x in chans if x != src],
                      index=max(0, [x for x in chans if x != src].index(b["to_channel"])) if b["to_channel"] != src else 0)
    basis = d.selectbox("Value sessions with", ["markov", "shapley", "last_touch"], format_func=MODEL_LABELS.get)
    s1, s2 = st.columns(2)
    shift = s1.slider("Share of source-channel sessions moved", 0, 50, int(b["shift_pct"] * 100), format="%d%%") / 100
    k = s2.slider("Returns factor k for added sessions", 0.1, 1.5, 1.0, 0.05)

    sessions = D["daily"].groupby("channel")["sessions"].sum()
    rev = D["results"][D["results"].model == basis].set_index("channel")["revenue"]
    vps = rev / sessions
    moved = shift * sessions[src]
    delta = moved * (k * vps[dst] - vps[src])
    bw = D["boot"][D["boot"].model == basis].pivot(index="replicate", columns="channel", values="revenue")
    dist = moved * (k * bw[dst] / sessions[dst] - bw[src] / sessions[src])
    lo, hi = np.percentile(dist, [2.5, 97.5])

    m = st.columns(4)
    m[0].metric("Sessions moved", f"{moved:,.0f}")
    m[1].metric(f"{src} value / session", f"${vps[src]:,.2f}")
    m[2].metric(f"{dst} value / session", f"${vps[dst]:,.2f}")
    m[3].metric("Revenue change (92 days)", viz.money(delta, signed=True),
                help=f"95% bootstrap CI {viz.money(lo, True)} to {viz.money(hi, True)}".replace("$", "\\$"))
    st.markdown(f"95% bootstrap interval: **{viz.money(lo, True)} to {viz.money(hi, True)}**. ".replace("$", "\\$")
                + ("The interval excludes zero." if (lo > 0 or hi < 0) else "The interval includes zero, so the direction is uncertain."))

    st.plotly_chart(viz.budget_chart(pd.read_csv(REPORTS / "budget_simulation.csv")), **CHART)
    vt = pd.DataFrame({"Markov $/session": pd.Series(b["rev_per_session_markov"]),
                       "Last touch $/session": pd.Series(b["rev_per_session_last_touch"]),
                       "Sessions": sessions}).reindex([c for c in viz.CHANNEL_ORDER if c in sessions.index])
    st.subheader("Credited revenue per session")
    st.dataframe(vt.round(2), use_container_width=True)
    st.markdown("**Assumptions:** sessions are interchangeable units of effort; the value of a moved session equals "
                "the channel's average credited value (times k for the receiving channel); no cost differences "
                "between channels; Direct and Other/Unknown are not budget levers. **Limitations:** attribution "
                "credit is correlational, sample period is 92 days of obfuscated data, and channel capacity "
                "(e.g. email list size, search demand) is not modelled.")


nav = st.navigation([
    st.Page(overview, title="Overview", icon="🏠", default=True),
    st.Page(funnel, title="Funnel", icon="🔻", url_path="funnel"),
    st.Page(journeys, title="Journeys", icon="🔀", url_path="journeys"),
    st.Page(attribution, title="Attribution comparison", icon="⚖️", url_path="attribution"),
    st.Page(intervals, title="Confidence intervals", icon="📏", url_path="intervals"),
    st.Page(budget, title="Budget simulator", icon="💸", url_path="budget"),
])
nav.run()
