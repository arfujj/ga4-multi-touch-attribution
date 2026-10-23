"""GA4 Multi-Touch Attribution Lab: Streamlit dashboard.

A guided story page plus explore pages. Reads only files written by the pipeline
(reports/*.csv, reports/metrics.json, reports/*.parquet); no numbers are hard-coded here,
and story text reuses the guarded findings from mta.writeup (same wording as the README/memo).
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
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
from mta.attribution import MODEL_LABELS  # noqa: E402
from mta.writeup import Facts, findings, guard, limitations  # noqa: E402

REPORTS = ROOT / "reports"
REPO_URL = "https://github.com/arfujj/ga4-multi-touch-attribution"

st.set_page_config(page_title="Which channels really drive sales? · GA4 Attribution Lab", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --line:#e6e5e1; --card:#ffffff; --tint:#f4f8fd;
        --blue:#2a78d6; --blue-dk:#1c5cab; --orange:#eb6834; --good:#1c5cab; --warn:#c43a3a; }
html, body, [class*="css"] { font-family: Inter, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.block-container { padding-top: 4.2rem; padding-bottom: 4rem; max-width: 1080px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 600; color: var(--blue); margin: 0 0 .4rem; }
.hero h1 { font-family: Newsreader, Georgia, serif; font-weight: 600; font-size: clamp(2.1rem, 4.6vw, 3.3rem);
           line-height: 1.08; letter-spacing: -.015em; margin: 0 0 .8rem; color: var(--ink); }
.hero .dek { font-size: 1.12rem; line-height: 1.6; color: var(--ink2); max-width: 760px; margin: 0; }
.byline { color: var(--muted); font-size: .86rem; margin-top: .9rem; }
.byline a { color: var(--ink2); }
.tldr { background: var(--tint); border: 1px solid #cde2fb; border-radius: 14px; padding: 18px 22px; margin: 6px 0 4px; }
.tldr h3 { margin: 0 0 .5rem; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; color: var(--blue-dk); }
.tldr ol { margin: 0; padding-left: 1.2rem; } .tldr li { margin: .35rem 0; line-height: 1.5; color: var(--ink); }
.chapter { margin-top: 3.2rem; padding-top: 1.6rem; border-top: 1px solid var(--line); }
.chapter .num { font-family: Newsreader, Georgia, serif; font-size: 1rem; color: var(--muted); margin-bottom: .2rem; }
.chapter h2 { font-family: Newsreader, Georgia, serif; font-weight: 600; font-size: clamp(1.6rem, 3vw, 2.1rem);
              line-height: 1.15; margin: 0 0 .7rem; color: var(--ink); letter-spacing: -.01em; }
.chapter p { font-size: 1.02rem; line-height: 1.65; color: var(--ink2); max-width: 780px; margin: 0 0 .6rem; }
.chapter p b { color: var(--ink); }
.takeaway { display: flex; gap: 12px; align-items: flex-start; background: var(--card); border: 1px solid var(--line);
            border-left: 4px solid var(--blue); border-radius: 10px; padding: 14px 18px; margin: .4rem 0 0; }
.takeaway .lbl { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--blue); white-space: nowrap; padding-top: 3px; }
.takeaway .txt { font-size: 1.02rem; line-height: 1.55; color: var(--ink); }
.takeaway.warn { border-left-color: var(--warn); } .takeaway.warn .lbl { color: var(--warn); }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; height: 100%; }
.stat .v { font-size: 2rem; font-weight: 700; color: var(--ink); line-height: 1.1; }
.stat .l { font-size: .92rem; color: var(--ink2); margin-top: .35rem; line-height: 1.45; }
.legend-note { font-size: .85rem; color: var(--muted); margin: -.2rem 0 .4rem; }
.recs { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: .6rem; }
.rec { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.rec .n { font-family: Newsreader, Georgia, serif; font-size: 1.5rem; color: var(--blue); line-height: 1; }
.rec .t { font-weight: 600; color: var(--ink); margin: .4rem 0 .35rem; line-height: 1.35; }
.rec .a { color: var(--ink2); font-size: .93rem; line-height: 1.5; }
.page-title h1 { font-family: Newsreader, Georgia, serif; font-weight: 600; font-size: 2.3rem; margin: 0 0 .3rem; }
.page-title p { color: var(--ink2); font-size: 1rem; line-height: 1.6; max-width: 780px; }
.footer { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .86rem; }
.footer a { color: var(--ink2); }
@media (max-width: 640px) { .takeaway { flex-direction: column; gap: 4px; } .stat .v { font-size: 1.6rem; } }
</style>
"""
st.html(CSS)


@st.cache_data
def load() -> dict:
    """Load every pipeline output once."""
    if not (REPORTS / "metrics.json").exists():
        st.error("No pipeline outputs found. Run `make all` first.")
        st.stop()
    rd = lambda n: pd.read_csv(REPORTS / n)  # noqa: E731
    F = Facts()
    return {
        "m": json.loads((REPORTS / "metrics.json").read_text()),
        "findings": findings(F),
        "limitations": limitations(F),
        "results": rd("attribution_results.csv"),
        "changes": rd("model_comparison.csv"),
        "ci": rd("bootstrap_ci.csv"),
        "diffs": rd("bootstrap_differences.csv"),
        "bias": rd("last_touch_bias.csv"),
        "sim": rd("budget_simulation.csv"),
        "paths": rd("converting_paths.csv"),
        "path_length": rd("path_length.csv"),
        "ttc": rd("time_to_conversion.csv"),
        "funnel": rd("funnel_overall.csv"),
        "funnel_channel": rd("funnel_by_channel.csv"),
        "funnel_device": rd("funnel_by_device.csv"),
        "daily": rd("sessions_daily.csv").assign(date=lambda d: pd.to_datetime(d["date"])),
        "channel_map": rd("channel_map.csv"),
        "boot": pd.read_parquet(REPORTS / "bootstrap_replicates.parquet"),
    }


D = load()
M = D["m"]
FIND = D["findings"]
CHART = {"width": "stretch", "config": {"displaylogo": False, "displayModeBar": False}}


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def esc(s: str) -> str:
    return html.escape(str(s))


def chapter(num: int, title: str, *paras: str) -> None:
    body = "".join(f"<p>{p}</p>" for p in paras)
    st.html(f"<div class='chapter'><div class='num'>Chapter {num}</div><h2>{esc(title)}</h2>{body}</div>")


def takeaway(text: str, warn: bool = False) -> None:
    st.html(f"<div class='takeaway{' warn' if warn else ''}'><div class='lbl'>Takeaway</div><div class='txt'>{text}</div></div>")


def stat(value: str, label: str) -> str:
    return f"<div class='stat'><div class='v'>{value}</div><div class='l'>{label}</div></div>"


def page_title(title: str, intro: str) -> None:
    st.html(f"<div class='page-title'><h1>{esc(title)}</h1><p>{intro}</p></div>")


def footer() -> None:
    st.html(f"<div class='footer'>Built with BigQuery, dbt, Python and Streamlit on the public GA4 obfuscated "
            f"sample (Google Merchandise Store). Every number on this page is computed by the pipeline. "
            f"<a href='{REPO_URL}' target='_blank'>Code and method on GitHub →</a></div>")


# --------------------------------------------------------------------------- the story

def story() -> None:
    o, j, f, b, p = M["overview"], M["journeys"], M["funnel"], M["budget"], M["profile"]
    bias = D["bias"].set_index(["metric", "channel"])
    org, dirc = bias.loc[("revenue", "Organic Search")], bias.loc[("revenue", "Direct")]
    ps = bias.loc[("conversions", "Paid Search")]
    vps = pd.Series(b["rev_per_session_markov"])
    base = D["sim"].iloc[0]

    st.html(f"""
    <div class='hero'>
      <div class='eyebrow'>Marketing analytics case study · Google Merchandise Store</div>
      <h1>Which marketing channels really drive sales?</h1>
      <p class='dek'>Most reports give all the credit for a sale to the <b>last</b> place a customer came from.
      I rebuilt {p['n_days']} days of Google Analytics data into {j['journeys']:,} customer journeys and tested eight
      different ways of sharing that credit, to see which channels are over- or under-valued and what to do about it.</p>
      <div class='byline'>{p['min_date']} to {p['max_date']} · {p['events']:,} raw events ·
      <a href='{REPO_URL}' target='_blank'>source code</a></div>
    </div>""")

    c = st.columns(4)
    c[0].metric("Revenue", viz.money(o["revenue_usd"]))
    c[1].metric("Purchases", f"{o['purchases']:,}")
    c[2].metric("Website visits", f"{o['sessions']:,}")
    c[3].metric("Multi-channel purchases", pct(j["multi_channel_share"], 0), help="Purchases whose 30-day journey touched 2 or more channels")

    items = "".join(f"<li><b>{esc(x['title'])}.</b></li>" for x in FIND)
    st.html(f"<div class='tldr'><h3>The short version</h3><ol>{items}</ol></div>")

    # 1. visibility
    known = o["sessions"] - o["inferred_direct_sessions"] - o["other_unknown_sessions"]
    chapter(1, "First: how much of the traffic can we actually see?",
            "Google Analytics records where each visit came from: a Google search, another website, an email. "
            "Before comparing channels, I checked how complete that information is.",
            f"Only <b>{pct(known / o['sessions'], 0)}</b> of visits have a clear source. "
            f"<b>{pct(o['inferred_direct_share'], 0)}</b> arrive with no source at all, and Google Analytics "
            f"files them under <i>Direct</i>. Another <b>{pct(o['other_unknown_share'], 0)}</b> were scrambled "
            "by Google when it anonymised this public dataset.")
    st.plotly_chart(viz.coverage_bar(o["sessions"], o["inferred_direct_sessions"], o["other_unknown_sessions"]), **CHART)
    direct_sessions = D["daily"].loc[D["daily"].channel == "Direct", "sessions"].sum()
    takeaway(f"<b>Direct</b> is inflated before any model runs: {pct(o['inferred_direct_sessions'] / direct_sessions, 0)} "
             "of “Direct” visits are really <i>unknown</i>. Fixing tracking is the cheapest win in this whole analysis.", warn=True)

    # 2. how people buy
    chapter(2, "How do people actually buy?",
            f"I grouped each shopper's visits from the 30 days before a purchase into a <b>journey</b>. "
            f"Most purchases are quick, but a large minority involve several channels. Those are the "
            "purchases where the choice of attribution model matters.")
    s = st.columns(3)
    s[0].html(stat(pct(j["single_touch_share"], 0), "of purchases happen on the shopper's <b>first and only</b> visit"))
    s[1].html(stat(pct(j["same_day_share"], 0), "of purchases happen within <b>a day</b> of the first visit"))
    s[2].html(stat(pct(j["multi_channel_revenue_share"], 0),
                   f"of revenue comes from journeys that touched <b>two or more channels</b> "
                   f"({pct(j['multi_channel_share'], 0)} of purchases)"))
    st.plotly_chart(viz.sankey(D["paths"], top_n=12), **CHART)
    st.html("<div class='legend-note'>The 12 most common paths to purchase, read left to right. Wider bands = more purchases.</div>")
    guard(D["paths"].iloc[0]["path"] == "Organic Search", "the most common converting path is a single Organic Search visit")
    takeaway(f"Organic Search is the most common way a buying journey <i>starts</i>: <b>{pct(D['paths'].iloc[0]['share'], 0)}</b> of all "
             "purchases are a single Google search visit. But many longer journeys start with search and end with a "
             "Direct visit, and last-click reporting gives Direct the whole sale.")

    # 3. what last click gets wrong
    chapter(3, "What does last-click reporting get wrong?",
            "Below, each row shows the revenue a channel gets credited with under three rules: "
            "<b>last click</b> (the usual report), and two data-driven models that look at the whole journey. "
            "<b>Markov</b> asks how many sales would be lost if a channel disappeared. <b>Shapley</b> shares each sale "
            "fairly among the channels that took part.",
            f"Organic Search gains <b>{viz.money(org.markov - org.last_touch)}</b> (Markov) and "
            f"<b>{viz.money(org.shapley - org.last_touch)}</b> (Shapley) of credit. Direct loses "
            f"<b>{viz.money(dirc.last_touch - dirc.markov)}</b> and <b>{viz.money(dirc.last_touch - dirc.shapley)}</b>.")
    st.plotly_chart(viz.dumbbell(D["bias"], "revenue", ["Direct", "Organic Search", "Referral", "Other/Unknown"]), **CHART)
    st.html("<div class='legend-note'>Gray band = how far a channel's credit moves between the three rules. "
            "Smaller channels are in the Attribution explorer.</div>")
    lo_o, hi_o = sorted([org.markov_pct_change, org.shapley_pct_change])
    takeaway(f"Last click <b>undervalues Organic Search by {lo_o:.0f} to {hi_o:.0f}%</b> "
             f"and <b>overvalues Direct</b>. Search starts journeys that other channels finish.")

    # 4. is it real
    chapter(4, "Is that a real difference, or just noise?",
            f"With a few thousand purchases, some differences could be luck. So I re-ran every model on "
            f"<b>{M['attribution']['n_boot']:,} random resamples</b> of the journeys and kept a 95% range for each "
            "channel. A filled dot means the range doesn't cross zero, so the difference is real. A hollow dot means "
            "it could be noise.",
            "I only call a channel over- or under-valued when <b>both</b> data-driven models agree and both are filled.")
    lt = D["results"][D["results"].model == "last_touch"].set_index("channel")
    fig = viz.diff_ci_plot(D["diffs"], lt["revenue"], "revenue")
    fig.update_layout(title=None, margin=dict(t=40))
    st.plotly_chart(fig, **CHART)
    rv = D["bias"][D["bias"].metric == "revenue"].set_index("channel").verdict
    robust = sorted(rv[rv.str.endswith("by last touch")].index)
    guard(robust == ["Direct", "Organic Search"], "exactly Direct and Organic Search have robust revenue verdicts")
    big = [c for c in ["Referral", "Paid Search", "Email"] if c in rv.index]
    disagree = [c for c in big if rv[c].startswith("mixed")]
    nodiff = [c for c in big if rv[c] == "no significant difference"]
    unsure = (f" On revenue, the models disagree for {' and '.join(disagree)}" if disagree else "") + \
             (f"{' and' if disagree else ' On revenue,'} show no real difference for {' and '.join(nodiff)}" if nodiff else "")
    takeaway(f"Two findings survive this test on revenue: <b>Organic Search is under-credited</b> and <b>Direct is "
             f"over-credited</b>. On purchase counts, <b>Paid Search is under-credited by about "
             f"{(ps.markov_pct_change + ps.shapley_pct_change) / 2:.0f}%</b>.{unsure}, so for those the honest answer "
             "is “we can't tell yet.”")

    # 5. budget
    chapter(5, "So should we move budget to the under-valued channels?",
            f"This is where it gets interesting. The textbook move is to shift effort from the most over-credited "
            f"channel to the most under-credited one: here, <b>{b['from_channel']} → {b['to_channel']}</b>. "
            f"Moving {pct(b['shift_pct'], 0)} of {b['from_channel']}'s visits would change revenue by "
            f"<b>{viz.money(base.delta_revenue, True)}</b> over {p['n_days']} days "
            f"(95% range {viz.money(base.ci_low, True)} to {viz.money(base.ci_high, True)}).",
            f"Why? <b>Being under-valued isn't the same as being efficient.</b> Even after the correction, each "
            f"{b['from_channel']} visit is worth {viz.money(vps[b['from_channel']], decimals=2)}, while "
            f"{b['to_channel']} is worth {viz.money(vps[b['to_channel']], decimals=2)} and Paid Search only "
            f"{viz.money(vps['Paid Search'], decimals=2)}.")
    st.plotly_chart(viz.value_per_session_bar(vps, [b["from_channel"], b["to_channel"]]), **CHART)
    takeaway("Use attribution to <b>judge</b> channels fairly, not to move money automatically. Budget decisions "
             "need cost data and a holdout test. Try other moves in the <b>Budget simulator</b>.", warn=True)

    # 6. funnel
    chapter(6, "Where is the biggest opportunity?",
            f"Of {f['sessions']:,} visits, only {f['purchase_sessions_closed']:,} complete every step from product page "
            f"to purchase in the same visit. The biggest leak isn't checkout: only "
            f"<b>{pct(f['weakest_post_view_rate'])}</b> of visits that view a product add it to the cart "
            f"({f['weakest_post_view_lost']:,} visits lost). Once checkout starts, "
            f"{pct(f['step_rates']['Purchase'], 0)} of shoppers who enter payment details buy.",
            f"It isn't a mobile problem either: desktop ({pct(f['desktop_step_rate'])}) and mobile "
            f"({pct(f['mobile_step_rate'])}) add to cart at the same rate (p = {f['device_p']:.2f}).")
    fig = viz.funnel_chart(D["funnel"])
    fig.update_layout(title=None, margin=dict(t=20))
    st.plotly_chart(fig, **CHART)
    takeaway("Test the <b>product page</b> (pricing and shipping clarity, stock and size info, the add-to-cart button) "
             "before redesigning checkout or the mobile site.")

    # recommendations
    st.html("<div class='chapter'><div class='num'>What to do</div><h2>Recommendations</h2></div>")
    cards = "".join(f"<div class='rec'><div class='n'>{i}</div><div class='t'>{esc(x['title'])}</div>"
                    f"<div class='a'>{esc(x['action'])}</div></div>" for i, x in enumerate(FIND, 1))
    st.html(f"<div class='recs'>{cards}</div>")

    with st.expander("How sure are we? Limitations of this analysis"):
        for x in D["limitations"]:
            st.markdown("- " + x.replace("$", "\\$"))
    footer()


# --------------------------------------------------------------------------- explore pages

def attribution_explorer() -> None:
    page_title("Attribution explorer", "All eight models side by side. Each one is implemented from scratch in "
               "<code>src/mta/attribution.py</code> and unit-tested against hand-worked examples.")
    res = D["results"]
    t1, t2, t3 = st.tabs(["Compare models", "Over- or under-credited?", "Confidence intervals"])
    with t1:
        a, b = st.columns(2)
        model = a.selectbox("Model", list(MODEL_LABELS), index=6, format_func=MODEL_LABELS.get)
        metric = b.segmented_control("Metric", ["conversions", "revenue"], default="revenue",
                                     format_func=str.capitalize, key="m1") or "revenue"
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.model_bars(res, model, metric, MODEL_LABELS[model]), **CHART)
        c2.plotly_chart(viz.model_bars(res, "last_touch", metric, "Last touch (baseline)"), **CHART)
        st.plotly_chart(viz.credit_heatmap(res, MODEL_LABELS), **CHART)
        with st.expander("How each model shares the credit"):
            st.markdown(
                "- **First / last touch:** all credit to the first or last visit.\n"
                "- **Last non-direct:** last visit that wasn't Direct (Google Analytics' old default).\n"
                "- **Linear:** equal credit to every visit.\n"
                "- **Time decay:** a visit 7 days before the purchase gets half the credit of one on the day.\n"
                "- **Position-based:** 40% first visit, 40% last, 20% shared by the middle.\n"
                "- **Markov:** how much the overall conversion rate drops if a channel is removed from all journeys.\n"
                "- **Shapley:** game-theory fair share across every combination of channels (computed exactly).")
    with t2:
        metric = st.segmented_control("Metric", ["conversions", "revenue"], default="revenue",
                                      format_func=str.capitalize, key="m2") or "revenue"
        st.plotly_chart(viz.dumbbell(D["bias"], metric), **CHART)
        st.plotly_chart(viz.pct_change_heatmap(D["changes"][D["changes"].model.isin(["markov", "shapley"])]), **CHART)
        bias = D["bias"][D["bias"].metric == metric]
        fmt = (lambda v: viz.money(v)) if metric == "revenue" else (lambda v: f"{v:,.1f}")
        st.dataframe(pd.DataFrame({
            "Channel": bias.channel, "Last touch": bias.last_touch.map(fmt), "Markov": bias.markov.map(fmt),
            "Shapley": bias.shapley.map(fmt),
            "Markov vs LT": bias.markov_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
            "Shapley vs LT": bias.shapley_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
            "Verdict": bias.verdict}), width="stretch", hide_index=True)
    with t3:
        metric = st.segmented_control("Metric", ["conversions", "revenue"], default="revenue",
                                      format_func=str.capitalize, key="m3") or "revenue"
        lt = res[res.model == "last_touch"].set_index("channel")[metric]
        st.plotly_chart(viz.diff_ci_plot(D["diffs"], lt, metric), **CHART)
        st.plotly_chart(viz.ci_plot(D["ci"], metric), **CHART)
        d = D["diffs"][D["diffs"].metric == metric].copy()
        d["95% CI"] = [f"{lo:,.1f} to {hi:,.1f}" for lo, hi in zip(d.ci_low, d.ci_high)]
        st.dataframe(d[["comparison", "channel", "estimate", "95% CI", "significant"]].round(1),
                     width="stretch", hide_index=True)
    footer()


def budget() -> None:
    b = M["budget"]
    page_title("Budget simulator", "Pick a channel to take effort from, one to give it to, and how much. "
               "<b>Directional, not a forecast:</b> GA4 has no spend data, so effort is counted in visits, and each "
               "visit is worth its channel's credited revenue per visit. <i>k</i> is how valuable the extra visits "
               "are (1 = as valuable as today's, below 1 = diminishing returns).")
    chans = [c for c in viz.ADDRESSABLE if c in b["rev_per_session_markov"]]
    a, c, d = st.columns(3)
    src = a.selectbox("Move effort from", chans, index=chans.index(b["from_channel"]))
    dests = [x for x in chans if x != src]
    dst = c.selectbox("to", dests, index=dests.index(b["to_channel"]) if b["to_channel"] in dests else 0)
    basis = d.selectbox("Value visits with", ["markov", "shapley", "last_touch"], format_func=MODEL_LABELS.get)
    s1, s2 = st.columns(2)
    shift = s1.slider("Share of the source channel's visits moved", 0, 50, int(b["shift_pct"] * 100), format="%d%%") / 100
    k = s2.slider("Value of added visits (k)", 0.1, 1.5, 1.0, 0.05)

    sessions = D["daily"].groupby("channel")["sessions"].sum()
    rev = D["results"][D["results"].model == basis].set_index("channel")["revenue"]
    vps = rev / sessions
    moved = shift * sessions[src]
    delta = moved * (k * vps[dst] - vps[src])
    bw = D["boot"][D["boot"].model == basis].pivot(index="replicate", columns="channel", values="revenue")
    dist = moved * (k * bw[dst] / sessions[dst] - bw[src] / sessions[src])
    lo, hi = np.percentile(dist, [2.5, 97.5])

    m = st.columns(4)
    m[0].metric("Visits moved", f"{moved:,.0f}")
    m[1].metric(f"{src}: value per visit", viz.money(vps[src], decimals=2))
    m[2].metric(f"{dst}: value per visit", viz.money(vps[dst], decimals=2))
    m[3].metric(f"Revenue change ({M['profile']['n_days']} days)", viz.money(delta, signed=True))
    sure = "The whole range is on one side of zero, so the direction is clear." if (lo > 0 or hi < 0) \
        else "The range includes zero, so even the direction is uncertain."
    takeaway(f"95% range: <b>{viz.money(lo, True)} to {viz.money(hi, True)}</b>. {sure}", warn=delta < 0)

    fig = go.Figure(go.Histogram(x=dist, nbinsx=40, marker_color="#2a78d6", opacity=.85,
                                 hovertemplate="%{x:$,.0f}: %{y} resamples<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#52514e", width=1.5, dash="dot"))
    fig.update_xaxes(title="Revenue change across 1,000 bootstrap resamples", tickformat="$,.0f")
    fig.update_yaxes(title="Resamples")
    st.plotly_chart(viz.style(fig, None, 300), **CHART)

    vt = pd.DataFrame({"Markov $/visit": pd.Series(b["rev_per_session_markov"]),
                       "Last touch $/visit": pd.Series(b["rev_per_session_last_touch"]),
                       "Visits": sessions}).reindex([c for c in viz.CHANNEL_ORDER if c in sessions.index])
    with st.expander("Credited revenue per visit, all channels"):
        st.dataframe(vt.round(2), width="stretch")
    st.caption("Assumes visits are interchangeable units of effort, no cost differences between channels, and no "
               "capacity limits (e.g. email list size). Direct and Other/Unknown are not budget levers.")
    footer()


def funnel_journeys() -> None:
    f, j = M["funnel"], M["journeys"]
    page_title("Funnel & journeys", "Where shoppers drop out, and the paths they take before buying.")
    t1, t2 = st.tabs(["Funnel", "Journeys"])
    with t1:
        c = st.columns(4)
        c[0].metric("Visits", f"{f['sessions']:,}")
        c[1].metric("Full-funnel purchases", f"{f['purchase_sessions_closed']:,}")
        c[2].metric("All purchase visits", f"{f['purchase_sessions_open']:,}",
                    help="Includes purchases whose cart was built in an earlier visit")
        c[3].metric(f"{f['device_test_step']}: desktop vs mobile",
                    f"{pct(f['desktop_step_rate'])} vs {pct(f['mobile_step_rate'])}", help=f"z-test p = {f['device_p']:.2f}")
        st.plotly_chart(viz.funnel_chart(D["funnel"]), **CHART)
        dim = st.segmented_control("Break down by", ["channel", "device_category"], default="channel",
                                   format_func=lambda x: "Channel" if x == "channel" else "Device") or "channel"
        src = D["funnel_channel"] if dim == "channel" else D["funnel_device"]
        st.plotly_chart(viz.step_rate_heatmap(src[src.step_order > 1], dim), **CHART)
    with t2:
        c = st.columns(4)
        c[0].metric("Buying journeys", f"{j['converting_journeys']:,}")
        c[1].metric("Single visit", pct(j["single_touch_share"]))
        c[2].metric("Median time to buy", f"{j['median_days_to_conversion'] * 24:,.1f} hours")
        c[3].metric("90th percentile", f"{j['p90_days_to_conversion']:,.1f} days")
        top_n = st.slider("Paths shown", 5, 30, 15)
        st.plotly_chart(viz.sankey(D["paths"], top_n=top_n), **CHART)
        a, b = st.columns(2)
        a.plotly_chart(viz.path_length_chart(D["path_length"]), **CHART)
        b.plotly_chart(viz.time_to_conversion_chart(D["ttc"]), **CHART)
        t = D["paths"].head(25).copy()
        t["share"] = t["share"].map(pct)
        t["revenue"] = t["revenue"].map(viz.money)
        st.dataframe(t, width="stretch", hide_index=True)
    footer()


def data_method() -> None:
    p, o = M["profile"], M["overview"]
    page_title("Data & method", f"Public GA4 export of the Google Merchandise Store, {p['min_date']} to {p['max_date']}, "
               "transformed in BigQuery with dbt and modeled in Python.")
    c = st.columns(4)
    c[0].metric("Events", f"{p['events']:,}")
    c[1].metric("Users", f"{p['users']:,}")
    c[2].metric("Visits", f"{p['sessions']:,}")
    c[3].metric("Duplicate purchases removed", f"{p['duplicate_purchase_events']:,}",
                help=f"{viz.money(p['gross_revenue_usd'] - p['dedup_revenue_usd'])} of double-counted revenue".replace("$", "\\$"))

    st.subheader("Daily visits by channel")
    g = D["daily"].groupby(["date", "channel"], as_index=False)["sessions"].sum()
    fig = go.Figure()
    for ch in [c for c in viz.CHANNEL_ORDER if c in set(g.channel)]:
        s = g[g.channel == ch]
        fig.add_trace(go.Scatter(x=s.date, y=s.sessions, name=ch, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=viz.CHANNEL_COLORS[ch]),
                                 hovertemplate=ch + ": %{y:,.0f}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%b %d")
    st.plotly_chart(viz.style(fig, None, 380), **CHART)

    st.subheader("Channel summary")
    ch = pd.DataFrame(o["channels"]).set_index("channel")
    ch = ch.reindex([c for c in viz.CHANNEL_ORDER if c in ch.index])
    st.dataframe(pd.DataFrame({
        "Visits": ch.sessions.map("{:,.0f}".format), "Share": ch.share.map(pct),
        "Buying visits": ch.converting_sessions.map("{:,.0f}".format),
        "Conversion rate": ch.session_cr.map(lambda v: pct(v, 2)), "Revenue (last click)": ch.revenue_usd.map(viz.money),
        "No source (inferred Direct)": ch.inferred_sessions.map("{:,.0f}".format)}), width="stretch")

    st.subheader("How it was built")
    st.markdown(
        "1. **dbt staging:** flatten GA4 `event_params`, remove double-fired purchase events.\n"
        "2. **Sessions:** one row per visit with landing page, device, country and source/medium.\n"
        "3. **Channels:** an ordered rules table (`dbt/seeds/channel_rules.csv`) maps source/medium to 8 channel groups.\n"
        "4. **Journeys:** each shopper's visits in the 30 days before a purchase.\n"
        "5. **Models:** 8 attribution models in Python, each with hand-calculated unit tests.\n"
        "6. **Uncertainty:** 1,000 bootstrap resamples of journeys for confidence intervals.\n"
        "7. **Quality gates:** dbt tests plus revenue reconciliation to raw events (within 0.5%); the build fails otherwise.")
    with st.expander("Channel rules and the source/medium values they matched"):
        st.dataframe(D["channel_map"], width="stretch", hide_index=True)
    footer()


nav = st.navigation({
    "Start here": [st.Page(story, title="The story", icon="📖", default=True)],
    "Explore": [
        st.Page(attribution_explorer, title="Attribution explorer", icon="⚖️", url_path="attribution"),
        st.Page(budget, title="Budget simulator", icon="💸", url_path="budget"),
        st.Page(funnel_journeys, title="Funnel & journeys", icon="🔻", url_path="funnel"),
        st.Page(data_method, title="Data & method", icon="🧰", url_path="data"),
    ],
})
nav.run()
