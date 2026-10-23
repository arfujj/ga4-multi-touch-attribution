"""Render README.md, docs/findings_memo.md and reports/report.html from pipeline outputs.

All numbers are read from reports/metrics.json and reports/*.csv. The narrative makes
qualitative claims (e.g. "under-credited", "would lose revenue"); each one is checked by
``guard`` against the data, so a re-run that contradicts the story fails loudly instead of
publishing stale prose.
Run: python -m mta.writeup
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from mta import viz
from mta.attribution import MODEL_LABELS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


class NarrativeError(AssertionError):
    """The data no longer supports a sentence in the write-up."""


def guard(condition: bool, claim: str) -> None:
    if not condition:
        raise NarrativeError(f"Write-up claim no longer supported by the data: {claim}")


def pct(x: float, d: int = 1, signed: bool = False) -> str:
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def money(x: float, signed: bool = False, decimals: int = 0) -> str:
    return viz.money(x, signed, decimals)


def scan_gb() -> float:
    """GB processed in the last pipeline run: Python queries (bq usage log) + dbt build (run_results)."""
    total = 0
    log = ROOT / "data" / "cache" / "bq_usage.jsonl"
    if log.exists():
        total += sum(json.loads(line)["bytes_processed"] for line in log.read_text().splitlines() if line.strip())
    rr = ROOT / "dbt" / "target" / "run_results.json"
    if rr.exists():
        for r in json.loads(rr.read_text())["results"]:
            total += int((r.get("adapter_response") or {}).get("bytes_processed") or 0)
    return total / 1e9


class Facts:
    """Every number the write-ups use, pulled from pipeline outputs."""

    def __init__(self) -> None:
        self.m = json.loads((REPORTS / "metrics.json").read_text())
        self.results = pd.read_csv(REPORTS / "attribution_results.csv")
        self.bias = pd.read_csv(REPORTS / "last_touch_bias.csv")
        self.sim = pd.read_csv(REPORTS / "budget_simulation.csv")
        self.paths = pd.read_csv(REPORTS / "converting_paths.csv")
        self.ch = pd.DataFrame(self.m["overview"]["channels"]).set_index("channel")
        self.p, self.o, self.j, self.f = self.m["profile"], self.m["overview"], self.m["journeys"], self.m["funnel"]
        self.b, self.a = self.m["budget"], self.m["attribution"]
        self.scan_gb = scan_gb()

    def credit(self, model: str, channel: str, metric: str = "conversions") -> float:
        r = self.results
        return float(r[(r.model == model) & (r.channel == channel)][metric].iloc[0])

    def share(self, model: str, channel: str) -> float:
        r = self.results
        return float(r[(r.model == model) & (r.channel == channel)]["share"].iloc[0])

    def verdict(self, channel: str, metric: str) -> pd.Series:
        b = self.bias
        return b[(b.channel == channel) & (b.metric == metric)].iloc[0]


# --------------------------------------------------------------------------- findings

def findings(F: Facts) -> list[dict[str, str]]:
    """The 5 headline findings: title, number, why it matters, action. Each claim is guarded."""
    out = []

    # 1. Organic Search under-credited on revenue
    org = F.verdict("Organic Search", "revenue")
    guard(org.verdict == "under-credited by last touch", "Organic Search revenue is under-credited by last touch")
    ft, lt = F.share("first_touch", "Organic Search"), F.share("last_touch", "Organic Search")
    guard(ft > lt, "Organic Search earns more first-touch than last-touch credit")
    out.append({
        "title": "Organic Search is worth more than last-click reporting shows",
        "number": (f"Both data-driven models give Organic Search more revenue credit than last touch: "
                   f"{money(org.last_touch)} → {money(org.markov)} (Markov, {org.markov_pct_change:+.1f}%) and "
                   f"{money(org.shapley)} (Shapley, {org.shapley_pct_change:+.1f}%); both 95% intervals exclude zero. "
                   f"It opens {pct(ft)} of converting journeys but closes only {pct(lt)}."),
        "why": "Search often starts the journey and another channel closes it, so last-click "
               "reports undervalue SEO and content work.",
        "action": "Judge SEO and content on data-driven or position-based credit, and don't cut organic "
                  "investment based on last-click numbers.",
    })

    # 2. Direct over-credited + measurement gap
    d = F.verdict("Direct", "revenue")
    guard(d.verdict == "over-credited by last touch", "Direct revenue is over-credited by last touch")
    inferred_of_direct = F.ch.at["Direct", "inferred_sessions"] / F.ch.at["Direct", "sessions"]
    lnd, ltd = F.share("last_non_direct", "Direct"), F.share("last_touch", "Direct")
    guard(0.4 <= inferred_of_direct <= 0.6, "about half of Direct sessions have no source information")
    out.append({
        "title": "Direct gets too much credit, and half of it is really missing tracking",
        "number": (f"Direct loses revenue credit under both data-driven models ({d.markov_pct_change:+.1f}% Markov, "
                   f"{d.shapley_pct_change:+.1f}% Shapley; both significant). {pct(inferred_of_direct, 0)} of Direct's "
                   f"{F.ch.at['Direct', 'sessions']:,.0f} sessions carry no source information at all, and under "
                   f"last non-direct click its share of conversions falls from {pct(ltd)} to {pct(lnd)}."),
        "why": "\"Direct\" is mostly people returning after another channel found them, or visits whose source "
               "was lost. It is an outcome, not a lever you can buy.",
        "action": "Fix tracking first: UTM-tag every email and partner link, configure cross-domain measurement "
                  "and unwanted-referral exclusions, then re-run this analysis.",
    })

    # 3. Paid Search under-credited in conversions but low value per session
    ps = F.verdict("Paid Search", "conversions")
    guard(ps.verdict == "under-credited by last touch", "Paid Search conversions are under-credited by last touch")
    conv_under = F.bias[(F.bias.metric == "conversions") & (F.bias.verdict == "under-credited by last touch")]
    guard(list(conv_under.channel) == ["Paid Search"], "Paid Search is the only channel under-credited on conversions by both models")
    guard(all(25 <= x <= 40 for x in (ps.markov_pct_change, ps.shapley_pct_change)), "the Paid Search correction is about a third")
    vps = pd.Series(F.b["rev_per_session_markov"])
    big = vps[[c for c in ["Organic Search", "Paid Search", "Referral", "Email"] if c in vps.index]]
    guard(big.idxmin() == "Paid Search", "Paid Search has the lowest data-driven revenue per session among volume channels")
    out.append({
        "title": "Paid Search assists more sales than it gets credit for, but each visit is worth the least",
        "number": (f"Paid Search is the only channel both data-driven models agree last touch under-credits on "
                   f"conversions: {ps.last_touch:.0f} → {ps.markov:.1f} (Markov, {ps.markov_pct_change:+.1f}%) and "
                   f"{ps.shapley:.1f} (Shapley, {ps.shapley_pct_change:+.1f}%). Even so, it earns "
                   f"{money(vps['Paid Search'], decimals=2)}/session vs {money(vps['Organic Search'], decimals=2)} for Organic Search and "
                   f"{money(vps['Referral'], decimals=2)} for Referral (Markov)."),
        "why": "Last-click ROAS will understate paid search by about a third, but the correction doesn't make "
               "it the most efficient channel.",
        "action": "Report paid search on data-driven conversions, then compare cost per data-driven conversion "
                  "to the other channels before changing its budget.",
    })

    # 4. Budget reallocation
    base = F.sim.iloc[0]
    guard(base.ci_high < 0, "the default reallocation reduces revenue with a CI excluding zero")
    guard(vps[F.b["from_channel"]] > vps[F.b["to_channel"]], "source channel earns more per session than destination")
    out.append({
        "title": "Moving budget toward the \"under-credited\" channel would lose money here",
        "number": (f"Shifting {pct(F.b['shift_pct'], 0)} of {F.b['from_channel']} effort "
                   f"({F.b['sessions_moved']:.0f} sessions) to {F.b['to_channel']} changes data-driven revenue by "
                   f"{money(base.delta_revenue, True)} over 92 days (95% CI {money(base.ci_low, True)} to "
                   f"{money(base.ci_high, True)}), even with constant returns. {F.b['from_channel']} still earns "
                   f"{money(vps[F.b['from_channel']], decimals=2)}/session vs {money(vps[F.b['to_channel']], decimals=2)}."),
        "why": "Being under-credited is about how a channel is measured, not how efficient it is. Budget "
               "decisions need value per unit of spend, and GA4 has no cost data.",
        "action": f"Protect {F.b['from_channel']}. Use the attribution results to re-rank channels, then make "
                  "reallocation decisions with cost data and a holdout test.",
    })

    # 5. Funnel
    f = F.f
    guard(f["device_p"] > 0.05, "the add-to-cart rate does not differ significantly between desktop and mobile")
    guard(f["weakest_post_view_step"] == "Add to cart", "add to cart is the weakest step after product view")
    out.append({
        "title": "The biggest leak is the product page, not checkout, and it isn't a mobile problem",
        "number": (f"Only {pct(f['weakest_post_view_rate'])} of sessions that view a product add it to the cart "
                   f"({f['weakest_post_view_lost']:,} of {f['weakest_post_view_prev']:,} sessions drop). Once checkout "
                   f"starts, {pct(f['step_rates']['Add payment info'])} add payment info and "
                   f"{pct(f['step_rates']['Purchase'])} of those buy. Desktop ({pct(f['desktop_step_rate'])}) and "
                   f"mobile ({pct(f['mobile_step_rate'])}) add to cart at the same rate (p = {f['device_p']:.2f})."),
        "why": "Effort spent on checkout UX or a mobile redesign would target the wrong step.",
        "action": "A/B test the product page: price and shipping clarity, stock and size information, and the "
                  "prominence of the add-to-cart button.",
    })
    return out


def limitations(F: Facts) -> list[str]:
    return [
        f"**Obfuscated sample data.** {pct(F.o['inferred_direct_share'])} of sessions have no source and are treated "
        f"as Direct (GA4's convention), and {pct(F.o['other_unknown_share'])} fall in Other/Unknown. Google also "
        "masks some values as `<Other>` or `(data deleted)`.",
        f"**Short window.** {F.p['n_days']} days ({F.p['min_date']} to {F.p['max_date']}), spanning the holidays. Journeys "
        "that started before November are cut off, and users who would have bought after January are counted as "
        "non-converting.",
        "**Attribution is correlational.** Every model, including the data-driven ones, redistributes credit for "
        "sales that happened. None measures what would have happened without a channel. Only holdout or geo "
        "experiments can show incrementality.",
        "**No cost or cross-device data.** Effort is proxied by sessions, and journeys are per browser "
        "(`user_pseudo_id`), so people who switch devices look like separate users.",
    ]


# --------------------------------------------------------------------------- memo

def memo(F: Facts) -> str:
    fs = findings(F)
    body = "\n\n".join(
        f"### {i}. {x['title']}\n**The number:** {x['number']}  \n**Why it matters:** {x['why']}  \n"
        f"**Recommended action:** {x['action']}" for i, x in enumerate(fs, 1))
    lim = "\n".join(f"- {x}" for x in limitations(F))
    return f"""# Findings memo: which channels drive purchases?

**To:** Marketing Director · **Re:** Google Merchandise Store, {F.p['min_date']} to {F.p['max_date']}

**Bottom line:** {F.o['purchases']:,} purchases worth {money(F.o['revenue_usd'])} came from {F.j['converting_journeys']:,}
customer journeys. {pct(F.j['multi_channel_share'], 0)} of them touched more than one channel. For those
journeys, the credit each channel gets depends on the attribution model, and last-click reporting
undervalues the channels that start journeys. The clearest gains are in tracking and the product page,
not in shifting budget.

{body}

### Limitations
{lim}

_Every number in this memo is generated from the pipeline outputs by `src/mta/writeup.py`._
"""


# --------------------------------------------------------------------------- README

def readme(F: Facts) -> str:
    fs = findings(F)
    key = "\n".join(f"{i}. **{x['title']}.** {x['number']}" for i, x in enumerate(fs, 1))
    r = F.results
    tbl = r.pivot(index="channel", columns="model", values="conversions")
    tbl = tbl.reindex([c for c in viz.CHANNEL_ORDER if c in tbl.index])[list(MODEL_LABELS)]
    tbl.columns = [MODEL_LABELS[c] for c in tbl.columns]
    tbl_md = tbl.to_markdown(floatfmt=",.1f")
    rv = F.bias[F.bias.metric == "revenue"].set_index("channel").reindex(
        [c for c in viz.CHANNEL_ORDER if c in set(F.bias.channel)])
    rv_md = pd.DataFrame({
        "Last touch": rv.last_touch.map(money), "Markov": rv.markov.map(money), "Shapley": rv.shapley.map(money),
        "Markov vs LT": rv.markov_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
        "Shapley vs LT": rv.shapley_pct_change.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%"),
        "Verdict (95% bootstrap)": rv.verdict,
    }).to_markdown()
    p = F.p
    dup = p["gross_revenue_usd"] - p["dedup_revenue_usd"]
    return f"""# GA4 Multi-Touch Attribution Lab

**Which marketing channels actually drive purchases, how much does the answer change with the attribution
model, and where should budget move?** This project answers that for the Google Merchandise Store using
{p['n_days']} days of raw GA4 event data ({p['events']:,} events, {p['sessions']:,} sessions). BigQuery and dbt
turn the events into sessions, channels and {F.j['journeys']:,} 30-day user journeys. Eight attribution models
are implemented from scratch in Python, including Markov-chain removal effects and exact Shapley values,
and compared with {F.a['n_boot']:,}-resample bootstrap confidence intervals. The results feed a funnel
analysis, a budget-reallocation simulator, a Streamlit dashboard and a one-page memo for a marketing
director. Every number below is generated by the code in this repo.

## Key findings

{key}

Full write-up for a non-technical audience: [`docs/findings_memo.md`](docs/findings_memo.md) ·
Data profile: [`docs/data_profile.md`](docs/data_profile.md) · Static report: [`reports/report.html`](reports/report.html)

### Credited conversions by model

{tbl_md}

### Revenue credit: data-driven vs last touch

{rv_md}

![Share of conversion credit by model](reports/figures/credit_heatmap.png)
![Revenue credit vs last touch with 95% CIs](reports/figures/ci_vs_last_touch_revenue.png)

## Architecture

```mermaid
flowchart LR
    A[(BigQuery public data<br/>ga4_obfuscated_sample_ecommerce<br/>events_*)] -->|_TABLE_SUFFIX filter,<br/>dry-run cost guard| B[stg_ga4__events<br/>flatten event_params,<br/>dedupe purchases]
    S[/seeds/channel_rules.csv/] --> D
    B --> C[int_sessions<br/>one row per session]
    C --> D[int_channel_map<br/>source/medium → channel]
    C --> E[fct_sessions]
    D --> E
    B --> P[fct_purchases]
    E --> J[fct_journey_touches<br/>fct_journeys<br/>30-day lookback]
    E --> F[fct_funnel<br/>channel × device]
    J & F & E -->|mta.extract| L[(local parquet cache)]
    L --> M[mta.attribution<br/>8 models + bootstrap]
    M --> R[reports/*.csv<br/>metrics.json, figures]
    R --> W[README · memo · report.html<br/>mta.writeup]
    R --> SD[Streamlit dashboard]
    T{{dbt tests + reconciliation<br/>pytest}} -.-> B & E & J & M
```

## Dashboard

A Streamlit app that opens on **"The story"**: six short chapters for a non-technical reader (what we can
measure, how people buy, what last click gets wrong, whether it's real, what to do with budget, where the
biggest opportunity is), followed by recommendations. Explore pages hold the detail: an attribution explorer
with all 8 models and confidence intervals, an interactive budget simulator, funnel and journey breakdowns,
and the data and method.

| The story | Attribution explorer |
|---|---|
| ![Story](docs/screenshots/story.png) | ![Attribution](docs/screenshots/attribution.png) |
| **Budget simulator** | **Funnel & journeys** |
| ![Budget](docs/screenshots/budget.png) | ![Funnel](docs/screenshots/funnel.png) |

## How to run

Prerequisites: Python 3.11, a Google Cloud project with BigQuery enabled (the free Sandbox works), and the
`gcloud` CLI.

```bash
gcloud auth application-default login     # credentials via ADC, no key files
cp .env.example .env                       # set GCP_PROJECT_ID
make setup                                 # venv + pinned requirements
make all                                   # profile → dbt build (+tests) → extract → models → write-ups
make dashboard                             # http://localhost:8501 (reads only reports/, no GCP needed)
```

The last full `make all` processed {F.scan_gb:,.1f} GB in BigQuery, measured from job statistics (the free tier is 1 TB/month). Every
query is dry-run first and refused above a per-query byte cap. Other targets: `make test` (pytest + dbt tests),
`make dryrun` (bytes per compiled model), `make screenshots`.

> **BigQuery Sandbox note:** the Sandbox expires tables after 60 days and partitions dated more than 60 days ago
> immediately, which is why `stg_ga4__events` is clustered, not date-partitioned. Re-running `make all` rebuilds
> everything if the tables have expired.

## Method notes

- **Sessions and channels.** A session's source/medium is taken from the first event in that session
  that carries the `source`/`medium` event params. The `traffic_source` struct is user-level first touch, so it isn't used.
  Channels come from an ordered regex rules table (`dbt/seeds/channel_rules.csv`) with a rationale for every rule.
  {pct(p['sessions_without_source_params'] / p['sessions'])} of sessions carry no source params and follow GA4's
  `(direct)` convention, flagged `source_inferred`. Self-referrals from the store's own domains are treated as Direct.
- **Journeys.** These are a user's sessions in the 30 days before a purchase (lookback is a dbt var). Sessions after a purchase
  start a new journey, and journeys without a purchase end at the user's last session.
- **Models.** First touch, last touch, last non-direct, linear, time decay (7-day half-life, parameterised),
  position-based 40/20/40, **Markov** (first-order, removal effects solved exactly from the absorbing chain),
  and **Shapley** (exact 2ⁿ enumeration, v(S) = conversions from journeys whose channel set ⊆ S).
  Every model has a hand-calculated unit test.
- **Uncertainty.** {F.a['n_boot']:,} bootstrap resamples of journeys, implemented as multinomial reweighting of unique
  (path, outcome, revenue) groups, which is exactly equivalent to resampling journeys and runs in seconds. Last touch is
  re-fit on each replicate, so "data-driven minus last touch" gets a paired interval. A channel is called over- or
  under-credited only if **both** data-driven models differ significantly in the same direction.
- **Data quality.** dbt tests (not_null, unique, accepted_values, relationships) plus custom checks: revenue
  reconciles between raw events and three modeled tables within 0.5%, no negative session durations, and
  purchase counts match across layers. The build fails on any of them. {p['duplicate_purchase_events']} duplicate
  purchase events (a double-firing tag, {money(dup)} or {pct(dup / p['gross_revenue_usd'])} of gross revenue) are
  removed, and the reconciliation recomputes that rule directly from raw events.

## Repository layout

```
dbt/        models (staging → intermediate → marts), seeds/channel_rules.csv, tests/ (custom checks)
src/mta/    bq.py (cost-guarded client) · profile.py · extract.py · attribution.py · analysis.py · viz.py · writeup.py
tests/      pytest unit tests for every attribution model
dashboard/  Streamlit app (6 pages)
docs/       data_profile.md · findings_memo.md · screenshots/
reports/    CSV outputs, metrics.json, figures/, report.html
```

## What I'd do next

1. **Incrementality.** Run a geo or audience holdout for Paid Search and Email to check whether the
   data-driven credit matches causal lift.
2. **Add cost data** (Google Ads, ESP exports) so the simulator optimises revenue per dollar instead of per session,
   with channel saturation curves in place of the constant k factor.
3. **Higher-order Markov and a learned-value Shapley** (conversion-probability characteristic function) to test
   how sensitive the conclusions are to model assumptions.
4. **Sensitivity of the Direct assumption.** Re-run with parameter-less sessions as Unknown instead of Direct.
5. **Schedule it.** Run as an incremental dbt model on a live GA4 export, with Airflow/Cloud Scheduler and data
   freshness tests.
"""


# --------------------------------------------------------------------------- static report

def report_html(F: Facts) -> str:
    import plotly.io as pio

    rd = lambda n: pd.read_csv(REPORTS / n)  # noqa: E731
    res, diffs = F.results, rd("bootstrap_differences.csv")
    lt = res[res.model == "last_touch"].set_index("channel")
    figs = [
        viz.credit_heatmap(res, MODEL_LABELS),
        viz.pct_change_heatmap(rd("model_comparison.csv").query("model in ['markov', 'shapley']")),
        viz.diff_ci_plot(diffs, lt.revenue, "revenue"),
        viz.diff_ci_plot(diffs, lt.conversions, "conversions"),
        viz.sankey(F.paths),
        viz.path_length_chart(rd("path_length.csv")),
        viz.time_to_conversion_chart(rd("time_to_conversion.csv")),
        viz.funnel_chart(rd("funnel_overall.csv")),
        viz.step_rate_heatmap(rd("funnel_by_channel.csv").query("step_order > 1"), "channel"),
        viz.budget_chart(F.sim),
    ]
    charts = "\n".join(
        f"<section class='card'>{pio.to_html(fig, include_plotlyjs='cdn' if i == 0 else False, full_html=False, config={'displaylogo': False})}</section>"
        for i, fig in enumerate(figs))
    kpis = [("Sessions", f"{F.o['sessions']:,}"), ("Users", f"{F.o['users']:,}"), ("Purchases", f"{F.o['purchases']:,}"),
            ("Revenue", money(F.o["revenue_usd"])), ("Session CR", pct(F.o["session_cr"], 2)),
            ("Multi-channel journeys", pct(F.j["multi_channel_share"]))]
    kp = "".join(f"<div class='kpi'><div class='k'>{html.escape(k)}</div><div class='v'>{v}</div></div>" for k, v in kpis)
    fl = "".join(f"<li><b>{html.escape(x['title'])}.</b> {html.escape(x['number'])} <i>Action:</i> {html.escape(x['action'])}</li>"
                 for x in findings(F))
    lim = "".join(f"<li>{html.escape(x).replace('**', '')}</li>" for x in limitations(F))
    tbl = res.pivot(index="channel", columns="model_label", values="conversions").round(1).to_html(classes="tbl")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GA4 Attribution Report</title>
<style>
:root{{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,-apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}} h1{{font-size:30px;margin:0 0 4px}} .sub{{color:var(--ink2)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}} .k{{color:var(--ink2);font-size:13px}} .v{{font-size:22px;font-weight:600}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px;margin:16px 0;overflow-x:auto}}
li{{margin:8px 0}} .tbl{{border-collapse:collapse;font-size:13px}} .tbl td,.tbl th{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}}
</style></head><body><main>
<h1>GA4 Multi-Touch Attribution Lab</h1>
<div class="sub">Google Merchandise Store · {F.p['min_date']} to {F.p['max_date']} · generated by <code>src/mta/writeup.py</code></div>
<div class="kpis">{kp}</div>
<h2>Key findings</h2><ol>{fl}</ol>
<h2>Charts</h2>{charts}
<h2>Credited conversions by model</h2><section class="card">{tbl}</section>
<h2>Limitations</h2><ul>{lim}</ul>
</main></body></html>"""


def main() -> None:
    F = Facts()
    (ROOT / "docs" / "findings_memo.md").write_text(memo(F))
    (ROOT / "README.md").write_text(readme(F))
    (REPORTS / "report.html").write_text(report_html(F))
    print("wrote README.md, docs/findings_memo.md, reports/report.html")


if __name__ == "__main__":
    main()
