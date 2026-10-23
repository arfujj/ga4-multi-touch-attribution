"""End-to-end analysis: journeys, funnel, 8 attribution models, comparison, bootstrap CIs
and the budget reallocation scenario. Reads data/cache/*.parquet (from mta.extract) and
writes reports/*.csv, reports/metrics.json and reports/figures/*.png.

Every number used in the README, memo and dashboard comes from reports/metrics.json or
the CSVs written here.
Run: python -m mta.analysis [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from mta import attribution as A
from mta import viz

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

STEP_LABELS = {
    "session_start": "Session start", "view_item": "View item", "add_to_cart": "Add to cart",
    "begin_checkout": "Begin checkout", "add_payment_info": "Add payment info", "purchase": "Purchase",
}
ADDRESSABLE = viz.ADDRESSABLE
MIN_CONVERSIONS = 30  # minimum last-touch or data-driven conversions to be a reallocation candidate
SHIFT_PCT = 0.10
RETURNS_SCENARIOS = {"Constant returns (k=1.0)": 1.0, "Moderate diminishing (k=0.75)": 0.75, "Strong diminishing (k=0.5)": 0.5}


# --------------------------------------------------------------------------- loading

def load_journeys() -> pd.DataFrame:
    """Journey table in the shape the attribution module expects."""
    t = pd.read_parquet(CACHE / "journey_touches.parquet").sort_values(["journey_id", "touch_number"])
    t["touch_ts"] = pd.to_datetime(t["touch_ts"], utc=True)
    t["conversion_ts"] = pd.to_datetime(t["conversion_ts"], utc=True)
    g = t.groupby("journey_id", sort=False)
    j = pd.DataFrame({
        "path": g["channel"].agg(list),
        "touch_ts": g["touch_ts"].agg(list),
        "converted": g["converted"].first().astype(bool),
        "conversion_ts": g["conversion_ts"].first(),
        "revenue": g["revenue_usd"].sum().astype(float),
    }).reset_index()
    return j


# --------------------------------------------------------------------------- journeys

def journey_stats(j: pd.DataFrame, journeys_mart: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conv = journeys_mart[journeys_mart.converted]
    n_conv = len(conv)
    buckets = conv["n_touches"].clip(upper=10).map(lambda v: "10+" if v >= 10 else str(v))
    order = [str(i) for i in range(1, 10)] + ["10+"]
    pl = buckets.value_counts().reindex(order, fill_value=0).rename("journeys").rename_axis("bucket").reset_index()
    pl["share"] = pl["journeys"] / n_conv

    d = conv["days_to_conversion"]
    edges = [-0.001, 1 / 24, 1, 7, 14, 30.001]
    names = ["< 1 hour", "1 h to 1 day", "1 to 7 days", "7 to 14 days", "14 to 30 days"]
    ttc = pd.cut(d, edges, labels=names).value_counts().reindex(names, fill_value=0)
    ttc = ttc.rename("journeys").rename_axis("bucket").reset_index()
    ttc["share"] = ttc["journeys"] / n_conv

    paths = (conv.groupby("path").agg(journeys=("journey_id", "size"), revenue=("revenue_usd", "sum"))
             .sort_values("journeys", ascending=False).reset_index())
    paths["share"] = paths["journeys"] / n_conv
    paths["n_touches"] = paths["path"].str.count(" > ") + 1

    multi = conv[conv.n_distinct_channels > 1]
    stats = {
        "journeys": int(len(journeys_mart)),
        "converting_journeys": n_conv,
        "journey_conversion_rate": n_conv / len(journeys_mart),
        "single_touch_share": float((conv.n_touches == 1).mean()),
        "multi_touch_share": float((conv.n_touches > 1).mean()),
        "multi_channel_share": float(len(multi) / n_conv),
        "multi_channel_revenue_share": float(multi.revenue_usd.sum() / conv.revenue_usd.sum()),
        "mean_touches": float(conv.n_touches.mean()),
        "p90_touches": float(conv.n_touches.quantile(0.9)),
        "median_days_to_conversion": float(d.median()),
        "mean_days_to_conversion": float(d.mean()),
        "p75_days_to_conversion": float(d.quantile(0.75)),
        "p90_days_to_conversion": float(d.quantile(0.9)),
        "same_day_share": float((d < 1).mean()),
        "distinct_converting_paths": int(len(paths)),
        "top_path": paths.iloc[0]["path"], "top_path_share": float(paths.iloc[0]["share"]),
        # revenue sanity: journeys built in python equal the dbt mart
        "python_journeys_revenue": float(j.revenue.sum()),
    }
    return stats, pl, ttc, paths


# --------------------------------------------------------------------------- funnel

def _step_rates(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df = df.sort_values(by + ["step_order"]).copy()
    df["prev"] = df.groupby(by)["sessions"].shift(1) if by else df["sessions"].shift(1)
    df["step_rate"] = df["sessions"] / df["prev"]
    df["drop_off"] = df["prev"] - df["sessions"]
    df["step_label"] = df["step"].map(STEP_LABELS)
    return df


def funnel_analysis(f: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f = f.rename(columns={"sessions_closed": "sessions"})
    overall = f.groupby(["step_order", "step"], as_index=False)[["sessions", "sessions_open"]].sum()
    overall = _step_rates(overall, [])
    overall["overall_rate"] = overall["sessions"] / overall["sessions"].iloc[0]

    by_channel = _step_rates(f.groupby(["channel", "step_order", "step"], as_index=False)["sessions"].sum(), ["channel"])
    by_device = _step_rates(f.groupby(["device_category", "step_order", "step"], as_index=False)["sessions"].sum(), ["device_category"])

    later = overall[overall.step_order >= 2]
    worst_rate = later.loc[later.step_rate.idxmin()]
    worst_abs = later.loc[later.drop_off.idxmax()]
    post_view = overall[overall.step_order >= 3]
    worst_post_view = post_view.loc[post_view.step_rate.idxmin()]

    # Device comparison at the weakest post-product-view step (two-proportion z-test)
    step = worst_post_view["step"]
    dv = by_device[(by_device.step == step) & by_device.device_category.isin(["desktop", "mobile"])].set_index("device_category")
    z, p = proportions_ztest(dv["sessions"].to_numpy(), dv["prev"].to_numpy())
    ci = {d: proportion_confint(dv.at[d, "sessions"], dv.at[d, "prev"], method="wilson") for d in dv.index}

    stats = {
        "sessions": int(overall.sessions.iloc[0]),
        "purchase_sessions_closed": int(overall.sessions.iloc[-1]),
        "purchase_sessions_open": int(overall.sessions_open.iloc[-1]),
        "closed_funnel_cr": float(overall.overall_rate.iloc[-1]),
        "largest_dropoff_step": STEP_LABELS[worst_rate.step],
        "largest_dropoff_from": STEP_LABELS[overall.loc[overall.step_order == worst_rate.step_order - 1, "step"].iloc[0]],
        "largest_dropoff_rate": float(worst_rate.step_rate),
        "largest_dropoff_lost": int(worst_rate.drop_off),
        "largest_abs_dropoff_step": STEP_LABELS[worst_abs.step],
        "largest_abs_dropoff_lost": int(worst_abs.drop_off),
        "weakest_post_view_step": STEP_LABELS[worst_post_view.step],
        "weakest_post_view_from": STEP_LABELS[overall.loc[overall.step_order == worst_post_view.step_order - 1, "step"].iloc[0]],
        "weakest_post_view_rate": float(worst_post_view.step_rate),
        "weakest_post_view_lost": int(worst_post_view.drop_off),
        "weakest_post_view_prev": int(worst_post_view.prev),
        "device_test_step": STEP_LABELS[step],
        "desktop_step_rate": float(dv.at["desktop", "step_rate"]),
        "mobile_step_rate": float(dv.at["mobile", "step_rate"]),
        "desktop_step_ci": [float(x) for x in ci["desktop"]],
        "mobile_step_ci": [float(x) for x in ci["mobile"]],
        "device_z": float(z), "device_p": float(p),
        "step_rates": {STEP_LABELS[s]: float(r) for s, r in zip(overall.step, overall.step_rate) if pd.notna(r)},
    }
    return stats, overall, by_channel, by_device


# --------------------------------------------------------------------------- attribution

def attribution(j: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all models, compare to last touch, bootstrap the data-driven models."""
    results = A.run_all_models(j)
    tot = results.groupby("model")[["conversions", "revenue"]].transform("sum")
    results["share"] = results["conversions"] / tot["conversions"]
    results["revenue_share"] = results["revenue"] / tot["revenue"]
    results["model_label"] = results["model"].map(A.MODEL_LABELS)

    lt = results[results.model == "last_touch"].set_index("channel")
    changes = []
    for model in ["markov", "shapley", "first_touch", "last_non_direct", "linear", "time_decay", "position_based"]:
        m = results[results.model == model].set_index("channel")
        for ch in m.index:
            base = lt.at[ch, "conversions"]
            changes.append({
                "model": model, "channel": ch, "last_touch": base, "credit": m.at[ch, "conversions"],
                "abs_change": m.at[ch, "conversions"] - base,
                "pct_change": (m.at[ch, "conversions"] / base - 1) * 100 if base > 0 else np.nan,
                "last_touch_rev": lt.at[ch, "revenue"], "credit_rev": m.at[ch, "revenue"],
                "pct_change_rev": (m.at[ch, "revenue"] / lt.at[ch, "revenue"] - 1) * 100 if lt.at[ch, "revenue"] > 0 else np.nan,
            })
    changes = pd.DataFrame(changes)

    t0 = time.time()
    boot = A.bootstrap_data_driven(j, n_boot=n_boot, seed=seed)
    boot_secs = time.time() - t0

    rows = []
    point = results.set_index(["model", "channel"])
    for metric in ["conversions", "revenue"]:
        for (model, ch), s in boot.groupby(["model", "channel"])[metric]:
            lo, hi = np.percentile(s, [2.5, 97.5])
            rows.append({"metric": metric, "model": model, "channel": ch, "estimate": point.at[(model, ch), metric],
                         "ci_low": lo, "ci_high": hi, "boot_se": s.std(ddof=1)})
    ci = pd.DataFrame(rows)

    # Paired differences on the same replicates: data-driven minus last touch, and Markov minus Shapley.
    wide = boot.pivot_table(index=["replicate", "channel"], columns="model", values=["conversions", "revenue"])
    diffs = []
    for metric in ["conversions", "revenue"]:
        w = wide[metric]
        for a, b in [("markov", "last_touch"), ("shapley", "last_touch"), ("markov", "shapley")]:
            d = (w[a] - w[b]).groupby(level="channel")
            for ch, s in d:
                lo, hi = np.percentile(s, [2.5, 97.5])
                est = point.at[(a, ch), metric] - point.at[(b, ch), metric]
                diffs.append({"metric": metric, "comparison": f"{a} - {b}", "channel": ch, "estimate": est,
                              "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    diffs = pd.DataFrame(diffs)

    # Classification of last-touch bias per channel, for conversions and for revenue.
    # "Over/under-credited" requires BOTH data-driven models to differ significantly from
    # last touch in the same direction; anything else is reported as mixed / not significant.
    verdict = []
    for metric in ["conversions", "revenue"]:
        dc = diffs[diffs.metric == metric].set_index(["comparison", "channel"])
        for ch in sorted(results.channel.unique()):
            mk, sh = dc.loc[("markov - last_touch", ch)], dc.loc[("shapley - last_touch", ch)]
            same_sign = np.sign(mk.estimate) == np.sign(sh.estimate)
            if mk.significant and sh.significant and same_sign:
                v = "under-credited by last touch" if mk.estimate > 0 else "over-credited by last touch"
            elif mk.significant or sh.significant:
                v = "mixed: models disagree"
            else:
                v = "no significant difference"
            base = lt.at[ch, metric]
            pm, ps = point.at[("markov", ch), metric], point.at[("shapley", ch), metric]
            pct = lambda x: (x / base - 1) * 100 if base > 0 else np.nan  # noqa: E731
            verdict.append({
                "metric": metric, "channel": ch, "last_touch": base, "markov": pm, "shapley": ps,
                "markov_pct_change": pct(pm), "shapley_pct_change": pct(ps),
                "mean_pct_change": np.nanmean([pct(pm), pct(ps)]) if base > 0 else np.nan,
                "markov_diff_low": mk.ci_low, "markov_diff_high": mk.ci_high, "markov_significant": bool(mk.significant),
                "shapley_diff_low": sh.ci_low, "shapley_diff_high": sh.ci_high, "shapley_significant": bool(sh.significant),
                "markov_vs_shapley_significant": bool(dc.loc[("markov - shapley", ch)].significant),
                "verdict": v,
            })
    verdict = pd.DataFrame(verdict)

    stats = {
        "n_boot": n_boot, "bootstrap_seconds": boot_secs, "seed": seed,
        "total_conversions": float(lt.conversions.sum()), "total_revenue": float(lt.revenue.sum()),
        "removal_effects": A.markov(j)["removal_effect"].to_dict(),
        "verdict": verdict.to_dict(orient="records"),
    }
    return stats, results, changes, ci, diffs, verdict, boot


# --------------------------------------------------------------------------- budget

def budget_simulation(verdict: pd.DataFrame, results: pd.DataFrame, boot: pd.DataFrame,
                      sessions_by_channel: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Shift SHIFT_PCT of sessions ("effort", GA4 has no spend) from the most over-credited
    addressable channel to the most under-credited one, valuing a session at its
    Markov-credited revenue per session.

    Over/under-credited is ranked on revenue, by the mean % change vs last touch across the
    two data-driven models, among addressable channels with >= MIN_CONVERSIONS conversions.
    """
    v = verdict[verdict.metric == "revenue"].set_index("channel")
    conv = verdict[verdict.metric == "conversions"].set_index("channel")
    cand = v.loc[[c for c in ADDRESSABLE if c in v.index]]
    big = cand[(conv.loc[cand.index, "last_touch"] >= MIN_CONVERSIONS) | (conv.loc[cand.index, "markov"] >= MIN_CONVERSIONS)]
    src = big.mean_pct_change.idxmin()
    dst = big.drop(index=src).mean_pct_change.idxmax()

    rev = results.pivot(index="channel", columns="model", values="revenue")
    vps = (rev["markov"] / sessions_by_channel).rename("markov_rev_per_session")
    vps_lt = rev["last_touch"] / sessions_by_channel
    moved = SHIFT_PCT * sessions_by_channel[src]

    bw = boot[boot.model == "markov"].pivot(index="replicate", columns="channel", values="revenue")
    rows = []
    for name, k in RETURNS_SCENARIOS.items():
        delta = moved * (k * vps[dst] - vps[src])
        dist = moved * (k * bw[dst] / sessions_by_channel[dst] - bw[src] / sessions_by_channel[src])
        lo, hi = np.percentile(dist, [2.5, 97.5])
        delta_lt = moved * (k * vps_lt[dst] - vps_lt[src])
        rows.append({"scenario": name, "k": k, "delta_revenue": delta, "ci_low": lo, "ci_high": hi,
                     "delta_revenue_last_touch_view": delta_lt})
    sim = pd.DataFrame(rows)
    stats = {
        "from_channel": src, "to_channel": dst, "shift_pct": SHIFT_PCT, "sessions_moved": float(moved),
        "from_sessions": float(sessions_by_channel[src]), "to_sessions": float(sessions_by_channel[dst]),
        "from_markov_rev_per_session": float(vps[src]), "to_markov_rev_per_session": float(vps[dst]),
        "from_lt_rev_per_session": float(vps_lt[src]), "to_lt_rev_per_session": float(vps_lt[dst]),
        "from_verdict": v.at[src, "verdict"], "to_verdict": v.at[dst, "verdict"],
        "from_mean_pct_change": float(v.at[src, "mean_pct_change"]), "to_mean_pct_change": float(v.at[dst, "mean_pct_change"]),
        "candidates": big[["mean_pct_change", "markov_pct_change", "shapley_pct_change", "verdict"]].reset_index().to_dict(orient="records"),
        "scenarios": sim.to_dict(orient="records"),
        "rev_per_session_markov": vps.to_dict(), "rev_per_session_last_touch": vps_lt.to_dict(),
    }
    return stats, sim


# --------------------------------------------------------------------------- main

def main(n_boot: int = 1000, seed: int = 42) -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    kpis = pd.read_parquet(CACHE / "kpis.parquet").iloc[0]
    daily = pd.read_parquet(CACHE / "sessions_daily.parquet")
    journeys_mart = pd.read_parquet(CACHE / "journeys.parquet")
    funnel = pd.read_parquet(CACHE / "funnel.parquet")
    profile = json.loads((REPORTS / "data_profile_metrics.json").read_text())

    print("[analysis] building journeys")
    j = load_journeys()
    jstats, pl, ttc, paths = journey_stats(j, journeys_mart)

    print("[analysis] funnel")
    fstats, f_overall, f_channel, f_device = funnel_analysis(funnel)

    print(f"[analysis] attribution: 8 models + bootstrap ({n_boot} resamples)")
    astats, results, changes, ci, diffs, verdict, boot = attribution(j, n_boot, seed)

    ch_sessions = daily.groupby("channel")["sessions"].sum()
    print("[analysis] budget simulation")
    bstats, sim = budget_simulation(verdict, results, boot, ch_sessions)

    by_ch = daily.groupby("channel")[["sessions", "converting_sessions", "revenue_usd", "inferred_sessions"]].sum()
    by_ch["share"] = by_ch.sessions / by_ch.sessions.sum()
    by_ch["session_cr"] = by_ch.converting_sessions / by_ch.sessions
    overview = {
        "sessions": int(kpis.sessions), "users": int(kpis.users), "converting_sessions": int(kpis.converting_sessions),
        "purchases": int(kpis.purchases), "revenue_usd": float(kpis.revenue_usd),
        "purchasing_users": int(kpis.purchasing_users),
        "session_cr": float(kpis.converting_sessions / kpis.sessions),
        "aov": float(kpis.revenue_usd / kpis.purchases),
        "inferred_direct_sessions": int(kpis.inferred_sessions),
        "inferred_direct_share": float(kpis.inferred_sessions / kpis.sessions),
        "other_unknown_sessions": int(kpis.other_unknown_sessions),
        "other_unknown_share": float(kpis.other_unknown_sessions / kpis.sessions),
        "channels": by_ch.reset_index().to_dict(orient="records"),
    }
    checks = {
        "journey_revenue_matches_sessions": abs(jstats["python_journeys_revenue"] - overview["revenue_usd"]) < 0.01,
        "converting_journeys_match_sessions": jstats["converting_journeys"] == overview["converting_sessions"],
        "every_model_conserves_conversions": bool(np.allclose(results.groupby("model").conversions.sum(), jstats["converting_journeys"])),
        "every_model_conserves_revenue": bool(np.allclose(results.groupby("model").revenue.sum(), overview["revenue_usd"])),
    }
    if not all(checks.values()):
        raise SystemExit(f"Consistency checks failed: {checks}")

    metrics = {"profile": profile, "overview": overview, "journeys": jstats, "funnel": fstats,
               "attribution": astats, "budget": bstats, "checks": checks}

    results.to_csv(REPORTS / "attribution_results.csv", index=False)
    changes.to_csv(REPORTS / "model_comparison.csv", index=False)
    ci.to_csv(REPORTS / "bootstrap_ci.csv", index=False)
    diffs.to_csv(REPORTS / "bootstrap_differences.csv", index=False)
    verdict.to_csv(REPORTS / "last_touch_bias.csv", index=False)
    sim.to_csv(REPORTS / "budget_simulation.csv", index=False)
    paths.to_csv(REPORTS / "converting_paths.csv", index=False)
    pl.to_csv(REPORTS / "path_length.csv", index=False)
    ttc.to_csv(REPORTS / "time_to_conversion.csv", index=False)
    f_overall.to_csv(REPORTS / "funnel_overall.csv", index=False)
    f_channel.to_csv(REPORTS / "funnel_by_channel.csv", index=False)
    f_device.to_csv(REPORTS / "funnel_by_device.csv", index=False)
    A.transition_matrix(j).to_csv(REPORTS / "markov_transition_matrix.csv")
    daily.to_csv(REPORTS / "sessions_daily.csv", index=False)
    pd.read_parquet(CACHE / "channel_map.parquet").to_csv(REPORTS / "channel_map.csv", index=False)
    boot.to_parquet(REPORTS / "bootstrap_replicates.parquet", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("[analysis] figures")
    lab = A.MODEL_LABELS
    figs = {
        "credit_heatmap": viz.credit_heatmap(results, lab),
        "pct_change_heatmap": viz.pct_change_heatmap(changes[changes.model.isin(["markov", "shapley"])]),
        "ci_vs_last_touch_conversions": viz.diff_ci_plot(diffs, results[results.model == "last_touch"].set_index("channel").conversions, "conversions"),
        "ci_vs_last_touch_revenue": viz.diff_ci_plot(diffs, results[results.model == "last_touch"].set_index("channel").revenue, "revenue"),
        "funnel": viz.funnel_chart(f_overall),
        "funnel_by_device": viz.step_rate_heatmap(f_device[f_device.step_order > 1], "device_category"),
        "path_length": viz.path_length_chart(pl),
        "sankey": viz.sankey(paths),
        "budget_simulation": viz.budget_chart(sim),
    }
    for name, fig in figs.items():
        fig.write_image(FIGS / f"{name}.png", width=1000, height=fig.layout.height or 420, scale=2)
    print(f"[analysis] done (bootstrap {astats['bootstrap_seconds']:.0f}s). Checks: {checks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n_boot, args.seed)
