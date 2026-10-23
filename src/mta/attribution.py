"""Multi-touch attribution models, implemented from scratch.

Every model takes a *journeys* DataFrame with one row per journey:

    journey_id   : str               unique id (user + journey index)
    path         : list[str]         ordered channels, one per session
    touch_ts     : list[pd.Timestamp] session start time for each touch (same length as path)
    converted    : bool              journey ended in a purchase
    conversion_ts: pd.Timestamp|NaT  purchase time (NaT when not converted)
    revenue      : float             purchase revenue (0 when not converted)

and returns a DataFrame indexed by channel with columns
``conversions`` and ``revenue`` (credited amounts, summing to the totals of
converting journeys).

Rule-based models (a-f) only look at converting journeys. The two data-driven
models (g, h) are described in their docstrings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

DIRECT = "Direct"
START, CONV, NULL = "(start)", "(conversion)", "(null)"

Weights = dict[str, float]


# ---------------------------------------------------------------------------
# Path-level weighting rules (heuristic models). Each returns channel -> weight
# with weights summing to 1 for a single converting path.
# ---------------------------------------------------------------------------

def _accumulate(channels: Sequence[str], weights: Sequence[float]) -> Weights:
    out: Weights = {}
    for ch, w in zip(channels, weights):
        out[ch] = out.get(ch, 0.0) + float(w)
    return out


def first_touch_weights(path: Sequence[str]) -> Weights:
    """All credit to the first touch."""
    return {path[0]: 1.0}


def last_touch_weights(path: Sequence[str]) -> Weights:
    """All credit to the last touch."""
    return {path[-1]: 1.0}


def last_non_direct_weights(path: Sequence[str], direct: str = DIRECT) -> Weights:
    """All credit to the last touch that is not Direct (Direct if the path is all Direct).

    This mirrors Google Analytics' long-standing default: a Direct revisit does
    not overwrite an earlier known campaign/source.
    """
    for ch in reversed(path):
        if ch != direct:
            return {ch: 1.0}
    return {direct: 1.0}


def linear_weights(path: Sequence[str]) -> Weights:
    """Equal credit to every touch (repeat touches of a channel add up)."""
    n = len(path)
    return _accumulate(path, [1.0 / n] * n)


def time_decay_weights(
    path: Sequence[str],
    touch_ts: Sequence[pd.Timestamp],
    conversion_ts: pd.Timestamp,
    half_life_days: float = 7.0,
) -> Weights:
    """Exponential decay: a touch ``half_life_days`` before conversion gets half
    the weight of a touch at the moment of conversion. Weights are normalised to 1.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    age_days = np.array(
        [(conversion_ts - t).total_seconds() / 86400.0 for t in touch_ts], dtype=float
    )
    age_days = np.clip(age_days, 0.0, None)
    raw = np.power(0.5, age_days / half_life_days)
    return _accumulate(path, raw / raw.sum())


def position_based_weights(
    path: Sequence[str], first: float = 0.4, last: float = 0.4
) -> Weights:
    """U-shaped: ``first`` to the first touch, ``last`` to the last, the rest
    spread evenly over the middle touches.

    One touch gets 100%. Two touches split first/last proportionally (50/50 for 40/40).
    """
    n = len(path)
    if n == 1:
        return {path[0]: 1.0}
    if n == 2:
        s = first + last
        return _accumulate(path, [first / s, last / s])
    middle = (1.0 - first - last) / (n - 2)
    return _accumulate(path, [first] + [middle] * (n - 2) + [last])


# ---------------------------------------------------------------------------
# Applying a path rule to a journeys table
# ---------------------------------------------------------------------------

def _credit_from_rule(journeys: pd.DataFrame, rule: Callable[[pd.Series], Weights]) -> pd.DataFrame:
    conv = journeys[journeys["converted"]]
    conv_credit: dict[str, float] = {}
    rev_credit: dict[str, float] = {}
    for _, row in conv.iterrows():
        for ch, w in rule(row).items():
            conv_credit[ch] = conv_credit.get(ch, 0.0) + w
            rev_credit[ch] = rev_credit.get(ch, 0.0) + w * float(row["revenue"])
    return _as_frame(conv_credit, rev_credit)


def _as_frame(conv_credit: dict[str, float], rev_credit: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({"conversions": pd.Series(conv_credit, dtype=float),
                       "revenue": pd.Series(rev_credit, dtype=float)}).fillna(0.0)
    df.index.name = "channel"
    return df.sort_index()


def first_touch(journeys: pd.DataFrame) -> pd.DataFrame:
    """(a) First-touch attribution."""
    return _credit_from_rule(journeys, lambda r: first_touch_weights(r["path"]))


def last_touch(journeys: pd.DataFrame) -> pd.DataFrame:
    """(b) Last-touch attribution."""
    return _credit_from_rule(journeys, lambda r: last_touch_weights(r["path"]))


def last_non_direct(journeys: pd.DataFrame) -> pd.DataFrame:
    """(c) Last non-direct click attribution."""
    return _credit_from_rule(journeys, lambda r: last_non_direct_weights(r["path"]))


def linear(journeys: pd.DataFrame) -> pd.DataFrame:
    """(d) Linear attribution."""
    return _credit_from_rule(journeys, lambda r: linear_weights(r["path"]))


def time_decay(journeys: pd.DataFrame, half_life_days: float = 7.0) -> pd.DataFrame:
    """(e) Time-decay attribution with a configurable half-life (days)."""
    return _credit_from_rule(
        journeys,
        lambda r: time_decay_weights(r["path"], r["touch_ts"], r["conversion_ts"], half_life_days),
    )


def position_based(journeys: pd.DataFrame, first: float = 0.4, last: float = 0.4) -> pd.DataFrame:
    """(f) Position-based (U-shaped, 40/20/40 by default) attribution."""
    return _credit_from_rule(journeys, lambda r: position_based_weights(r["path"], first, last))


# ---------------------------------------------------------------------------
# (g) Markov chain removal effect
# ---------------------------------------------------------------------------

def path_groups(journeys: pd.DataFrame) -> pd.DataFrame:
    """Collapse journeys into unique (path, converted, revenue) groups.

    Columns: path (tuple), converted, unit_revenue, n (journeys). Both data-driven
    models depend only on these sufficient statistics, so resampling journeys with
    replacement is exactly equivalent to re-drawing the group counts ``n`` (grouping on
    revenue too keeps revenue variance intact in the bootstrap).
    """
    g = journeys.assign(path=journeys["path"].map(tuple), unit_revenue=journeys["revenue"].astype(float))
    return (
        g.groupby(["path", "converted", "unit_revenue"], sort=True)
        .size().rename("n").reset_index()
    )


def transition_counts(
    groups: pd.DataFrame, channels: Sequence[str], counts: np.ndarray | None = None
) -> np.ndarray:
    """First-order transition count matrix over states
    [START, *channels, CONV, NULL]; each path is START -> c1 -> ... -> ck -> CONV|NULL.
    """
    states = [START, *channels, CONV, NULL]
    idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    counts = groups["n"].to_numpy(dtype=float) if counts is None else counts
    mat = np.zeros((n_states, n_states))
    for path, converted, c in zip(groups["path"], groups["converted"], counts):
        if c == 0:
            continue
        seq = [START, *path, CONV if converted else NULL]
        for a, b in zip(seq[:-1], seq[1:]):
            mat[idx[a], idx[b]] += c
    return mat


def _group_transition_tensor(groups: pd.DataFrame, channels: Sequence[str]) -> np.ndarray:
    """Per-group transition counts, shape (n_groups, n_states*n_states), so that
    ``counts @ tensor`` gives the aggregate transition matrix for any reweighting."""
    states = [START, *channels, CONV, NULL]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    tensor = np.zeros((len(groups), n * n))
    for g, (path, converted) in enumerate(zip(groups["path"], groups["converted"])):
        seq = [START, *path, CONV if converted else NULL]
        for x, y in zip(seq[:-1], seq[1:]):
            tensor[g, idx[x] * n + idx[y]] += 1
    return tensor


def _incidence(groups: pd.DataFrame, channels: Sequence[str]) -> np.ndarray:
    """Binary matrix (n_groups, n_channels): does the group's path touch the channel?"""
    col = {c: i for i, c in enumerate(channels)}
    m = np.zeros((len(groups), len(channels)))
    for g, path in enumerate(groups["path"]):
        for ch in set(path):
            m[g, col[ch]] = 1.0
    return m


def conversion_probability(trans: np.ndarray, removed: int | None = None) -> float:
    """P(reaching CONV from START) in the absorbing chain given by row-normalised
    ``trans``. If ``removed`` is a state index, every transition into it is
    redirected to NULL (the channel "disappears" and those users are lost).

    Solved exactly: for transient states T, x = Q x + r, so x = (I - Q)^-1 r.
    """
    n = trans.shape[0]
    conv_i, null_i = n - 2, n - 1
    t = trans.astype(float).copy()
    if removed is not None:
        t[:, null_i] += t[:, removed]
        t[:, removed] = 0.0
        t[removed, :] = 0.0
    transient = [i for i in range(n) if i not in (conv_i, null_i)]
    row_sums = t[transient].sum(axis=1)
    live = [s for s, rs in zip(transient, row_sums) if rs > 0]
    if 0 not in live:
        return 0.0
    p = t[np.ix_(live, live)] / t[live].sum(axis=1, keepdims=True)
    r = t[live, conv_i] / t[live].sum(axis=1)
    x = np.linalg.solve(np.eye(len(live)) - p, r)
    return float(x[live.index(0)])


def removal_effects(trans: np.ndarray, channels: Sequence[str]) -> pd.Series:
    """Removal effect per channel: 1 - P(conv without channel) / P(conv)."""
    base = conversion_probability(trans)
    effects = {}
    for k, ch in enumerate(channels, start=1):
        effects[ch] = 0.0 if base == 0 else 1.0 - conversion_probability(trans, removed=k) / base
    return pd.Series(effects, name="removal_effect")


def _markov_from_groups(
    groups: pd.DataFrame,
    channels: Sequence[str],
    counts: np.ndarray,
    trans: np.ndarray,
    incidence: np.ndarray | None = None,
) -> pd.DataFrame:
    re = removal_effects(trans, channels)
    conv_mask = groups["converted"].to_numpy()
    total_conv = float(counts[conv_mask].sum())
    share = re / re.sum() if re.sum() > 0 else re * 0.0

    # Revenue: each converting path's revenue is split across the distinct channels on
    # that path in proportion to their removal effects (so high-value paths credit the
    # channels that actually appear on them). Uniform split if all effects are <= 0.
    inc = _incidence(groups, channels) if incidence is None else incidence
    w = inc * np.clip(re.to_numpy(), 0.0, None)
    row = w.sum(axis=1, keepdims=True)
    w = np.where(row > 0, w / np.where(row > 0, row, 1.0), inc / inc.sum(axis=1, keepdims=True).clip(min=1))
    value = counts * groups["unit_revenue"].to_numpy(dtype=float) * conv_mask
    rev_credit = pd.Series(value @ w, index=list(channels))
    out = pd.DataFrame(
        {"conversions": share * total_conv, "revenue": rev_credit, "removal_effect": re}
    )
    out.index.name = "channel"
    return out.sort_index()


def markov(journeys: pd.DataFrame, channels: Sequence[str] | None = None) -> pd.DataFrame:
    """(g) First-order Markov chain attribution via removal effects.

    Uses *all* journeys (converting and not). Conversions are credited as
    total_conversions * removal_effect / sum(removal_effects). Revenue is split
    path by path in proportion to the removal effects of the channels on it.
    Also returns the raw ``removal_effect`` column.
    """
    groups = path_groups(journeys)
    channels = sorted({c for p in groups["path"] for c in p}) if channels is None else list(channels)
    counts = groups["n"].to_numpy(dtype=float)
    return _markov_from_groups(groups, channels, counts, transition_counts(groups, channels))


def transition_matrix(journeys: pd.DataFrame, channels: Sequence[str] | None = None) -> pd.DataFrame:
    """Row-normalised first-order transition probabilities (for inspection/plots)."""
    groups = path_groups(journeys)
    channels = sorted({c for p in groups["path"] for c in p}) if channels is None else list(channels)
    states = [START, *channels, CONV, NULL]
    t = transition_counts(groups, channels)
    rs = t.sum(axis=1, keepdims=True)
    p = np.divide(t, rs, out=np.zeros_like(t), where=rs > 0)
    return pd.DataFrame(p, index=states, columns=states)


# ---------------------------------------------------------------------------
# (h) Shapley value attribution
# ---------------------------------------------------------------------------

def shapley_values(value: dict[frozenset[str], float], players: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values for a characteristic function given as a dict
    (missing coalitions are worth 0). Complexity O(n * 2^n); fine for n <= ~15.

    phi_i = sum_{S subset N\\{i}} |S|!(n-|S|-1)!/n! * (v(S u {i}) - v(S))
    """
    n = len(players)
    w = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for k in range(n):
            for s in combinations(others, k):
                fs = frozenset(s)
                total += w[k] * (value.get(fs | {i}, 0.0) - value.get(fs, 0.0))
        phi[i] = total
    return phi


def coalition_values(set_totals: dict[frozenset[str], float], players: Sequence[str]) -> dict[frozenset[str], float]:
    """v(S) = sum of totals of journeys whose channel *set* is a subset of S
    (Zhao et al., 2018). ``set_totals`` maps exact channel set -> total."""
    v: dict[frozenset[str], float] = {}
    for k in range(len(players) + 1):
        for s in combinations(players, k):
            fs = frozenset(s)
            v[fs] = sum(t for cs, t in set_totals.items() if cs <= fs)
    return v


def _shapley_from_groups(groups: pd.DataFrame, channels: Sequence[str], counts: np.ndarray) -> pd.DataFrame:
    conv_mask = groups["converted"].to_numpy()
    mean_rev = groups["unit_revenue"].to_numpy(dtype=float)
    conv_by_set: dict[frozenset[str], float] = {}
    rev_by_set: dict[frozenset[str], float] = {}
    for path, is_conv, c, mr in zip(groups["path"], conv_mask, counts, mean_rev):
        if not is_conv or c == 0:
            continue
        key = frozenset(path)
        conv_by_set[key] = conv_by_set.get(key, 0.0) + c
        rev_by_set[key] = rev_by_set.get(key, 0.0) + c * mr
    out = pd.DataFrame({
        "conversions": pd.Series(shapley_values(coalition_values(conv_by_set, channels), channels)),
        "revenue": pd.Series(shapley_values(coalition_values(rev_by_set, channels), channels)),
    })
    out.index.name = "channel"
    return out.sort_index()


def _shapley_closed_form(
    groups: pd.DataFrame, channels: Sequence[str], counts: np.ndarray, incidence: np.ndarray
) -> pd.DataFrame:
    """Vectorised Shapley for the subset-sum value function: v is a sum of unanimity
    games, so each converting journey's value is split evenly over its distinct
    channels. Identical to the exact enumeration (see tests); used in the bootstrap."""
    conv_mask = groups["converted"].to_numpy()
    share = incidence / incidence.sum(axis=1, keepdims=True).clip(min=1)
    conv = (counts * conv_mask) @ share
    rev = (counts * conv_mask * groups["unit_revenue"].to_numpy(dtype=float)) @ share
    out = pd.DataFrame({"conversions": conv, "revenue": rev}, index=pd.Index(list(channels), name="channel"))
    return out.sort_index()


def shapley(journeys: pd.DataFrame, channels: Sequence[str] | None = None) -> pd.DataFrame:
    """(h) Shapley value attribution, computed exactly over all 2^n coalitions.

    Characteristic function: v(S) = conversions (or revenue) from converting
    journeys whose set of channels is contained in S. By efficiency the credits
    sum to total conversions/revenue. Note: under this value function the Shapley
    value has the closed form phi_i = sum_{T contains i} v_T / |T| - each
    conversion is split evenly over the *distinct* channels on its path,
    ignoring order and repeats (tests verify the exact computation matches).
    """
    groups = path_groups(journeys)
    channels = sorted({c for p in groups["path"] for c in p}) if channels is None else list(channels)
    return _shapley_from_groups(groups, channels, groups["n"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Running everything + bootstrap
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "first_touch": "First touch",
    "last_touch": "Last touch",
    "last_non_direct": "Last non-direct",
    "linear": "Linear",
    "time_decay": "Time decay (7d)",
    "position_based": "Position-based 40/20/40",
    "markov": "Markov (data-driven)",
    "shapley": "Shapley (data-driven)",
}


def run_all_models(journeys: pd.DataFrame, half_life_days: float = 7.0) -> pd.DataFrame:
    """Run all eight models. Returns long format: model, channel, conversions, revenue."""
    channels = sorted({c for p in journeys["path"] for c in p})
    results = {
        "first_touch": first_touch(journeys),
        "last_touch": last_touch(journeys),
        "last_non_direct": last_non_direct(journeys),
        "linear": linear(journeys),
        "time_decay": time_decay(journeys, half_life_days),
        "position_based": position_based(journeys),
        "markov": markov(journeys, channels)[["conversions", "revenue"]],
        "shapley": shapley(journeys, channels),
    }
    frames = []
    for name, df in results.items():
        df = df.reindex(channels).fillna(0.0).reset_index()
        df.insert(0, "model", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _last_touch_from_groups(groups: pd.DataFrame, channels: Sequence[str], counts: np.ndarray) -> pd.DataFrame:
    conv = groups["converted"].to_numpy()
    last = groups["path"].map(lambda p: p[-1]).to_numpy()
    rev = groups["unit_revenue"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "conversions": [float(counts[conv & (last == ch)].sum()) for ch in channels],
        "revenue": [float((counts * rev)[conv & (last == ch)].sum()) for ch in channels],
    }, index=pd.Index(channels, name="channel"))
    return out


def bootstrap_data_driven(
    journeys: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Bootstrap (resampling journeys with replacement) for Markov and Shapley, plus
    last-touch on the same replicates so differences get paired confidence intervals.

    Resampling N journeys with replacement is equivalent to drawing group counts
    from Multinomial(N, n_g / N) over unique (path, converted) groups, which is
    what we do. Returns long format: replicate, model, channel, conversions, revenue.
    """
    rng = np.random.default_rng(seed)
    groups = path_groups(journeys)
    channels = sorted({c for p in groups["path"] for c in p})
    base = groups["n"].to_numpy(dtype=float)
    n_total = int(base.sum())
    probs = base / base.sum()
    tensor = _group_transition_tensor(groups, channels)
    inc = _incidence(groups, channels)
    n_states = len(channels) + 3
    frames = []
    for b in range(n_boot):
        counts = rng.multinomial(n_total, probs).astype(float)
        trans = (counts @ tensor).reshape(n_states, n_states)
        results = {
            "markov": _markov_from_groups(groups, channels, counts, trans, inc),
            "shapley": _shapley_closed_form(groups, channels, counts, inc),
            "last_touch": _last_touch_from_groups(groups, channels, counts),
        }
        for model, df in results.items():
            frames.append(df[["conversions", "revenue"]].assign(replicate=b, model=model).reset_index())
    out = pd.concat(frames, ignore_index=True)
    return out[["replicate", "model", "channel", "conversions", "revenue"]]
