"""Unit tests for every attribution model, each with a tiny hand-calculated case."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mta import attribution as A

T0 = pd.Timestamp("2021-01-01")


def _journeys(rows: list[tuple]) -> pd.DataFrame:
    """rows: (path, days_before_conversion_per_touch | None, converted, revenue)."""
    recs = []
    for i, (path, ages, conv, rev) in enumerate(rows):
        ages = ages or [0] * len(path)
        recs.append({
            "journey_id": f"j{i}",
            "path": list(path),
            "touch_ts": [T0 - pd.Timedelta(days=a) for a in ages],
            "converted": conv,
            "conversion_ts": T0 if conv else pd.NaT,
            "revenue": float(rev),
        })
    return pd.DataFrame(recs)


# Shared toy dataset:
#   j0: Email -> Organic -> Direct      converted, $100
#   j1: Organic                         converted, $50
#   j2: Email -> Direct                 not converted
TOY = _journeys([
    (["Email", "Organic", "Direct"], [14, 7, 0], True, 100),
    (["Organic"], [0], True, 50),
    (["Email", "Direct"], None, False, 0),
])


def credit(df: pd.DataFrame, col: str = "conversions") -> dict[str, float]:
    return {k: round(v, 6) for k, v in df[col].items() if abs(v) > 1e-12}


# ---------------------------- path-level rules -----------------------------

def test_first_touch_hand_calc():
    assert credit(A.first_touch(TOY)) == {"Email": 1.0, "Organic": 1.0}
    assert credit(A.first_touch(TOY), "revenue") == {"Email": 100.0, "Organic": 50.0}


def test_last_touch_hand_calc():
    assert credit(A.last_touch(TOY)) == {"Direct": 1.0, "Organic": 1.0}
    assert credit(A.last_touch(TOY), "revenue") == {"Direct": 100.0, "Organic": 50.0}


def test_last_non_direct_skips_direct_but_falls_back():
    assert credit(A.last_non_direct(TOY)) == {"Organic": 2.0}
    assert A.last_non_direct_weights(["Direct", "Direct"]) == {"Direct": 1.0}
    assert A.last_non_direct_weights(["Email", "Direct", "Paid Search", "Direct"]) == {"Paid Search": 1.0}


def test_linear_hand_calc():
    # j0 gives 1/3 each; j1 gives 1 to Organic.
    assert credit(A.linear(TOY)) == {"Direct": round(1 / 3, 6), "Email": round(1 / 3, 6), "Organic": round(4 / 3, 6)}
    assert A.linear_weights(["A", "B", "A", "A"]) == {"A": 0.75, "B": 0.25}


def test_time_decay_hand_calc():
    # Ages 14, 7, 0 days with 7-day half-life -> raw 0.25, 0.5, 1 -> /1.75
    w = A.time_decay_weights(["Email", "Organic", "Direct"],
                             [T0 - pd.Timedelta(days=d) for d in (14, 7, 0)], T0, 7.0)
    assert w == pytest.approx({"Email": 0.25 / 1.75, "Organic": 0.5 / 1.75, "Direct": 1 / 1.75})
    td = A.time_decay(TOY)
    assert td.loc["Organic", "conversions"] == pytest.approx(1 + 0.5 / 1.75)
    assert td.loc["Direct", "revenue"] == pytest.approx(100 / 1.75)


def test_time_decay_half_life_parameter():
    ts = [T0 - pd.Timedelta(days=d) for d in (2, 0)]
    w = A.time_decay_weights(["A", "B"], ts, T0, half_life_days=2.0)
    assert w == pytest.approx({"A": 0.5 / 1.5, "B": 1 / 1.5})
    with pytest.raises(ValueError):
        A.time_decay_weights(["A"], ts[:1], T0, half_life_days=0)


def test_position_based_hand_calc():
    assert A.position_based_weights(["A"]) == {"A": 1.0}
    assert A.position_based_weights(["A", "B"]) == pytest.approx({"A": 0.5, "B": 0.5})
    assert A.position_based_weights(["A", "B", "C", "D"]) == pytest.approx(
        {"A": 0.4, "B": 0.1, "C": 0.1, "D": 0.4})
    pb = A.position_based(TOY)
    assert credit(pb) == {"Email": 0.4, "Organic": 1.2, "Direct": 0.4}


@pytest.mark.parametrize("model", [A.first_touch, A.last_touch, A.last_non_direct,
                                   A.linear, A.time_decay, A.position_based, A.markov, A.shapley])
def test_every_model_conserves_totals(model):
    out = model(TOY)
    assert out["conversions"].sum() == pytest.approx(2.0)
    assert out["revenue"].sum() == pytest.approx(150.0)


# --------------------------------- Markov ----------------------------------

def test_markov_hand_calc():
    """Hand calculation on TOY.

    Transitions: start->Email 2, start->Organic 1, Email->Organic 1, Email->Direct 1,
    Organic->Direct 1, Organic->conv 1, Direct->conv 1, Direct->null 1.
    P(Direct->conv)=1/2, P(Organic->conv)=1/2 + 1/2*1/2 = 3/4,
    P(Email->conv)=1/2*3/4 + 1/2*1/2 = 5/8,
    P(conv)=2/3*5/8 + 1/3*3/4 = 2/3.
    Remove Email:   1/3*3/4 = 1/4        -> RE = 1 - (1/4)/(2/3) = 5/8
    Remove Organic: Email->conv = 1/4; P = 2/3*1/4 = 1/6 -> RE = 3/4
    Remove Direct:  Organic->conv = 1/2, Email->conv = 1/4; P = 1/6 + 1/6 = 1/3 -> RE = 1/2
    Shares over sum 15/8: Email 1/3, Organic 2/5, Direct 4/15; x 2 conversions.
    """
    m = A.markov(TOY)
    assert m["removal_effect"].to_dict() == pytest.approx({"Direct": 0.5, "Email": 0.625, "Organic": 0.75})
    assert m["conversions"].to_dict() == pytest.approx({"Direct": 8 / 15, "Email": 2 / 3, "Organic": 4 / 5})
    # Revenue: j1 ($50) all to Organic; j0 ($100) split by RE 4/15:5/15:6/15 (Direct:Email:Organic)
    assert m["revenue"].to_dict() == pytest.approx(
        {"Direct": 100 * 4 / 15, "Email": 100 * 5 / 15, "Organic": 50 + 100 * 6 / 15})


def test_conversion_probability_simple_chain():
    tm = A.transition_matrix(TOY)
    assert tm.loc[A.START, "Email"] == pytest.approx(2 / 3)
    assert tm.loc["Direct", A.CONV] == pytest.approx(0.5)
    groups = A.path_groups(TOY)
    trans = A.transition_counts(groups, ["Direct", "Email", "Organic"])
    assert A.conversion_probability(trans) == pytest.approx(2 / 3)


# --------------------------------- Shapley ---------------------------------

def test_shapley_values_textbook_game():
    # Glove game: v({A,B}) = v({A,C}) = v(all) = 1 else 0. phi = (2/3, 1/6, 1/6).
    v = {frozenset("AB"): 1.0, frozenset("AC"): 1.0, frozenset("ABC"): 1.0}
    phi = A.shapley_values(v, ["A", "B", "C"])
    assert phi == pytest.approx({"A": 2 / 3, "B": 1 / 6, "C": 1 / 6})


def test_shapley_hand_calc():
    # Converting sets: {Email,Organic,Direct}: 1 ($100), {Organic}: 1 ($50).
    # Closed form: each conversion split evenly over distinct channels in its set.
    s = A.shapley(TOY)
    assert s["conversions"].to_dict() == pytest.approx({"Direct": 1 / 3, "Email": 1 / 3, "Organic": 4 / 3})
    assert s["revenue"].to_dict() == pytest.approx({"Direct": 100 / 3, "Email": 100 / 3, "Organic": 50 + 100 / 3})


def test_shapley_ignores_repeats_unlike_linear():
    j = _journeys([(["A", "A", "A", "B"], None, True, 10)])
    assert A.shapley(j)["conversions"].to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert A.linear(j)["conversions"].to_dict() == pytest.approx({"A": 0.75, "B": 0.25})


# -------------------------------- bootstrap --------------------------------

def test_bootstrap_shapes_and_conservation():
    boot = A.bootstrap_data_driven(TOY, n_boot=50, seed=1)
    assert set(boot["model"]) == {"markov", "shapley", "last_touch"}
    assert boot["replicate"].nunique() == 50
    # Each replicate's credit sums to that replicate's number of conversions (0..3).
    sums = boot.groupby(["replicate", "model"])["conversions"].sum()
    assert np.all(np.isclose(sums, np.round(sums)))


def test_run_all_models_long_format():
    out = A.run_all_models(TOY)
    assert set(out["model"]) == set(A.MODEL_LABELS)
    assert out.groupby("model")["conversions"].sum().round(9).eq(2.0).all()


def test_bootstrap_full_sample_matches_point_estimate():
    """With counts equal to the observed counts, the group machinery reproduces the
    point estimates (guards the resampling code path)."""
    groups = A.path_groups(TOY)
    ch = ["Direct", "Email", "Organic"]
    n = groups["n"].to_numpy(dtype=float)
    lt = A._last_touch_from_groups(groups, ch, n)
    assert lt["conversions"].to_dict() == pytest.approx({"Direct": 1, "Email": 0, "Organic": 1})
    assert lt["revenue"].to_dict() == pytest.approx({"Direct": 100, "Email": 0, "Organic": 50})
    trans = (n @ A._group_transition_tensor(groups, ch)).reshape(6, 6)
    mk = A._markov_from_groups(groups, ch, n, trans)
    assert mk["conversions"].to_dict() == pytest.approx(A.markov(TOY)["conversions"].to_dict())


def test_shapley_closed_form_equals_exact_enumeration():
    """On random journeys over 5 channels, the vectorised closed form used in the
    bootstrap equals exact 2^n Shapley enumeration."""
    rng = np.random.default_rng(0)
    chans = ["A", "B", "C", "D", "E"]
    rows = []
    for _ in range(300):
        k = int(rng.integers(1, 6))
        rows.append((list(rng.choice(chans, size=k)), None, bool(rng.random() < 0.3), float(rng.integers(1, 100))))
    j = _journeys(rows)
    groups = A.path_groups(j)
    exact = A.shapley(j, chans)
    closed = A._shapley_closed_form(groups, chans, groups["n"].to_numpy(dtype=float), A._incidence(groups, chans))
    pd.testing.assert_frame_equal(exact, closed, check_exact=False, atol=1e-9)
