"""Tests for write-up helpers: number formatting and the narrative guard."""

import pytest

from mta.viz import money
from mta.writeup import NarrativeError, guard


def test_money_puts_sign_before_currency():
    assert money(-113.9, signed=True) == "-$114"
    assert money(1140, signed=True) == "+$1,140"
    assert money(0.3348, decimals=2) == "$0.33"


def test_guard_raises_when_claim_unsupported():
    guard(True, "fine")
    with pytest.raises(NarrativeError, match="no longer supported"):
        guard(False, "Organic Search is under-credited")
