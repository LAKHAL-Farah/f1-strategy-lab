
from __future__ import annotations

from train import select_bakeoff_winner


def test_refined_wins_clearly_above_margin():
    # refined beats simple by 500ms; margin (noise) is only 200ms — a real win
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=4500, margin_ms=200) == "refined"


def test_refined_wins_but_within_noise_margin_stays_simple():
    # refined is numerically lower, but the gain (50ms) doesn't clear the
    # margin (200ms) — should NOT switch away from simple
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=4950, margin_ms=200) == "simple"


def test_simple_outright_wins_stays_simple():
    # the actual result seen on real project data: refined's val MAE was
    # HIGHER than simple's (gain is negative), regardless of margin size
    assert select_bakeoff_winner(simple_mae=4760.81, refined_mae=4895.29, margin_ms=1988.68) == "simple"


def test_exact_tie_on_mae_stays_simple():
    # gain == 0 must not switch — ties go to the simpler model
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=5000, margin_ms=200) == "simple"


def test_gain_exactly_equal_to_margin_stays_simple():
    # boundary case: strictly greater than margin is required to switch,
    # so gain == margin must NOT switch (see select_bakeoff_winner's
    # docstring — "strictly MORE than margin")
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=4800, margin_ms=200) == "simple"


def test_gain_just_above_margin_switches_to_refined():
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=4799.99, margin_ms=200) == "refined"


def test_zero_margin_switches_on_any_positive_gain():
    # margin_ms=0 recovers the old "any improvement wins" behavior
    assert select_bakeoff_winner(simple_mae=5000, refined_mae=4999, margin_ms=0) == "refined"


def test_negative_margin_is_rejected():
    try:
        select_bakeoff_winner(simple_mae=5000, refined_mae=4000, margin_ms=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative margin")


def test_return_value_is_always_one_of_the_two_valid_strings():
    for simple_mae, refined_mae, margin in [
        (5000, 4000, 100), (4000, 5000, 100), (5000, 5000, 0), (1, 1, 1),
    ]:
        result = select_bakeoff_winner(simple_mae, refined_mae, margin)
        assert result in ("simple", "refined"), f"unexpected return value: {result!r}"


if __name__ == "__main__":
    # Lightweight runner so this works without pytest installed — pytest
    # (or `python -m pytest`) is still the preferred way to run this.
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
