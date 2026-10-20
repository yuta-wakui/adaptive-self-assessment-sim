# -*- coding: utf-8 -*-
"""
Unit tests for question item selection during adaptive self-assessment simulations.
This module tests the functions for selecting question items during adaptive self-assessment simulations.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/unit/test_selector.py
# Author: Yuta Wakui
# Date: 2026-02-11
# Description: Unit tests for question item selection during adaptive self-assessment

import pytest
import pandas as pd
import numpy as np

from adaptive_self_assessment.components.selector import (
    QuestionSelector,
    SelectionStrategy,
    FIXED_STRATEGIES,
    ADAPTIVE_STRATEGIES,
    compute_importance_order,
    compute_redundancy_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def items():
    return [f"item-{i}" for i in range(1, 6)]

@pytest.fixture
def importance_order(items):
    return items.copy()  # item-1 is most important

@pytest.fixture
def item_cols():
    return ["item-1", "item-2", "item-3", "item-4", "item-5"]

@pytest.fixture
def df_train_mi():
    """Training data with a known mutual information structure.

    The target is built from two independent factors, target = strong + weak:
    item-1 and item-2 are both copies of `strong`, so they carry the highest
    relevance I(Q; Y) = 0.64 nats but are fully redundant with each other.
    item-3 is `weak`: lower relevance (0.23 nats), yet independent of item-1.
    item-4 and item-5 are independent noise: near-zero relevance.

    MaxRev should therefore rank item-2 second, while mRMR should push it below
    item-3 because of the redundancy penalty.
    """
    rng = np.random.default_rng(0)
    n = 300
    strong = rng.integers(0, 3, size=n)
    weak = rng.integers(0, 2, size=n)
    data = {
        "item-1": strong,
        "item-2": strong,              # identical to item-1
        "item-3": weak,
        "item-4": rng.integers(0, 3, size=n),
        "item-5": rng.integers(0, 3, size=n),
        "target": strong + weak,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# RANDOM strategy
# ---------------------------------------------------------------------------

def test_select_question_reproducible():
    C = [f"item-{i}" for i in range(1, 11)]

    s1 = QuestionSelector(strategy=SelectionStrategy.RANDOM, seed=123)
    s2 = QuestionSelector(strategy=SelectionStrategy.RANDOM, seed=123)

    first = s1.select(C)
    second = s2.select(C)

    assert first == second

def test_select_question_empty_list():
    selector = QuestionSelector(seed=123)
    with pytest.raises(ValueError):
        selector.select([])

def test_dynamic_question_selection_until_empty():
    C = [f"item-{i}" for i in range(1, 11)]
    selector = QuestionSelector(strategy=SelectionStrategy.RANDOM, seed=123)

    selected = []
    remaining = C.copy()

    while remaining:
        q = selector.select(remaining)
        assert q in remaining
        assert q not in selected
        selected.append(q)
        remaining.remove(q)

    assert len(selected) == 10

def test_unknown_strategy_raises_not_implemented():
    selector = QuestionSelector(seed=123)
    selector.strategy = "unknown"

    with pytest.raises(NotImplementedError):
        selector.select(["item-1"])


# ---------------------------------------------------------------------------
# Fixed-order strategies: QuestionSelector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_fixed_strategy_requires_importance_order(strategy):
    with pytest.raises(ValueError, match="importance_order"):
        QuestionSelector(strategy=strategy, seed=0)


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_fixed_strategy_selects_highest_ranked_available(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    # With all items available, should pick item-1 (top of importance_order)
    assert selector.select(items) == "item-1"


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_fixed_strategy_skips_removed_items(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    remaining = items.copy()
    remaining.remove("item-1")  # top item already answered

    assert selector.select(remaining) == "item-2"


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_fixed_strategy_exhausts_in_order(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    selected = []
    remaining = items.copy()

    while remaining:
        q = selector.select(remaining)
        selected.append(q)
        remaining.remove(q)

    assert selected == importance_order


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_fixed_strategy_raises_when_no_match(strategy, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    # C contains items not in importance_order
    with pytest.raises(ValueError):
        selector.select(["unknown-item"])


# ---------------------------------------------------------------------------
# compute_importance_order: Rubric-Based MaxRev
# ---------------------------------------------------------------------------

def test_maxrev_ranks_by_relevance(df_train_mi, item_cols):
    """Items informative about the target outrank independent noise items."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MAXREV,
    )
    assert order[:3] == ["item-1", "item-2", "item-3"]
    assert set(order[3:]) == {"item-4", "item-5"}


def test_maxrev_keeps_redundant_item_near_top(df_train_mi, item_cols):
    """MaxRev ignores redundancy, so the duplicated item-2 stays at rank 2."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MAXREV,
    )
    assert order[1] == "item-2"


def test_maxrev_returns_all_items(df_train_mi, item_cols):
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MAXREV,
    )
    assert len(order) == len(item_cols)
    assert set(order) == set(item_cols)


# ---------------------------------------------------------------------------
# compute_importance_order: Rubric-Based mRMR
# ---------------------------------------------------------------------------

def test_mrmr_first_item_is_most_relevant(df_train_mi, item_cols):
    """With S empty, the first pick falls back to the Max-Relevance criterion."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MRMR,
    )
    assert order[0] == "item-1"


def test_mrmr_demotes_redundant_item(df_train_mi, item_cols):
    """item-2 duplicates item-1, so mRMR prefers the less redundant item-3."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MRMR,
    )
    assert order[1] == "item-3"
    assert order.index("item-2") > order.index("item-3")


def test_mrmr_differs_from_maxrev(df_train_mi, item_cols):
    """The redundancy term must actually change the resulting order."""
    kwargs = dict(df_train=df_train_mi, item_cols=item_cols, target_col="target")

    maxrev = compute_importance_order(strategy=SelectionStrategy.RUBRIC_BASED_MAXREV, **kwargs)
    mrmr = compute_importance_order(strategy=SelectionStrategy.RUBRIC_BASED_MRMR, **kwargs)

    assert maxrev != mrmr
    assert set(maxrev) == set(mrmr)


def test_mrmr_returns_all_items(df_train_mi, item_cols):
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.RUBRIC_BASED_MRMR,
    )
    assert len(order) == len(item_cols)
    assert set(order) == set(item_cols)


# ---------------------------------------------------------------------------
# compute_importance_order: common behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_compute_importance_order_reproducible(df_train_mi, item_cols, strategy):
    kwargs = dict(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=strategy,
    )
    assert compute_importance_order(**kwargs) == compute_importance_order(**kwargs)


def test_compute_importance_order_raises_for_random():
    with pytest.raises(ValueError, match="fixed strategies"):
        compute_importance_order(
            df_train=pd.DataFrame(),
            item_cols=[],
            target_col="target",
            strategy=SelectionStrategy.RANDOM,
        )


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_compute_importance_order_raises_when_less_than_two_items(strategy):
    df = pd.DataFrame({"item-1": [0, 1, 2], "target": [0, 1, 2]})

    with pytest.raises(ValueError, match="At least two item columns"):
        compute_importance_order(
            df_train=df,
            item_cols=["item-1"],
            target_col="target",
            strategy=strategy,
        )


@pytest.mark.parametrize("strategy", sorted(FIXED_STRATEGIES, key=lambda s: s.value))
def test_compute_importance_order_raises_for_missing_target(df_train_mi, item_cols, strategy):
    with pytest.raises(ValueError, match="not found in df_train"):
        compute_importance_order(
            df_train=df_train_mi,
            item_cols=item_cols,
            target_col="missing-col",
            strategy=strategy,
        )


# ---------------------------------------------------------------------------
# FIXED_STRATEGIES set
# ---------------------------------------------------------------------------

def test_fixed_strategies_set_contents():
    assert SelectionStrategy.RUBRIC_BASED_MAXREV in FIXED_STRATEGIES
    assert SelectionStrategy.RUBRIC_BASED_MRMR in FIXED_STRATEGIES
    assert SelectionStrategy.RANDOM not in FIXED_STRATEGIES


# ---------------------------------------------------------------------------
# Inter-item strategies: fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def df_train_inter_item():
    """Training data with a known *inter-item* mutual information structure.

    item-1, item-2 and item-3 are copies of a three-level factor `g`, so each carries
    I(Q_i; Q_j) = H(g) ~ 1.10 nats with the other two and ~0 with the rest. item-4 and
    item-5 are copies of an independent two-level factor `h`: I = H(h) ~ 0.69 nats with
    each other and ~0 with the `g` cluster.

    Mean mutual information over the full item set F (|F| - 1 = 4) therefore ranks the
    dense `g` cluster above the `h` pair: 2*H(g)/4 ~ 0.55 against H(h)/4 ~ 0.17. Once
    item-2 and item-3 are answered the ranking flips, because item-1 has no remaining
    partner while item-4 and item-5 still have each other.
    """
    rng = np.random.default_rng(1)
    n = 300
    g = rng.integers(0, 3, size=n)
    h = rng.integers(0, 2, size=n)
    return pd.DataFrame({
        "item-1": g,
        "item-2": g,  # identical to item-1
        "item-3": g,  # identical to item-1
        "item-4": h,
        "item-5": h,  # identical to item-4
        "target": g + h,
    })


@pytest.fixture
def redundancy_inter_item(df_train_inter_item, item_cols):
    return compute_redundancy_matrix(df_train_inter_item, item_cols)


@pytest.fixture
def df_train_symmetric(item_cols):
    """Every item is a copy of the same factor, so all pairwise MI values are equal."""
    rng = np.random.default_rng(2)
    g = rng.integers(0, 3, size=200)
    data = {c: g for c in item_cols}
    data["target"] = g
    return pd.DataFrame(data)


def _mean_mi(redundancy, item, others):
    """Mean mutual information between `item` and `others`; 0.0 for an empty set."""
    others = [j for j in others if j != item]
    if not others:
        return 0.0
    return float(redundancy.loc[item, others].sum() / len(others))


def _relevance_unanswered_score(redundancy, item, U, A):
    """Inter-Item Relevance (Unanswered), spec section 4.5."""
    return _mean_mi(redundancy, item, U)


def _mrmr_unanswered_score(redundancy, item, U, A):
    """Inter-Item mRMR (Unanswered), spec section 4.6."""
    return _mean_mi(redundancy, item, U) - _mean_mi(redundancy, item, A)


def _mrmr_all_score(redundancy, item, U, A):
    """Inter-Item mRMR (All), spec section 4.7.

    Kept as a test-only helper: the strategy is excluded from the comparison because it
    selects the same candidate as the (Unanswered) variant, which this module verifies.
    """
    F = list(redundancy.index)
    return _mean_mi(redundancy, item, F) - _mean_mi(redundancy, item, A)


def _argmax(redundancy, score_fn, U, A):
    """Highest-scoring candidate, ties broken by the order of the redundancy index."""
    candidates = [c for c in redundancy.index if c in set(U)]
    return max(candidates, key=lambda c: score_fn(redundancy, c, U, A))


# ---------------------------------------------------------------------------
# compute_redundancy_matrix
# ---------------------------------------------------------------------------

def test_compute_redundancy_matrix_is_symmetric_with_zero_diagonal(redundancy_inter_item, item_cols):
    assert list(redundancy_inter_item.index) == item_cols
    assert list(redundancy_inter_item.columns) == item_cols
    assert np.allclose(redundancy_inter_item.to_numpy(), redundancy_inter_item.to_numpy().T)
    assert np.allclose(np.diag(redundancy_inter_item.to_numpy()), 0.0)


def test_compute_redundancy_matrix_reflects_known_structure(redundancy_inter_item):
    # the `g` cluster shares more information internally than with the `h` pair
    assert redundancy_inter_item.loc["item-1", "item-2"] > redundancy_inter_item.loc["item-4", "item-5"]
    assert redundancy_inter_item.loc["item-1", "item-4"] < redundancy_inter_item.loc["item-4", "item-5"]


def test_compute_redundancy_matrix_reproducible(df_train_inter_item, item_cols):
    first = compute_redundancy_matrix(df_train_inter_item, item_cols)
    second = compute_redundancy_matrix(df_train_inter_item, item_cols)
    assert np.allclose(first.to_numpy(), second.to_numpy())


def test_compute_redundancy_matrix_raises_when_less_than_two_items(df_train_inter_item):
    with pytest.raises(ValueError, match="At least two item columns"):
        compute_redundancy_matrix(df_train_inter_item, ["item-1"])


def test_compute_redundancy_matrix_raises_for_missing_columns(df_train_inter_item, item_cols):
    with pytest.raises(ValueError, match="not found in df_train"):
        compute_redundancy_matrix(df_train_inter_item, item_cols + ["missing-col"])


def test_compute_redundancy_matrix_handles_missing_values(df_train_inter_item, item_cols):
    df = df_train_inter_item.copy()
    df.loc[df.index[:50], "item-5"] = np.nan  # pairwise dropna, estimated on training data only

    redundancy = compute_redundancy_matrix(df, item_cols)

    assert np.isfinite(redundancy.to_numpy()).all()
    assert redundancy.loc["item-4", "item-5"] > 0.0


# ---------------------------------------------------------------------------
# Inter-Item Relevance (All)
# ---------------------------------------------------------------------------

def test_inter_item_relevance_all_ranks_dense_cluster_first(df_train_inter_item, item_cols):
    order = compute_importance_order(
        df_train=df_train_inter_item,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
    )
    assert set(order[:3]) == {"item-1", "item-2", "item-3"}
    assert set(order[3:]) == {"item-4", "item-5"}


def test_inter_item_relevance_all_ignores_the_target(df_train_inter_item, item_cols):
    """Section 4.4 scores items against each other, never against the rubric score."""
    shuffled = df_train_inter_item.copy()
    shuffled["target"] = shuffled["target"].sample(frac=1.0, random_state=0).to_numpy()

    kwargs = dict(
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
    )
    original = compute_importance_order(df_train=df_train_inter_item, **kwargs)
    reshuffled = compute_importance_order(df_train=shuffled, **kwargs)

    assert original == reshuffled


def test_inter_item_relevance_all_order_is_not_recomputed(df_train_inter_item, item_cols):
    """Section 4.4 keeps one fixed order: neither A nor U changes the ranking."""
    order = compute_importance_order(
        df_train=df_train_inter_item,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
    )
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
        seed=0,
        importance_order=order,
    )

    selected = []
    remaining = item_cols.copy()
    while remaining:
        q = selector.select(remaining, A=selected.copy())
        selected.append(q)
        remaining.remove(q)

    assert selected == order


def test_inter_item_relevance_all_differs_from_unanswered(df_train_inter_item, item_cols, redundancy_inter_item):
    """RQ3: scoring against F and against U pick different items once U shrinks."""
    order = compute_importance_order(
        df_train=df_train_inter_item,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
    )
    fixed = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
        seed=0,
        importance_order=order,
    )
    adaptive = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )

    U = ["item-1", "item-4", "item-5"]  # item-1 lost both of its partners
    assert fixed.select(U) == "item-1"
    assert adaptive.select(U) == "item-4"


# ---------------------------------------------------------------------------
# Inter-Item Relevance (Unanswered)
# ---------------------------------------------------------------------------

def test_inter_item_relevance_unanswered_prefers_dense_cluster_at_start(redundancy_inter_item, item_cols):
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    # with U = F, the `g` cluster has two partners each against one for the `h` pair
    assert selector.select(item_cols) == "item-1"


def test_inter_item_relevance_unanswered_recomputes_on_u(redundancy_inter_item):
    """The score must follow the current U, not a precomputed order."""
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    # item-2 and item-3 are gone, so item-1 has no informative partner left in U
    assert selector.select(["item-1", "item-4", "item-5"]) == "item-4"
    # item-1 keeps a partner, so the `g` cluster wins again
    assert selector.select(["item-1", "item-2", "item-4"]) == "item-1"


def test_inter_item_relevance_unanswered_ignores_answered_set(redundancy_inter_item):
    """Section 4.5 depends on U alone: neither A nor the answer values x_A enter it."""
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    U = ["item-1", "item-2", "item-4"]

    baseline = selector.select(U)
    for A in ([], ["item-3"], ["item-3", "item-5"], ["item-5"]):
        assert selector.select(U, A=A) == baseline


# ---------------------------------------------------------------------------
# Inter-Item mRMR (Unanswered)
# ---------------------------------------------------------------------------

def test_inter_item_mrmr_unanswered_penalises_redundant_candidate(redundancy_inter_item):
    """RQ4: a candidate redundant with the answered set is demoted."""
    relevance = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    mrmr = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )

    U = ["item-1", "item-2", "item-4"]
    A = ["item-3"]  # a copy of item-1 and item-2

    assert relevance.select(U, A=A) == "item-1"  # no penalty: the `g` cluster wins
    assert mrmr.select(U, A=A) == "item-4"       # penalised: item-1/item-2 add nothing new


def test_inter_item_mrmr_unanswered_matches_relevance_when_a_is_empty(redundancy_inter_item, item_cols):
    """Section 4.8: with A = {} the redundancy term is dropped."""
    relevance = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    mrmr = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )

    for U in (item_cols, ["item-1", "item-4", "item-5"], ["item-2", "item-3", "item-4"]):
        assert mrmr.select(U) == relevance.select(U)
        assert mrmr.select(U, A=[]) == relevance.select(U)


def test_inter_item_mrmr_unanswered_scores_match_the_specification(redundancy_inter_item):
    """The selected candidate maximizes the section 4.6 score."""
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )
    U = ["item-1", "item-2", "item-4"]
    A = ["item-3", "item-5"]

    picked = selector.select(U, A=A)
    best = max(_mrmr_unanswered_score(redundancy_inter_item, c, U, A) for c in U)

    assert np.isclose(_mrmr_unanswered_score(redundancy_inter_item, picked, U, A), best)


# --------------------------------------------------------------
# Adaptive strategies: common behaviour and boundary conditions
# --------------------------------------------------------------

@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_requires_redundancy(strategy):
    with pytest.raises(ValueError, match="redundancy"):
        QuestionSelector(strategy=strategy, seed=0)


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_returns_last_item_when_u_has_one_element(strategy, redundancy_inter_item):
    """|U| = 1: the score is undefined (|U| - 1 = 0), so the item is returned directly."""
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)

    assert selector.select(["item-3"]) == "item-3"
    assert selector.select(["item-3"], A=["item-1", "item-2"]) == "item-3"


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_raises_on_empty_u(strategy, redundancy_inter_item):
    """|U| = 0: selection is impossible and fails explicitly, as for every strategy."""
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)

    with pytest.raises(ValueError, match="empty"):
        selector.select([])


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_raises_for_unknown_item(strategy, redundancy_inter_item):
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)

    with pytest.raises(ValueError, match="redundancy matrix"):
        selector.select(["unknown-item"])


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_raises_for_overlapping_items(strategy, redundancy_inter_item):
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)

    with pytest.raises(ValueError, match="both answered and unanswered"):
        selector.select(["item-1", "item-2"], A=["item-1"])


def test_inter_item_mrmr_raises_for_unknown_answered_item(redundancy_inter_item):
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
        seed=0,
        redundancy=redundancy_inter_item,
    )

    with pytest.raises(ValueError, match="Answered items missing"):
        selector.select(["item-1", "item-2"], A=["unknown-item"])


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_exhausts_without_repeats(strategy, redundancy_inter_item, item_cols):
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)

    selected = []
    remaining = item_cols.copy()
    while remaining:
        q = selector.select(remaining, A=selected.copy())
        assert q in remaining
        assert q not in selected
        selected.append(q)
        remaining.remove(q)

    assert len(selected) == len(item_cols)
    assert set(selected) == set(item_cols)


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_is_reproducible(strategy, redundancy_inter_item, item_cols):
    first = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy_inter_item)
    second = QuestionSelector(strategy=strategy, seed=999, redundancy=redundancy_inter_item)

    # deterministic: the RNG seed must not affect a mutual-information based score
    assert first.select(item_cols, A=[]) == second.select(item_cols, A=[])


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_tie_break_follows_item_order(strategy, df_train_symmetric, item_cols):
    """All pairwise MI values are equal, so ties resolve to the configured item order."""
    redundancy = compute_redundancy_matrix(df_train_symmetric, item_cols)
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy)

    selected = []
    remaining = item_cols.copy()
    while len(remaining) > 1:
        q = selector.select(remaining, A=selected.copy())
        selected.append(q)
        remaining.remove(q)

    assert selected == item_cols[:-1]


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_tie_break_ignores_candidate_order(strategy, df_train_symmetric, item_cols):
    """The order in which C happens to be listed must not change the pick."""
    redundancy = compute_redundancy_matrix(df_train_symmetric, item_cols)
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy)

    assert selector.select(list(reversed(item_cols))) == selector.select(item_cols)


@pytest.mark.parametrize("strategy", sorted(ADAPTIVE_STRATEGIES, key=lambda s: s.value))
def test_adaptive_strategy_handles_missing_values(strategy, df_train_inter_item, item_cols):
    df = df_train_inter_item.copy()
    df.loc[df.index[:40], "item-2"] = np.nan

    redundancy = compute_redundancy_matrix(df, item_cols)
    selector = QuestionSelector(strategy=strategy, seed=0, redundancy=redundancy)

    assert selector.select(item_cols, A=[]) in item_cols


# ---------------------------------------------------------------------------
# Selection Strategies: Inter-Item mRMR (All) vs (Unanswered)
# ---------------------------------------------------------------------------

def _equivalence_datasets():
    """Small synthetic datasets with assorted inter-item dependence structures."""
    datasets = []
    for seed in (3, 11, 29):
        rng = np.random.default_rng(seed)
        n = 150
        cols = [f"item-{i}" for i in range(1, 6)]
        factors = rng.integers(0, 3, size=(n, 2))
        data = {}
        for k, c in enumerate(cols):
            base = factors[:, k % 2]
            noise = rng.integers(0, 3, size=n)
            # a per-item mix of a shared factor and noise gives uneven redundancy
            data[c] = np.where(rng.random(n) < 0.3 + 0.15 * k, base, noise)
        datasets.append((cols, pd.DataFrame(data)))
    return datasets


@pytest.mark.parametrize("cols,df", _equivalence_datasets())
def test_inter_item_mrmr_all_selects_the_same_candidate(cols, df):
    """Section 4.7: with |U| >= 2 the (All) variant is a positive multiple of (Unanswered).

    score_all(i) = ((|U| - 1) / (|F| - 1)) * score_unanswered(i), because
    |F| - 1 = |A| + |U| - 1 makes the redundancy terms cancel exactly. The argmax is
    therefore identical, which is why the (All) variant is left out of the comparison.
    """
    redundancy = compute_redundancy_matrix(df, cols)
    selector = QuestionSelector(
        strategy=SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
        seed=0,
        redundancy=redundancy,
    )

    checked = 0
    for mask in range(1 << len(cols)):
        A = [c for k, c in enumerate(cols) if mask >> k & 1]
        U = [c for c in cols if c not in A]
        if len(U) < 2:
            continue

        picked = selector.select(U, A=A)

        # the selected candidate also maximizes the section 4.7 score
        best_all = max(_mrmr_all_score(redundancy, c, U, A) for c in U)
        assert np.isclose(_mrmr_all_score(redundancy, picked, U, A), best_all), (A, U, picked)

        # and the proportionality itself holds candidate by candidate
        ratio = (len(U) - 1) / (len(cols) - 1)
        for c in U:
            assert np.isclose(
                _mrmr_all_score(redundancy, c, U, A),
                ratio * _mrmr_unanswered_score(redundancy, c, U, A),
            ), (A, U, c)

        checked += 1

    assert checked > 0


# ---------------------------------------------------------------------------
# Strategy sets
# ---------------------------------------------------------------------------

def test_strategy_sets_are_disjoint_and_cover_every_non_random_strategy():
    assert FIXED_STRATEGIES.isdisjoint(ADAPTIVE_STRATEGIES)
    assert FIXED_STRATEGIES | ADAPTIVE_STRATEGIES | {SelectionStrategy.RANDOM} == set(SelectionStrategy)


def test_inter_item_strategies_are_classified_correctly():
    assert SelectionStrategy.INTER_ITEM_RELEVANCE_ALL in FIXED_STRATEGIES
    assert SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED in ADAPTIVE_STRATEGIES
    assert SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED in ADAPTIVE_STRATEGIES
