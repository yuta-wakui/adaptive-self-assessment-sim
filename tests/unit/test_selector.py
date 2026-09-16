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
    compute_importance_order,
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
# compute_importance_order: MaxRev
# ---------------------------------------------------------------------------

def test_maxrev_ranks_by_relevance(df_train_mi, item_cols):
    """Items informative about the target outrank independent noise items."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MaxRev,
    )
    assert order[:3] == ["item-1", "item-2", "item-3"]
    assert set(order[3:]) == {"item-4", "item-5"}


def test_maxrev_keeps_redundant_item_near_top(df_train_mi, item_cols):
    """MaxRev ignores redundancy, so the duplicated item-2 stays at rank 2."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MaxRev,
    )
    assert order[1] == "item-2"


def test_maxrev_returns_all_items(df_train_mi, item_cols):
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MaxRev,
    )
    assert len(order) == len(item_cols)
    assert set(order) == set(item_cols)


# ---------------------------------------------------------------------------
# compute_importance_order: mRMR
# ---------------------------------------------------------------------------

def test_mrmr_first_item_is_most_relevant(df_train_mi, item_cols):
    """With S empty, the first pick falls back to the Max-Relevance criterion."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MRMR,
    )
    assert order[0] == "item-1"


def test_mrmr_demotes_redundant_item(df_train_mi, item_cols):
    """item-2 duplicates item-1, so mRMR prefers the less redundant item-3."""
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MRMR,
    )
    assert order[1] == "item-3"
    assert order.index("item-2") > order.index("item-3")


def test_mrmr_differs_from_maxrev(df_train_mi, item_cols):
    """The redundancy term must actually change the resulting order."""
    kwargs = dict(df_train=df_train_mi, item_cols=item_cols, target_col="target")

    maxrev = compute_importance_order(strategy=SelectionStrategy.MaxRev, **kwargs)
    mrmr = compute_importance_order(strategy=SelectionStrategy.MRMR, **kwargs)

    assert maxrev != mrmr
    assert set(maxrev) == set(mrmr)


def test_mrmr_returns_all_items(df_train_mi, item_cols):
    order = compute_importance_order(
        df_train=df_train_mi,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.MRMR,
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
    assert SelectionStrategy.MaxRev in FIXED_STRATEGIES
    assert SelectionStrategy.MRMR in FIXED_STRATEGIES
    assert SelectionStrategy.RANDOM not in FIXED_STRATEGIES
