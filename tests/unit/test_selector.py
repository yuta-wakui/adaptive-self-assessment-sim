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
def df_train_corr():
    """Training data where item-1 correlates perfectly with target, others weakly."""
    rng = np.random.default_rng(0)
    n = 50
    target = rng.integers(0, 5, size=n)
    data = {
        "item-1": target,                             # perfect correlation
        "item-2": rng.integers(0, 3, size=n),         # weak
        "item-3": rng.integers(0, 3, size=n),         # weak
        "item-4": rng.integers(0, 3, size=n),         # weak
        "item-5": rng.integers(0, 3, size=n),         # weak
        "target": target,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# RANDOM strategy (existing)
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
# Fixed strategies: QuestionSelector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
])
def test_fixed_strategy_requires_importance_order(strategy):
    with pytest.raises(ValueError, match="importance_order"):
        QuestionSelector(strategy=strategy, seed=0)


@pytest.mark.parametrize("strategy", [
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
])
def test_fixed_strategy_selects_highest_ranked_available(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    # With all items available, should pick item-1 (top of importance_order)
    assert selector.select(items) == "item-1"


@pytest.mark.parametrize("strategy", [
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
])
def test_fixed_strategy_skips_removed_items(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    remaining = items.copy()
    remaining.remove("item-1")  # top item already answered

    assert selector.select(remaining) == "item-2"


@pytest.mark.parametrize("strategy", [
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
])
def test_fixed_strategy_exhausts_in_order(strategy, items, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    selected = []
    remaining = items.copy()

    while remaining:
        q = selector.select(remaining)
        selected.append(q)
        remaining.remove(q)

    assert selected == importance_order


@pytest.mark.parametrize("strategy", [
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
])
def test_fixed_strategy_raises_when_no_match(strategy, importance_order):
    selector = QuestionSelector(strategy=strategy, seed=0, importance_order=importance_order)

    # C contains items not in importance_order
    with pytest.raises(ValueError):
        selector.select(["unknown-item"])


# ---------------------------------------------------------------------------
# compute_importance_order
# ---------------------------------------------------------------------------

def test_compute_importance_order_correlation_top_item(df_train_corr):
    item_cols = ["item-1", "item-2", "item-3", "item-4", "item-5"]
    order = compute_importance_order(
        df_train=df_train_corr,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.FIXED_CORRELATION,
    )
    assert order[0] == "item-1"
    assert set(order) == set(item_cols)


def test_compute_importance_order_correlation_returns_all_items(df_train_corr):
    item_cols = ["item-1", "item-2", "item-3", "item-4", "item-5"]
    order = compute_importance_order(
        df_train=df_train_corr,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.FIXED_CORRELATION,
    )
    assert len(order) == len(item_cols)


def test_compute_importance_order_partial_regression_returns_all_items(df_train_corr):
    item_cols = ["item-1", "item-2", "item-3", "item-4", "item-5"]
    order = compute_importance_order(
        df_train=df_train_corr,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.FIXED_PARTIAL_REGRESSION,
        random_state=42,
    )
    assert len(order) == len(item_cols)
    assert set(order) == set(item_cols)


def test_compute_importance_order_feature_importance_returns_all_items(df_train_corr):
    item_cols = ["item-1", "item-2", "item-3", "item-4", "item-5"]
    order = compute_importance_order(
        df_train=df_train_corr,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
        random_state=42,
    )
    assert len(order) == len(item_cols)
    assert set(order) == set(item_cols)


def test_compute_importance_order_reproducible(df_train_corr):
    item_cols = ["item-1", "item-2", "item-3", "item-4", "item-5"]
    kwargs = dict(
        df_train=df_train_corr,
        item_cols=item_cols,
        target_col="target",
        strategy=SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
        random_state=42,
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


# ---------------------------------------------------------------------------
# FIXED_STRATEGIES set
# ---------------------------------------------------------------------------

def test_fixed_strategies_set_contents():
    assert SelectionStrategy.FIXED_CORRELATION in FIXED_STRATEGIES
    assert SelectionStrategy.FIXED_PARTIAL_REGRESSION in FIXED_STRATEGIES
    assert SelectionStrategy.FIXED_FEATURE_IMPORTANCE in FIXED_STRATEGIES
    assert SelectionStrategy.RANDOM not in FIXED_STRATEGIES
