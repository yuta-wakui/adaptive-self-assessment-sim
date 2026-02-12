# -*= coding: utf-8 -*-
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

from adaptive_self_assessment.components.selector import (
    QuestionSelector,
    SelectionStrategy,
)

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