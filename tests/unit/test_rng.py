# -*- coding: utf-8 -*-
"""
Unit tests for the RNG component of adaptive self-assessment simulations.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/unit/test_rng.py
# Author: Yuta Wakui
# Date: 2026-02-11
# Description: Unit tests for RNG component

import pytest

from adaptive_self_assessment.components.rng import make_selector_seed

def test_make_selector_seed_reproducible():
    s1 = make_selector_seed(42, 3, "userA")
    s2 = make_selector_seed(42, 3, "userA")
    assert s1 == s2

def test_make_selector_seed_different_cv_seed():
    s1 = make_selector_seed(42, 3, "userA")
    s2 = make_selector_seed(43, 3, "userA")
    assert s1 != s2

def test_make_selector_seed_different_folds():
    s1 = make_selector_seed(42, 1, "userA")
    s2 = make_selector_seed(42, 2, "userA")
    assert s1 != s2

def test_make_selector_seed_different_users():
    s1 = make_selector_seed(42, 3, "userA")
    s2 = make_selector_seed(42, 3, "userB")
    assert s1 != s2