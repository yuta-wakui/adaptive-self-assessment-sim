# -*- coding: utf-8 -*-
"""
Unit tests for the ModelStore class.
This module tests the functionality of the ModelStore class for caching trained models.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/unit/test_model_store.py
# Author: Yuta Wakui
# Date: 2026-02-11
# Description: Unit tests for ModelStore

from sklearn.base import BaseEstimator

from adaptive_self_assessment.components.model_store import ModelStore

class DummyEstimator:
    """A lightweight stand-in for sklearn estimator in unit tests."""
    pass


def test_model_store_get_returns_none_when_missing():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    assert store.get(key) is None

def test_model_store_set_then_get_returns_same_object():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    model = DummyEstimator()

    store.set(key, model)
    assert store.get(key) is model

def test_model_store_overwrites_existing_key():
    store = ModelStore()
    key = ("ws1", 0, "item_1")
    model1 = DummyEstimator()
    model2 = DummyEstimator()

    store.set(key, model1)
    store.set(key, model2)
    assert store.get(key) is model2

def test_model_store_size_increments_and_clear_resets():
    store = ModelStore()
    key1 = ("ws1", 0, "item_1")
    key2 = ("ws1", 0, "item_2")
    model1 = DummyEstimator()
    model2 = DummyEstimator()

    assert store.size() == 0

    store.set(key1, model1)
    assert store.size() == 1

    store.set(key2, model2)
    assert store.size() == 2

    store.clear()
    assert store.size() == 0