# -*- coding: utf-8 -*-
"""
Unit tests for common.py: CVConfig, load_cv_config, validate_cv_config.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/unit/test_common.py
# Author: Yuta Wakui
# Date: 2026-04-04

import pytest
from adaptive_self_assessment.simulation.common import (
    CVConfig,
    load_cv_config,
    validate_cv_config,
)


class TestValidateCvConfig:
    def test_kfold_valid(self):
        cv = CVConfig(method="kfold", folds=5, stratified=True, random_seed=42)
        validate_cv_config(cv)  # should not raise

    def test_loo_valid(self):
        cv = CVConfig(method="loo", folds=1, stratified=False, random_seed=42)
        validate_cv_config(cv)  # should not raise even with folds=1

    def test_kfold_invalid_folds(self):
        cv = CVConfig(method="kfold", folds=1, stratified=False, random_seed=42)
        with pytest.raises(ValueError, match="cv.folds must be >= 2"):
            validate_cv_config(cv)

    def test_invalid_method(self):
        cv = CVConfig(method="stratified_loo", folds=5, stratified=True, random_seed=42)
        with pytest.raises(ValueError, match="cv.method must be 'kfold' or 'loo'"):
            validate_cv_config(cv)


class TestLoadCvConfig:
    def test_default_is_kfold(self):
        cfg = {"cv": {"folds": 5, "stratified": True, "random_seed": 42}}
        cv = load_cv_config(cfg)
        assert cv.method == "kfold"
        assert cv.folds == 5
        assert cv.stratified is True
        assert cv.random_seed == 42

    def test_explicit_kfold(self):
        cfg = {"cv": {"method": "kfold", "folds": 3, "stratified": False, "random_seed": 0}}
        cv = load_cv_config(cfg)
        assert cv.method == "kfold"
        assert cv.folds == 3
        assert cv.stratified is False

    def test_loo_method(self):
        cfg = {"cv": {"method": "loo", "folds": 5, "stratified": True, "random_seed": 42}}
        cv = load_cv_config(cfg)
        assert cv.method == "loo"

    def test_loo_ignores_folds_validation(self):
        # folds=1 is invalid for kfold but should be accepted for loo
        cfg = {"cv": {"method": "loo", "folds": 1, "stratified": False, "random_seed": 42}}
        cv = load_cv_config(cfg)
        assert cv.method == "loo"

    def test_invalid_method_raises(self):
        cfg = {"cv": {"method": "invalid", "folds": 5, "stratified": True, "random_seed": 42}}
        with pytest.raises(ValueError, match="cv.method must be 'kfold' or 'loo'"):
            load_cv_config(cfg)

    def test_stratified_must_be_bool(self):
        cfg = {"cv": {"method": "kfold", "folds": 5, "stratified": "yes", "random_seed": 42}}
        with pytest.raises(ValueError, match="cv.stratified must be a boolean"):
            load_cv_config(cfg)

    def test_empty_cv_section_uses_defaults(self):
        cv = load_cv_config({})
        assert cv.method == "kfold"
        assert cv.folds == 5
        assert cv.random_seed == 42
