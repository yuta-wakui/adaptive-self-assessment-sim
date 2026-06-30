# -*- coding: utf-8 -*-

"""
Question item selection during adaptive self-assessment simulations.
This module provides functions for selecting question items during adaptive self-assessment simulations.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/components/selector.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Question item selection during adaptive self-assessment

import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Optional

class SelectionStrategy(str, Enum):
    """
    Question selection strategies
    """
    RANDOM = "random"
    FIXED_CORRELATION = "fixed_correlation"
    FIXED_PARTIAL_REGRESSION = "fixed_partial_regression"
    FIXED_FEATURE_IMPORTANCE = "fixed_feature_importance"

FIXED_STRATEGIES = {
    SelectionStrategy.FIXED_CORRELATION,
    SelectionStrategy.FIXED_PARTIAL_REGRESSION,
    SelectionStrategy.FIXED_FEATURE_IMPORTANCE,
}

class QuestionSelector:
    """
    Question selector for adaptive self-assessment.

    Each selector instance maintains its own random number generator (RNG) state to ensure
    reproducibility and independence across users or folds.
    """

    def __init__(
            self,
            strategy: SelectionStrategy = SelectionStrategy.RANDOM,
            seed: Optional[int] = None,
            importance_order: Optional[List[str]] = None,
        ):
        """
        Initialize the QuestionSelector.

        Parameters:
        ----------
        strategy: SelectionStrategy
            Strategy for selecting question items (default: RANDOM)
        seed: Optional[int]
            Seed for the random number generator (default: None)
        importance_order: Optional[List[str]]
            Pre-sorted list of item names from most to least important.
            Required for fixed strategies (FIXED_CORRELATION, FIXED_PARTIAL_REGRESSION,
            FIXED_FEATURE_IMPORTANCE).
        """
        self.strategy = SelectionStrategy(strategy)
        self.rng = np.random.default_rng(seed)
        self.importance_order = importance_order

        if strategy in FIXED_STRATEGIES and importance_order is None:
            raise ValueError(
                f"importance_order must be provided for strategy '{strategy}'."
            )

    def select(self, C: List[str]) -> str:
        """
        Select a question item from the remaining list C based on the selector's strategy.

        Parameters:
        ----------
        C: list of str
            Remaining question items to select from
        Returns:
            str: Selected question item
        Raises:
        -------
        ValueError
            If 'C' is empty or no item in importance_order is found in C
        NotImplementedError
            If the strategy is not implemented
        """
        if not C:
            raise ValueError("The list of question items is empty.")

        if self.strategy == SelectionStrategy.RANDOM:
            return str(self.rng.choice(C))

        if self.strategy in FIXED_STRATEGIES:
            c_set = set(C)
            for item in self.importance_order:  # type: ignore[union-attr]
                if item in c_set:
                    return item
            raise ValueError(
                "No item from importance_order is found in C. "
                "Ensure importance_order covers all item columns."
            )

        raise NotImplementedError(f"Unknown selection strategy: {self.strategy}")


def compute_importance_order(
        df_train: pd.DataFrame,
        item_cols: List[str],
        target_col: str,
        strategy: SelectionStrategy,
        random_state: int = 42,
    ) -> List[str]:
    """
    Compute a fixed importance order of item_cols w.r.t. target_col.

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to rank
    target_col: str
        Target column (overall score)
    strategy: SelectionStrategy
        Must be one of the FIXED_STRATEGIES
    random_state: int
        Random seed (used for model-based strategies)

    Returns:
    -------
    List[str]
        Item column names sorted from most to least important
    """
    if strategy not in FIXED_STRATEGIES:
        raise ValueError(
            f"compute_importance_order is only for fixed strategies, got '{strategy}'."
        )

    if strategy == SelectionStrategy.FIXED_CORRELATION:
        corr = df_train[item_cols].corrwith(df_train[target_col]).abs()
        return corr.sort_values(ascending=False).index.tolist()

    if strategy == SelectionStrategy.FIXED_PARTIAL_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = df_train[item_cols].values
        y = df_train[target_col].values
        X_scaled = StandardScaler().fit_transform(X)
        model = LogisticRegression(max_iter=5000, random_state=random_state)
        model.fit(X_scaled, y)
        # For multi-class, take max absolute coefficient across classes per feature
        coef_abs = np.abs(model.coef_).max(axis=0)
        order = np.argsort(coef_abs)[::-1]
        return [item_cols[i] for i in order]

    if strategy == SelectionStrategy.FIXED_FEATURE_IMPORTANCE:
        from sklearn.ensemble import RandomForestClassifier

        X = df_train[item_cols].values
        y = df_train[target_col].values
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X, y)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        return [item_cols[i] for i in order]

    raise NotImplementedError(f"Unknown fixed strategy: {strategy}")