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

from sklearn.metrics import mutual_info_score

class SelectionStrategy(str, Enum):
    """
    Question selection strategies
    """
    RANDOM = "random"
    MaxRev = "maxrev"
    MRMR = "mrmr"

# Strategies that follow a fixed order computed once per fold from the training data
FIXED_STRATEGIES = {
    SelectionStrategy.MaxRev,
    SelectionStrategy.MRMR,
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
            Required for fixed-order strategies (MaxRev, MRMR).
        """
        self.strategy = SelectionStrategy(strategy)
        self.rng = np.random.default_rng(seed)
        self.importance_order = importance_order

        if self.strategy in FIXED_STRATEGIES and importance_order is None:
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
    ) -> List[str]:
    """
    Compute a fixed question order of item_cols w.r.t. target_col.

    Both supported strategies score items with mutual information estimated from the
    training data, so the resulting order does not depend on the respondent's answers
    and is shared by every user within a fold.

    MaxRev ranks items by relevance I(Q_i; Y) alone. MRMR selects items sequentially,
    maximizing relevance minus the mean redundancy I(Q_i; Q_j) against the items
    already selected.

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to rank
    target_col: str
        Target column (overall/rubric score)
    strategy: SelectionStrategy
        Must be one of the FIXED_STRATEGIES

    Returns:
    -------
    List[str]
        Item column names sorted from most to least important
    """
    if strategy not in FIXED_STRATEGIES:
        raise ValueError(
            f"compute_importance_order is only for fixed strategies, got '{strategy}'."
        )

    if len(item_cols) < 2:
        raise ValueError("At least two item columns are required for fixed strategies.")

    if target_col not in df_train.columns:
        raise ValueError(f"target_col '{target_col}' not found in df_train.")

    relevance = _relevance(df_train, item_cols, target_col)

    if strategy == SelectionStrategy.MaxRev:
        # Q* = argmax I(Q_i; Y), ties broken by the order of item_cols
        return sorted(item_cols, key=lambda c: -relevance[c])

    if strategy == SelectionStrategy.MRMR:
        return _mrmr_order(df_train, item_cols, relevance)

    raise NotImplementedError(f"Unknown fixed strategy: {strategy}")


def _mutual_information(x: pd.Series, y: pd.Series) -> float:
    """
    Mutual information (in nats) between two discrete columns.

    Checklist answers and rubric scores are discrete, so the exact contingency-table
    estimator is used rather than a continuous approximation. Rows where either value
    is missing are dropped pairwise.

    Parameters:
    ----------
    x: pd.Series
        First discrete column
    y: pd.Series
        Second discrete column

    Returns:
    -------
    float
        Mutual information I(x; y); 0.0 if fewer than two rows remain
    """
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return 0.0
    return float(mutual_info_score(pair.iloc[:, 0], pair.iloc[:, 1]))


def _relevance(df_train: pd.DataFrame, item_cols: List[str], target_col: str) -> pd.Series:
    """
    Relevance of each item to the target: Relevance(Q_i) = I(Q_i; Y).

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to score
    target_col: str
        Target column (overall/rubric score)

    Returns:
    -------
    pd.Series
        Mutual information with the target, indexed by item column name
    """
    return pd.Series(
        {c: _mutual_information(df_train[c], df_train[target_col]) for c in item_cols},
        dtype=float,
    )


def _redundancy_matrix(df_train: pd.DataFrame, item_cols: List[str]) -> pd.DataFrame:
    """
    Pairwise redundancy between items: Redundancy(Q_i, Q_j) = I(Q_i; Q_j).

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to score

    Returns:
    -------
    pd.DataFrame
        Symmetric matrix of mutual information between items, with a zero diagonal
    """
    m = len(item_cols)
    matrix = np.zeros((m, m))

    # mutual information is symmetric, so only the upper triangle is computed
    for i in range(m):
        for j in range(i + 1, m):
            mi = _mutual_information(df_train[item_cols[i]], df_train[item_cols[j]])
            matrix[i, j] = mi
            matrix[j, i] = mi

    return pd.DataFrame(matrix, index=item_cols, columns=item_cols)


def _mrmr_order(
        df_train: pd.DataFrame,
        item_cols: List[str],
        relevance: pd.Series,
    ) -> List[str]:
    """
    Sequentially rank item_cols by mRMR (minimum redundancy maximum relevance).

    At each step, selects the unselected item Q_i maximizing:
        I(Q_i; Y) - (1/|S|) * sum_{Q_j in S} I(Q_i; Q_j)
    where S is the set of already selected items. The first item is selected by
    relevance alone, since redundancy is undefined for S = {}. Ties are broken by
    the order of item_cols.

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to rank
    relevance: pd.Series
        Mutual information with the target, indexed by item column name

    Returns:
    -------
    List[str]
        Item column names sorted from most to least important
    """
    redundancy = _redundancy_matrix(df_train, item_cols)

    remaining = list(item_cols)
    selected: List[str] = []

    def mrmr_score(c: str) -> float:
        if not selected:
            return float(relevance[c])
        return float(relevance[c] - redundancy.loc[c, selected].mean())

    while remaining:
        best = max(remaining, key=mrmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected
