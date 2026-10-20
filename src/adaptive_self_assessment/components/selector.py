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
from typing import List, Optional, Sequence

from sklearn.metrics import mutual_info_score

class SelectionStrategy(str, Enum):
    """
    Question selection strategies
    """
    RANDOM = "random"
    RUBRIC_BASED_MAXREV = "rubric_based_maxrev"
    RUBRIC_BASED_MRMR = "rubric_based_mrmr"
    INTER_ITEM_RELEVANCE_ALL = "inter_item_relevance_all"
    INTER_ITEM_RELEVANCE_UNANSWERED = "inter_item_relevance_unanswered"
    INTER_ITEM_MRMR_UNANSWERED = "inter_item_mrmr_unanswered"

# Strategies that follow a fixed order computed once per fold from the training data
FIXED_STRATEGIES = {
    SelectionStrategy.RUBRIC_BASED_MAXREV,
    SelectionStrategy.RUBRIC_BASED_MRMR,
    SelectionStrategy.INTER_ITEM_RELEVANCE_ALL,
}

# Strategies that re-score the candidates at every step from the answered set A and the
# unanswered set U. They adapt to those *sets*, never to the respondent's answer values.
ADAPTIVE_STRATEGIES = {
    SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED,
    SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
}

# Adaptive strategies that subtract the redundancy against the answered set A
REDUNDANCY_PENALISED_STRATEGIES = {
    SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED,
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
            redundancy: Optional[pd.DataFrame] = None,
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
            Required for fixed-order strategies (Rubric-based MaxRev, Rubric-based mRMR,
            Inter-Item Relevance (All)).
        redundancy: Optional[pd.DataFrame]
            Square matrix of inter-item mutual information I(Q_i; Q_j), indexed and
            columned by item name. Required for adaptive strategies, which re-score the
            candidates at every step.
        """
        self.strategy = SelectionStrategy(strategy)
        self.rng = np.random.default_rng(seed)
        self.importance_order = importance_order
        self.redundancy = redundancy

        if self.strategy in FIXED_STRATEGIES and importance_order is None:
            raise ValueError(
                f"importance_order must be provided for strategy '{strategy}'."
            )

        if self.strategy in ADAPTIVE_STRATEGIES and redundancy is None:
            raise ValueError(
                f"redundancy must be provided for strategy '{strategy}'."
            )

    def select(self, C: List[str], A: Optional[Sequence[str]] = None) -> str:
        """
        Select a question item from the remaining list C based on the selector's strategy.

        Parameters:
        ----------
        C: list of str
            Remaining (unanswered) question items to select from, i.e. U
        A: Optional[Sequence[str]]
            Items already answered by the respondent or complemented above the confidence
            threshold, i.e. A. Only the *membership* of A is used, never the answer values.
            Ignored by every strategy except Inter-Item mRMR (Unanswered).
        Returns:
            str: Selected question item
        Raises:
        -------
        ValueError
            If 'C' is empty, if no item in importance_order is found in C, or if an item of
            C is missing from the redundancy matrix
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

        if self.strategy in ADAPTIVE_STRATEGIES:
            return self._select_adaptive(C, A)

        raise NotImplementedError(f"Unknown selection strategy: {self.strategy}")

    def _select_adaptive(self, C: List[str], A: Optional[Sequence[str]]) -> str:
        """
        Select the candidate maximizing the inter-item score recomputed for the current sets.

        Inter-Item Relevance (Unanswered):
            score(i; U) = (1/(|U|-1)) * sum_{j in U\\{i}} I(Q_i; Q_j)

        Inter-Item mRMR (Unanswered):
            score(i; A, U) = score(i; U) - (1/|A|) * sum_{j in A} I(Q_i; Q_j)

        The redundancy term is dropped when A is empty.
        With a single candidate left, that candidate is returned directly.
        Ties are broken by the order of the redundancy matrix index.

        Parameters
        ----------
        C: List[str]
            Remaining (unanswered) question items, i.e. U
        A: Optional[Sequence[str]]
            Answered or complemented items, i.e. A

        Returns
        -------
        str
            Selected question item

        Raises
        ------
        ValueError
            If answered and unanswered items overlap, or required items
            are missing from the redundancy matrix.
        """
        redundancy: pd.DataFrame = self.redundancy  # type: ignore[assignment]

        c_set = set(C)
        a_set = set(A or ())

        # Validate that answered and unanswered items do not overlap
        overlap = c_set & a_set
        if overlap:
            raise ValueError(
                f"Items cannot be both answered and unanswered: {sorted(overlap)}."
            )

        # Validate that all unanswered items exist in the MI matrix
        unknown = c_set - set(redundancy.index)
        if unknown:
            raise ValueError(
                f"Items missing from the redundancy matrix: {sorted(unknown)}. "
                "Ensure the matrix covers all item columns."
            )

        # Validate answered items for redundancy-penalised strategies
        if self.strategy in REDUNDANCY_PENALISED_STRATEGIES:
            unknown_answered = a_set - set(redundancy.index)
            if unknown_answered:
                raise ValueError(
                    "Answered items missing from the redundancy matrix: "
                    f"{sorted(unknown_answered)}."
                )

        # Candidates in the stable order of the matrix index
        candidates = [c for c in redundancy.index if c in c_set]

        # |U| = 1: return the last remaining item
        if len(candidates) == 1:
            return candidates[0]

        # Calculate mean mutual information with other unanswered items
        sub = redundancy.loc[candidates, candidates].to_numpy(dtype=float)
        scores = sub.sum(axis=1) / (len(candidates) - 1)

        # Subtract redundancy against answered items for mRMR
        if self.strategy in REDUNDANCY_PENALISED_STRATEGIES:
            answered = [j for j in redundancy.index if j in a_set]

            if answered:
                penalty = (
                    redundancy.loc[candidates, answered]
                    .to_numpy(dtype=float)
                    .mean(axis=1)
                )
                scores = scores - penalty

        # Select the candidate with the highest score
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]


def compute_importance_order(
        df_train: pd.DataFrame,
        item_cols: List[str],
        target_col: str,
        strategy: SelectionStrategy,
    ) -> List[str]:
    """
    Compute a fixed question order of item_cols w.r.t. target_col.

    All supported strategies score items with mutual information estimated from the
    training data, so the resulting order does not depend on the respondent's answers
    and is shared by every user within a fold.

    Rubric-based MaxRev ranks items by relevance I(Q_i; Y) alone. Rubric-based mRMR
    selects items sequentially, maximizing relevance minus the mean redundancy
    I(Q_i; Q_j) against the items already selected. Inter-Item Relevance (All) ignores
    the target and ranks items by their mean mutual information with every other item.

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to rank
    target_col: str
        Target column (overall/rubric score). Validated for every fixed strategy, even
        though Inter-Item Relevance (All) does not score against it.
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

    if strategy == SelectionStrategy.INTER_ITEM_RELEVANCE_ALL:
        return _inter_item_relevance_all_order(df_train, item_cols)

    relevance = _relevance(df_train, item_cols, target_col)

    if strategy == SelectionStrategy.RUBRIC_BASED_MAXREV:
        # Q* = argmax I(Q_i; Y), ties broken by the order of item_cols
        return sorted(item_cols, key=lambda c: -relevance[c])

    if strategy == SelectionStrategy.RUBRIC_BASED_MRMR:
        return _mrmr_order(df_train, item_cols, relevance)

    raise NotImplementedError(f"Unknown fixed strategy: {strategy}")


def compute_redundancy_matrix(df_train: pd.DataFrame, item_cols: List[str]) -> pd.DataFrame:
    """
    Compute the inter-item mutual information matrix used by the adaptive strategies.

    The matrix is estimated once per fold from the training data only, so the per-step
    re-scoring never touches the respondent's answers.

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
    if len(item_cols) < 2:
        raise ValueError("At least two item columns are required for adaptive strategies.")

    missing = [c for c in item_cols if c not in df_train.columns]
    if missing:
        raise ValueError(f"item_cols not found in df_train: {missing}")

    return _redundancy_matrix(df_train, item_cols)


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


def _inter_item_relevance_all_order(
        df_train: pd.DataFrame,
        item_cols: List[str],
    ) -> List[str]:
    """
    Rank item_cols by their mean mutual information with every other item.

        score(Q_i) = (1/(|F|-1)) * sum_{Q_j in F\\{Q_i}} I(Q_i; Q_j)

    The score is computed over the full item set F once per fold, so the order is fixed:
    it is not recomputed as items move from the unanswered to the answered set. Ties are
    broken by the order of item_cols.

    Parameters:
    ----------
    df_train: pd.DataFrame
        Training data
    item_cols: List[str]
        Item column names to rank

    Returns:
    -------
    List[str]
        Item column names sorted from most to least important
    """
    redundancy = _redundancy_matrix(df_train, item_cols)

    # the diagonal is zero, so the row sum already excludes the item itself
    scores = redundancy.sum(axis=1) / (len(item_cols) - 1)

    return sorted(item_cols, key=lambda c: -scores[c])


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
