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
from enum import Enum
from typing import List, Optional

class SelectionStrategy(str, Enum):
    """
    Question selection strategies
    """
    RANDOM = "random"
    FEATURE_IMPORTANCE = "feature_importance"
    FEATURE_IMPORTANCE_WITH_NOISE = "feature_importance_with_noise"

class QuestionSelector:
    """
    Question selector for adaptive self-assessment.

    Each selector instance maintains its own random number generator (RNG) state to ensure
    reproducibly and independence across users or folds.
    """
    
    def __init__(
            self,
            strategy: SelectionStrategy = SelectionStrategy.RANDOM,
            seed: Optional[int] = None
        ):
        """
        Initialize the QuestionSelector.

        Parameters:
        ----------
        strategy: SelectionStrategy
            Strategy for selecting question items (default: RANDOM)
        seed: Optional[int]
            Seed for the random number generator (default: None)
        """
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

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
            If 'C" is empty
        NotImplementedError
            If the strategy is not implemented
        """
        if not C: 
            raise ValueError("The list of question items is empty.")
        
        if self.strategy == SelectionStrategy.RANDOM:
            return str(self.rng.choice(C))
        
        if self.strategy == SelectionStrategy.FEATURE_IMPORTANCE:
            raise NotImplementedError("FEATURE_IMPORTANCE not yet implemented.")

        if self.strategy == SelectionStrategy.FEATURE_IMPORTANCE_WITH_NOISE:
            raise NotImplementedError(
                "FEATURE_IMPORTANCE_WITH_NOISE not yet implemented."
            )

        
        raise NotImplementedError(f"Unknown selection strategy: {self.strategy}")