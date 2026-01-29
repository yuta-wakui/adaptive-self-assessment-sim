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
from typing import List

class SelectionStrategy(str, Enum):
    """
    Question selection strategies
    """
    RANDOM = "random"

# Module-level RNG for question selection
_rng = np.random.default_rng(42)

def set_selector_seed(seed: int) -> None:
    """
    Set the random seed for the initial selector RNG
    
    Parameters:
    ----------
    seed: int
        Seed value for reproducibility
    """
    global _rng
    _rng = np.random.default_rng(seed)

def select_question(C: List[str], strategy: SelectionStrategy = SelectionStrategy.RANDOM) -> str:
    """
    Select a question item from the remaining list C based on the specified strategy.
    
    Parameters:
    ----------
    C: list of str
        Remaining question items to select from
    strategy: SelectionStrategy
        Strategy for selecting the question item (default: RANDOM)
    Returns:
        str: Selected question item
    Raises:
    -------
    ValueError
        If 'C" is empty
    NotImplementedError
        If the strategy is not implemented
    """
    if len(C) == 0: 
        raise ValueError("The list of question items is empty.")

    if strategy == SelectionStrategy.RANDOM:
        return str(_rng.choice(C))
    
    raise NotImplementedError(f"Unknown selection strategy: {strategy}")