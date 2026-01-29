# -*- coding: utf-8 -*-

"""
Model store for caching trained scikit-learn Pipeline models.
This module defines the ModelStore class, which provides a simple in-memory cache
for storing and retrieving trained models based on unique cache keys.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/components/model_store.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Model store for caching trained models

from dataclasses import dataclass, field
from typing import Dict, Hashable, Tuple
import logging

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Type alias for cache keys used in the model cache dictionary
CacheKey = Tuple[Hashable, ...]

@dataclass
class ModelStore:
    """
    In-memory cache for trained scikit-learn Pipeline models.

    This class is used to avoid retraining identical models during
    simulation by caching models with hashable cache keys.
    """
    models: Dict[CacheKey, Pipeline] = field(default_factory=dict)

    def get(self, key: CacheKey) -> Pipeline | None:
        """
        Retrieve a model from the cache.

        Parameters
        ----------
        key : CacheKey
            Cache key representing model configuration.

        Returns
        -------
        Pipeline or None
            Cached model if present, otherwise None.
        """
        model = self.models.get(key)

        if logger.isEnabledFor(logging.DEBUG):
            if model is None:
                logger.debug("[ModelStore][MISS] key=%s size=%d", key, len(self.models))
            else:
                logger.debug("[ModelStore][HIT] key=%s size=%d", key, len(self.models))

        return model

    def set(self, key: CacheKey, model: Pipeline) -> None:
        """
        Store a trained model in the cache.

        Parameters
        ----------
        key : CacheKey
            Cache key representing model configuration.
        model : Pipeline
            Trained scikit-learn Pipeline model.
        """
        self.models[key] = model

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[ModelStore][STORE] key=%s size=%d", key, len(self.models))