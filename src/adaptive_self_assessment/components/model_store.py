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
    A simple in-memory cache for trained scikit-learn Pipeline models
    """
    models: Dict[CacheKey, Pipeline] = field(default_factory=dict)

    def get(self, key: CacheKey) -> Pipeline | None:
        model = self.models.get(key)

        if logger.isEnabledFor(logging.DEBUG):
            if model is None:
                logger.debug(f"[CACHE][MISS] key={key} (size={len(self.models)})")
            else:
                logger.debug(f"[CACHE][HIT] key={key} (size={len(self.models)})")

        return model

    def set(self, key: CacheKey, model: Pipeline) -> None:
        self.models[key] = model
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[CACHE][STORE] key={key} (size={len(self.models)})")