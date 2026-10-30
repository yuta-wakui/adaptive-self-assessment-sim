# -*- coding: utf-8 -*-
"""
Integration tests for question selection strategies in adaptive simulations.
This module runs every SelectionStrategy end-to-end through both the WS1 and WS2
simulations, so that a strategy cannot be added to the selector without also being
wired into the simulation loop.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/integration/test_strategies.py
# Author: Yuta Wakui
# Date: 2026-10-30
# Description: Integration tests for every question selection strategy (WS1 and WS2)

import copy
import os
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

from adaptive_self_assessment.components.selector import SelectionStrategy
from adaptive_self_assessment.simulation.adaptive_ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.adaptive_ws2 import run_ws2_simulation

CFG_PATH = "configs/config.yaml"

# mode -> (simulation function, item column key, overall column key)
MODES: Dict[str, Tuple[Callable, str, str]] = {
    "ws1": (run_ws1_simulation, "item_cols", "overall_col"),
    "ws2": (run_ws2_simulation, "current_item_cols", "current_overall_col"),
}

ALL_STRATEGIES: List[SelectionStrategy] = list(SelectionStrategy)


# ----------------------------
# Helpers / fixtures
# ----------------------------

def _load_config(config_path: str = CFG_PATH) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def base_cfg() -> Dict[str, Any]:
    if not os.path.exists(CFG_PATH):
        pytest.skip(f"Config file not found: {CFG_PATH}")
    return _load_config(CFG_PATH)


@pytest.fixture(scope="module")
def splits(base_cfg) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Fixed train/test split per mode, shared by every strategy."""
    out = {}
    for mode in MODES:
        path = base_cfg["data"][mode]["input_path"]
        if not os.path.exists(path):
            pytest.skip(f"File not found: {path}")
        df = pd.read_csv(path)
        ignore = base_cfg["data"]["common"].get("ignore_items") or []
        df = df.drop(columns=[c for c in ignore if c in df.columns])
        out[mode] = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    return out


_cache: Dict[Tuple[str, str], Tuple[Dict[str, Any], pd.DataFrame]] = {}


def _run(mode: str, strategy: SelectionStrategy, base_cfg, splits):
    """Run one (mode, strategy) simulation; results are cached across tests."""
    key = (mode, strategy.value)
    if key not in _cache:
        cfg = copy.deepcopy(base_cfg)
        cfg["question_selection"]["strategy"] = strategy.value
        train_df, test_df = splits[mode]
        sim_fn = MODES[mode][0]
        _cache[key] = sim_fn(train_df=train_df, test_df=test_df, cfg=cfg, fold=0)
    return _cache[key]


def _item_cols(mode: str, cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["data"][mode][MODES[mode][1]])


# ----------------------------
# Tests
# ----------------------------

@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.value)
def test_every_strategy_runs_end_to_end(mode, strategy, base_cfg, splits):
    """Every strategy must run through the simulation without raising."""
    sim_results, logs_df = _run(mode, strategy, base_cfg, splits)

    _, test_df = splits[mode]
    assert sim_results["selector_strategy"] == strategy.value
    assert len(logs_df) == len(test_df)
    assert (logs_df["selector_strategy"] == strategy.value).all()

    rr = sim_results["reduction_rate"]
    assert rr is not None and 0.0 <= rr <= 1.0


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.value)
def test_items_are_partitioned(mode, strategy, base_cfg, splits):
    """Answered and complemented items must be disjoint and together cover all items."""
    _, logs_df = _run(mode, strategy, base_cfg, splits)
    items = set(_item_cols(mode, base_cfg))

    for _, row in logs_df.iterrows():
        answered = list(row["answered_items"])
        complemented = [c[0] for c in row["complemented_items"]]

        assert len(answered) >= 1, "at least one item must be asked"
        assert len(answered) == len(set(answered)), "an item was asked twice"
        assert len(complemented) == len(set(complemented)), "an item was complemented twice"
        assert set(answered).isdisjoint(complemented)
        assert set(answered) | set(complemented) == items
        assert row["num_answered_questions"] + row["num_complemented_questions"] == len(items)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.value)
def test_strategy_is_reproducible(mode, strategy, base_cfg, splits):
    """Running the same strategy twice must ask the same items in the same order."""
    _, logs_first = _run(mode, strategy, base_cfg, splits)

    cfg = copy.deepcopy(base_cfg)
    cfg["question_selection"]["strategy"] = strategy.value
    train_df, test_df = splits[mode]
    _, logs_second = MODES[mode][0](train_df=train_df, test_df=test_df, cfg=cfg, fold=0)

    assert logs_first["answered_items"].tolist() == logs_second["answered_items"].tolist()
    assert logs_first["predicted_ra"].tolist() == logs_second["predicted_ra"].tolist()


@pytest.mark.parametrize("mode", list(MODES))
def test_unanswered_relevance_starts_like_all_relevance(mode, base_cfg, splits):
    """
    With A empty and U equal to the full item set, Inter-Item Relevance (Unanswered)
    reduces to Inter-Item Relevance (All), so both must ask the same first item.
    """
    _, logs_all = _run(mode, SelectionStrategy.INTER_ITEM_RELEVANCE_ALL, base_cfg, splits)
    _, logs_unans = _run(mode, SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED, base_cfg, splits)

    first_all = [items[0] for items in logs_all["answered_items"]]
    first_unans = [items[0] for items in logs_unans["answered_items"]]
    assert first_all == first_unans


@pytest.mark.parametrize("mode", list(MODES))
def test_mrmr_redundancy_term_is_active(mode, base_cfg, splits):
    """
    Inter-Item mRMR (Unanswered) must receive the answered set A from the simulation.
    If A were not passed, the redundancy term would vanish and mRMR would behave exactly
    like Inter-Item Relevance (Unanswered).

    Note: this relies on the sample data producing at least one differing question order.
    """
    _, logs_rel = _run(mode, SelectionStrategy.INTER_ITEM_RELEVANCE_UNANSWERED, base_cfg, splits)
    _, logs_mrmr = _run(mode, SelectionStrategy.INTER_ITEM_MRMR_UNANSWERED, base_cfg, splits)

    assert logs_rel["answered_items"].tolist() != logs_mrmr["answered_items"].tolist()


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    "strategy",
    [s for s in ALL_STRATEGIES if s != SelectionStrategy.RANDOM],
    ids=lambda s: s.value,
)
def test_deterministic_strategies_share_first_item(mode, strategy, base_cfg, splits):
    """
    Non-random strategies score items from the training data only (never from the
    respondent's answers), so every user in a fold must be asked the same first item.
    """
    _, logs_df = _run(mode, strategy, base_cfg, splits)
    first_items = {items[0] for items in logs_df["answered_items"]}
    assert len(first_items) == 1