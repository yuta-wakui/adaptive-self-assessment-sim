# -*- coding: utf-8 -*-
"""
Integration tests for adaptive self-assessment simulations.
This module tests the end-to-end functionality of the adaptive self-assessment simulations for both WS1
and WS2.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: tests/integration/test_sim.py
# Author: Yuta Wakui
# Date: 2026-02-11
# Description: Integration tests for adaptive self-assessment simulations (WS1 and WS2)

import os
import yaml
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from adaptive_self_assessment.components.selector import SelectionStrategy
from adaptive_self_assessment.simulation.adaptive_ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.adaptive_ws2 import run_ws2_simulation


def _load_config(config_path = "configs/config.yaml"):
    """
    load configuration from a YAML file
    
    Parameters:
    ----------
    config_path: str
        path to the configuration file (default: "configs/config.yaml")
    
    Returns:
    -------
    cfg: Dict[str, Any]
        configuration dictionary loaded from the file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

@pytest.mark.parametrize(
    ("ws1_path", "cfg_path"),
    [
        ("data/sample/ws1/ws1_data_sample.csv", "configs/config.yaml"),
    ]
)
def test_run_ws1_simulation(ws1_path, cfg_path):
    print("=== WS1 Simulation Test ===")

    # load data
    if not os.path.exists(ws1_path):
        pytest.skip(f"File not found: {ws1_path}")
    df_ws1 = pd.read_csv(ws1_path)

    # load config
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config file not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    # split into train and test sets (80% train, 20% test)
    df_ws1_train, df_ws1_test = train_test_split(df_ws1, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws1_train)}, Test data size: {len(df_ws1_test)}")

    # run simulation
    sim_results, logs_df = run_ws1_simulation(
        train_df=df_ws1_train,
        test_df=df_ws1_test,
        cfg=cfg,
    )

    # validate results
    assert isinstance(sim_results, dict)
    assert isinstance(logs_df, pd.DataFrame)

    rr = sim_results["reduction_rate"]
    assert rr is None or (0.0 <= float(rr) <= 100.0)


    expected_keys = {
        "skill","model_type","RC_THRESHOLD","RI_THRESHOLD","num_train","num_test",
        "selector_strategy",
        "accuracy_over_threshold","accuracy_all",
        "f1_macro_over_threshold","f1_macro_all",
        "coverage_over_threshold","total_questions",
        "avg_answered_questions","avg_complemented_questions",
        "avg_complement_accuracy", 
        "reduction_rate",
        "avg_response_time","max_response_time","min_response_time",
    }

    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}"

    assert sim_results["selector_strategy"] in {s.value for s in SelectionStrategy}

    print("WS1 simulation Results: ", sim_results)

@pytest.mark.parametrize(
    ("ws2_path", "cfg_path"),
    [
        ("data/sample/ws2/ws2_data_sample.csv", "configs/config.yaml"),
    ]
)
def test_run_ws2_simulation(ws2_path, cfg_path):
    print("=== WS2 Simulation Test ===")

    # load data
    if not os.path.exists(ws2_path):
        pytest.skip(f"File not found: {ws2_path}")
    df_ws2 = pd.read_csv(ws2_path)

    # load config
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    # split into train and test sets (80% train, 20% test)
    df_ws2_train, df_ws2_test = train_test_split(df_ws2, test_size=0.2, random_state=42,shuffle=True)
    print(f"Training data size: {len(df_ws2_train)}, Test data size: {len(df_ws2_test)}")
    
    # run simulation
    sim_results, logs_df = run_ws2_simulation(
        train_df=df_ws2_train,
        test_df=df_ws2_test,
        cfg=cfg,
    )

    # validate results
    assert isinstance(sim_results, dict)
    assert isinstance(logs_df, pd.DataFrame)

    rr = sim_results["reduction_rate"]
    assert rr is None or (0.0 <= float(rr) <= 100.0)


    expected_keys = {
        "skill","model_type","RC_THRESHOLD","RI_THRESHOLD","num_train","num_test",
        "selector_strategy",
        "accuracy_over_threshold","accuracy_all",
        "f1_macro_over_threshold","f1_macro_all",
        "coverage_over_threshold","total_questions",
        "avg_answered_questions","avg_complemented_questions",
        "avg_complement_accuracy", 
        "reduction_rate",
        "avg_response_time","max_response_time","min_response_time",
    }

    missing = expected_keys - set(sim_results.keys())
    assert not missing, f"Missing keys in simulation results: {missing}" 

    assert sim_results["selector_strategy"] in {s.value for s in SelectionStrategy}
    
    print("WS2 simulation Results: ", sim_results)

# ---------------------------------------------------------------------------
# Every selection strategy, end to end
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [s.value for s in SelectionStrategy]


def _with_strategy(cfg, strategy):
    """Copy of cfg with only the question selection strategy replaced."""
    out = dict(cfg)
    out["question_selection"] = dict(cfg.get("question_selection", {}) or {})
    out["question_selection"]["strategy"] = strategy
    return out


def _load_sample(mode, test_size=0.1):
    """Load the sample dataset and config for `mode`, skipping if either is missing.

    The test split is kept small on purpose: these tests sweep every strategy over both
    datasets, and each respondent costs one model fit per unanswered item per step.
    """
    cfg_path = "configs/config.yaml"
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config file not found: {cfg_path}")
    cfg = _load_config(cfg_path)

    data_path = f"data/sample/{mode}/{mode}_data_sample.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"File not found: {data_path}")
    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    return cfg, train_df, test_df


def _run(mode, cfg, train_df, test_df, fold=0):
    sim_fn = run_ws1_simulation if mode == "ws1" else run_ws2_simulation
    return sim_fn(train_df=train_df, test_df=test_df, cfg=cfg, fold=fold)


def _item_cols(cfg, mode):
    ws = cfg["data"][mode]
    return ws["item_cols"] if mode == "ws1" else ws["current_item_cols"]


@pytest.mark.parametrize("mode", ["ws1", "ws2"])
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_all_strategies_run_under_identical_settings(mode, strategy):
    """Smoke test: every strategy completes on WS1 and WS2 with the same fold and config."""
    cfg, train_df, test_df = _load_sample(mode)
    item_cols = _item_cols(cfg, mode)

    sim_results, logs_df = _run(mode, _with_strategy(cfg, strategy), train_df, test_df)

    assert sim_results["selector_strategy"] == strategy
    assert len(logs_df) == len(test_df)

    total = len(item_cols)
    assert sim_results["total_questions"] == total

    for _, row in logs_df.iterrows():
        answered = list(row["answered_items"])
        complemented = [item for (item, *_rest) in row["complemented_items"]]

        # no item is ever asked twice, and every asked item is a configured item
        assert len(answered) == len(set(answered))
        assert set(answered) <= set(item_cols)

        # answered and complemented items partition the item set exactly
        assert set(answered).isdisjoint(complemented)
        assert set(answered) | set(complemented) == set(item_cols)

        assert row["num_answered_questions"] == len(answered)
        assert 0 <= row["num_answered_questions"] <= total

    # the reduction rate counts asked items only, so it stays a proper ratio
    rr = sim_results["reduction_rate"]
    assert rr is not None and 0.0 <= float(rr) <= 1.0


@pytest.mark.parametrize("mode", ["ws1", "ws2"])
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_all_strategies_are_reproducible(mode, strategy):
    """The same seed and the same input must reproduce the same run."""
    cfg, train_df, test_df = _load_sample(mode, test_size=0.05)
    run_cfg = _with_strategy(cfg, strategy)

    first_results, first_logs = _run(mode, run_cfg, train_df, test_df)
    second_results, second_logs = _run(mode, run_cfg, train_df, test_df)

    assert first_results["reduction_rate"] == second_results["reduction_rate"]
    assert first_results["accuracy_all"] == second_results["accuracy_all"]

    for col in ("answered_items", "num_answered_questions", "predicted_ra"):
        assert list(first_logs[col]) == list(second_logs[col])


@pytest.mark.parametrize("mode", ["ws1", "ws2"])
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_test_rubric_labels_do_not_leak_into_the_run(mode, strategy):
    """Rewriting the test respondents' rubric labels must not change the simulation.

    Question selection scores mutual information on the training data, and both models are
    fitted on the training data, so the ground-truth label of a test respondent may only be
    read when the final prediction is scored.
    """
    cfg, train_df, test_df = _load_sample(mode, test_size=0.05)
    run_cfg = _with_strategy(cfg, strategy)
    ra_col = cfg["data"][mode]["overall_col" if mode == "ws1" else "current_overall_col"]

    baseline_results, baseline_logs = _run(mode, run_cfg, train_df, test_df)

    # shift every test label to a different valid class, leaving the training data alone
    corrupted_test = test_df.copy()
    classes = sorted(train_df[ra_col].unique())
    shifted = {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
    corrupted_test[ra_col] = corrupted_test[ra_col].map(shifted).fillna(corrupted_test[ra_col])

    _, corrupted_logs = _run(mode, run_cfg, train_df, corrupted_test)

    assert list(corrupted_logs["answered_items"]) == list(baseline_logs["answered_items"])
    assert list(corrupted_logs["predicted_ra"]) == list(baseline_logs["predicted_ra"])
    assert list(corrupted_logs["num_answered_questions"]) == list(baseline_logs["num_answered_questions"])

    # the labels really did change, so the check above is not vacuous
    assert list(corrupted_logs["actual_ra"]) != list(baseline_logs["actual_ra"])
