# -*- coding: utf-8 -*-

"""
Run comparison of different question selection strategies for a single skill configuration (single cfg).
This script allows you to compare the performance of various strategies while keeping other parameters (fold splits, model settings, thresholds) fixed across all strategies.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: scripts/cmp_strategies.py
# Author: Yuta Wakui
# Date: 2026-10-20
# Description: Run comparison of different question selection strategies for a single skill configuration (single cfg).

import os
import glob
import time
import argparse
import yaml
import pandas as pd

from datetime import datetime
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold

from adaptive_self_assessment.components.selector import SelectionStrategy
from adaptive_self_assessment.simulation.adaptive_ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.adaptive_ws2 import run_ws2_simulation

ALL_STRATEGIES: List[str] = [s.value for s in SelectionStrategy]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _calc_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return float(s.mean())


def compare_strategies(
    cfg: Dict[str, Any],
    strategies: Optional[List[str]] = None,
    output_dir: str = "outputs/results/cmp_strategies",
    save_user_logs: bool = True,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare question selection strategies for a single skill configuration (a single cfg),
    swapping only question_selection.strategy. Fold splits, model settings, and thresholds
    stay fixed at the cfg's own values and are shared across every strategy.

    Parameters:
    -----------
        cfg: Dict[str, Any]
            Skill configuration to compare (a single skill, a single mode)
        strategies: Optional[List[str]]
            Strategies to compare (default: all 6 implemented strategies)
        output_dir: str
            Base output directory for the comparison CSV
        save_user_logs: bool
            Whether to save a user-log CSV per strategy
        run_id: Optional[str]
            Timestamp for the output directory (shared across skills when combining results)

    Returns:
    -------
        comparison_df: pd.DataFrame
            One row per strategy
    """
    strategies = list(strategies) if strategies else list(ALL_STRATEGIES)

    # ---- mode ----
    mode = str(cfg.get("mode", "ws1")).lower()
    if mode not in ("ws1", "ws2"):
        raise ValueError(f"Unsupported mode: {mode}")

    # ---- data ----
    data_cfg = cfg.get("data", {})
    common_cfg = data_cfg.get("common", {})
    ws_cfg = data_cfg.get(mode, {})

    input_path: str = ws_cfg.get("input_path", "")
    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input CSV path '{input_path}' does not exist.")

    ignore_items: List[str] = common_cfg.get("ignore_items", [])
    skill_name: str = common_cfg.get("skill_name", "unknown_skill")

    if mode == "ws1":
        ra_col: str = ws_cfg.get("overall_col", "")
    else:
        ra_col: str = ws_cfg.get("current_overall_col", "")
    if not ra_col:
        raise ValueError("overall label column must be specified in config.")

    # ---- thresholds / model (kept fixed across strategies) ----
    thresholds_cfg = cfg.get("thresholds", {})
    RC_THRESHOLD: float = float(thresholds_cfg.get("RC", 0.80))
    RI_THRESHOLD: float = float(thresholds_cfg.get("RI", 0.70))

    model_cfg = cfg.get("model", {})
    item_type: str = str(model_cfg.get("item_model", {}).get("type", "logistic_regression"))
    overall_type: str = str(model_cfg.get("overall_model", {}).get("type", "logistic_regression"))

    # ---- CV ----
    cv_cfg = cfg.get("cv", {})
    K = int(cv_cfg.get("folds", 5))
    stratified = bool(cv_cfg.get("stratified", True))
    random_seed = int(cv_cfg.get("random_seed", 42))

    splitter = (
        StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        if stratified
        else KFold(n_splits=K, shuffle=True, random_state=random_seed)
    )

    # ---- logging ----
    logging_cfg = cfg.get("logging", {})
    save_logs = bool(logging_cfg.get("save_logs", True))
    base_log_dir = logging_cfg.get("log_dir", "outputs/logs")
    timestamped = bool(logging_cfg.get("timestamped", True))

    # ---- load df once ----
    df = pd.read_csv(input_path)
    drop_cols = [c for c in ignore_items if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if ra_col not in df.columns:
        raise ValueError(f"Label column '{ra_col}' not found in {input_path}")

    y = df[ra_col].values if stratified else None

    # fixed splits for every strategy: this is what keeps the comparison fair
    splits = list(splitter.split(df, y))

    # ---- output base ----
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(output_dir, run_id)
    os.makedirs(out_base, exist_ok=True)

    comparison_rows: List[Dict[str, Any]] = []

    sim_fn = run_ws1_simulation if mode == "ws1" else run_ws2_simulation

    for strategy in strategies:
        cfg_g = dict(cfg)
        cfg_g["question_selection"] = dict(cfg.get("question_selection", {}))
        cfg_g["question_selection"]["strategy"] = strategy

        # prevent per-fold log saving inside sim
        cfg_g["logging"] = dict(cfg.get("logging", {}))
        cfg_g["logging"]["save_logs"] = False

        print(f"\n=== {mode.upper()} [{skill_name}] strategy={strategy} ===")

        fold_results: List[Dict[str, Any]] = []
        all_logs: List[pd.DataFrame] = []

        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(splits, start=0):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            sim_result, logs_df = sim_fn(train_df=train_df, test_df=test_df, cfg=cfg_g, fold=fold)

            sim_result = dict(sim_result)
            sim_result["fold"] = fold
            sim_result["mode"] = mode
            sim_result["csv_path"] = input_path
            fold_results.append(sim_result)

            logs_df = logs_df.copy()
            logs_df["fold"] = fold
            logs_df["mode"] = mode
            logs_df["skill_name"] = skill_name
            logs_df["strategy"] = strategy
            all_logs.append(logs_df)
        elapsed_sec = time.time() - start

        df_fold = pd.DataFrame(fold_results)

        # Secondary metric "computation time" at the strategy level (wall time for the full
        # CV run); avg_response_time below is the per-respondent figure already produced by the sim.
        row: Dict[str, Any] = {
            "mode": mode,
            "skill_name": skill_name,
            "csv_path": input_path,
            "cv_seed": random_seed,
            "strategy": strategy,
            "item_model": item_type,
            "overall_model": overall_type,
            "RC_THRESHOLD": RC_THRESHOLD,
            "RI_THRESHOLD": RI_THRESHOLD,
            "num_folds": K,
            "total_questions": _calc_mean(df_fold, "total_questions"),
            "avg_answered_questions": _calc_mean(df_fold, "avg_answered_questions"),
            "avg_complemented_questions": _calc_mean(df_fold, "avg_complemented_questions"),
            "avg_complement_accuracy": _calc_mean(df_fold, "avg_complement_accuracy"),
            "avg_reduction_rate": _calc_mean(df_fold, "reduction_rate"),
            "accuracy_over_threshold": _calc_mean(df_fold, "accuracy_over_threshold"),
            "f1_macro_over_threshold": _calc_mean(df_fold, "f1_macro_over_threshold"),
            "coverage_over_threshold": _calc_mean(df_fold, "coverage_over_threshold"),
            "accuracy_all": _calc_mean(df_fold, "accuracy_all"),
            "f1_macro_all": _calc_mean(df_fold, "f1_macro_all"),
            "avg_response_time": _calc_mean(df_fold, "avg_response_time"),
            "elapsed_sec": float(elapsed_sec),
        }
        comparison_rows.append(row)

        if save_user_logs and save_logs and all_logs:
            logs_all_df = pd.concat(all_logs, ignore_index=True)

            log_subdir = os.path.join(base_log_dir, mode, "user_logs_strategies")
            if timestamped:
                log_subdir = os.path.join(log_subdir, run_id)
            os.makedirs(log_subdir, exist_ok=True)

            log_name = f"{mode}_{skill_name}_user_logs_{strategy}.csv"
            logs_all_df.to_csv(os.path.join(log_subdir, log_name), index=False)
            print(f"Saved user logs: {os.path.join(log_subdir, log_name)}")

    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(out_base, f"{mode}_{skill_name}_strategy_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to: {comp_path}")
    return comparison_df


def compare_strategies_across_skills(
    config_paths: List[str],
    strategies: Optional[List[str]] = None,
    output_dir: str = "outputs/results/cmp_strategies",
    save_user_logs: bool = True,
) -> pd.DataFrame:
    """
    Run compare_strategies for each of several skill configuration files and combine the
    results into a single table. Each skill keeps its fold splits, model settings, and
    thresholds fixed at its own config's values; the only thing shared across skills is
    the set of strategies.

    Parameters:
    -----------
        config_paths: List[str]
            Paths to the skill configuration files to compare
        strategies: Optional[List[str]]
            Strategies to compare (default: all 6 implemented strategies)
        output_dir: str
            Base output directory for the comparison CSV
        save_user_logs: bool
            Whether to save a user-log CSV per skill-strategy pair

    Returns:
    -------
        combined_df: pd.DataFrame
            One row per skill-strategy pair
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_rows: List[pd.DataFrame] = []

    for path in config_paths:
        print(f"\n########## {path} ##########")
        cfg = load_config(path)
        df = compare_strategies(
            cfg=cfg,
            strategies=strategies,
            output_dir=output_dir,
            save_user_logs=save_user_logs,
            run_id=run_id,
        )
        df = df.copy()
        df["config_path"] = path
        combined_rows.append(df)

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

    out_base = os.path.join(output_dir, run_id)
    os.makedirs(out_base, exist_ok=True)
    out_path = os.path.join(out_base, "all_skills_strategy_comparison.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"\nSaved combined comparison to: {out_path}")

    return combined_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config file paths to compare (default: configs/soft_skills/*.yaml)",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to compare (default: all implemented strategies)",
    )
    p.add_argument("--outdir", default="outputs/results/cmp_strategies")
    p.add_argument("--no_user_logs", action="store_true")
    args = p.parse_args()

    config_paths = args.configs or sorted(glob.glob("configs/soft_skills/*.yaml"))
    if not config_paths:
        raise SystemExit("No config files found. Pass --configs explicitly.")

    compare_strategies_across_skills(
        config_paths=config_paths,
        strategies=args.strategies,
        output_dir=args.outdir,
        save_user_logs=(not args.no_user_logs),
    )
