# -*- coding: utf-8 -*-

"""
Run adaptive self-assessment simulations.
This script executes simulations for WS1 and WS2 based on the provided configuration file.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: scripts/run_sim.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Run adaptive self-assessment simulations based on configuration.

import os
import pandas as pd
import argparse
import yaml

from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from sklearn.model_selection import KFold, StratifiedKFold

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation
from adaptive_self_assessment.simulation.common import load_app_config

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to the config file")
    return p.parse_args()

def _load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    load configuration from a YAML file.
    
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

def _calc_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    """
    calculate the mean of a specified column in a DataFrame.
    Parameters:
    -----------
    df: pd.DataFrame
        dataframe to calculate the mean from    
    col: str
        column name to calculate the mean
    Returns:
    -------
    Optional[float]
        mean value of the specified column, or None if the column does not exist or is empty
    """
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return float(s.mean())

def run_simulations(config_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    run adaptive self-assessment simulations based on the provided configuration file.
    Parameters:
    -----------
        config_path: str
            path to the configuration file
    Returns:
    -------
        results_df: pd.DataFrame
            summary of simulation results
        logs_all_df: pd.DataFrame
            detailed logs for all users (if saved), otherwise None
    """

    # load config
    cfg = _load_config(config_path)
    app = load_app_config(cfg)

    # execution mode（WS1/WS2)
    mode: str = str(cfg.get("mode", "ws1")).lower()
    if mode not in ("ws1", "ws2"):
        raise ValueError(f"Unsupported mode: {mode} (expected 'ws1' or 'ws2')")

    # choose ws config
    if mode == "ws1":
        ws = app.ws1_data
        ra_col = ws.overall_col
    else:
        ws = app.ws2_data
        ra_col = ws.current_overall_col

    # model type
    overall_model_type = app.overall_model.type

    # thresholds
    RC_THRESHOLD: float = float(app.thresholds.rc)
    RI_THRESHOLD: float = float(app.thresholds.ri)
    rc_str = str(RC_THRESHOLD).replace(".", "p")
    ri_str = str(RI_THRESHOLD).replace(".", "p")    

    # data settings
    input_path: str = ws.input_path
    skill_name: str = app.common_data.skill_name
    ignore_items: List[str] = app.common_data.ignore_items

    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    # CV settings
    K: int = int(app.cv.folds)
    stratified: bool = bool(app.cv.stratified)
    random_seed: int = int(app.cv.random_seed)

    splitter = (
        StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        if stratified
        else KFold(n_splits=K, shuffle=True, random_state=random_seed)
    )

    # execution ID for timestamped outputs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # results saving settings
    results_cfg = cfg.get("results", {}) or {}
    save_csv: bool = bool(results_cfg.get("save_csv", True))
    output_dir: str = str(results_cfg.get("output_dir", "outputs/results"))
    out_timestamped: bool = bool(results_cfg.get("timestamped", True))
    filename_suffix: str = str(results_cfg.get("filename_suffix", "")).strip()
    save_fold_results: bool = bool(results_cfg.get("save_fold_results", False))
    suffix = f"_{filename_suffix}" if filename_suffix else ""

    # execution log saving settings
    logging_cfg = cfg.get("logging", {}) or {}
    save_logs = bool(logging_cfg.get("save_logs", True))
    base_log_dir = str(logging_cfg.get("log_dir", "outputs/logs"))
    timestamped = bool(logging_cfg.get("timestamped", True))

    print(f"=== {mode.upper()} Simulation Started ===")
    print(f"input: {input_path}")
    print(f"skill: {skill_name}")
    print(f"model(overall): {overall_model_type}")
    print(f"thresholds: RC={RC_THRESHOLD}, RI={RI_THRESHOLD}")
    print(f"cv: folds={K}, stratified={stratified}, seed={random_seed}")

    # load data
    df = pd.read_csv(input_path)

    # drop ignore_items columns
    drop_cols = [col for col in ignore_items if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped ignore_items columns: {drop_cols}") 

    print(f"\n==== {mode.upper()}: {skill_name} ====")
    print(f"csv: {input_path}")
    print(f"n_rows: {len(df)}, n_cols: {len(df.columns)}")

    if ra_col not in df.columns:
        raise ValueError(f"Label column '{ra_col}' not found in {input_path}.")

    # for stratifiedKFold
    y = df[ra_col].astype(int).values if stratified else None

    fold_results: List[Dict[str, Any]] = []
    all_fold_results: List[Dict[str, Any]] = []
    all_logs: List[pd.DataFrame] = []

    # CV execution
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y), start=0):
        print(f"---- Fold {fold+1}/{K} ----")
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        if mode == "ws1":
            sim_result, logs_df = run_ws1_simulation(train_df=train_df, test_df=test_df, cfg=cfg, fold=fold)
        else:
            sim_result, logs_df = run_ws2_simulation(train_df=train_df, test_df=test_df, cfg=cfg, fold=fold)

        # store fold results
        sim_result = dict(sim_result)
        sim_result["fold"] = fold
        sim_result["mode"] = mode
        sim_result["csv_path"] = input_path
        fold_results.append(sim_result)
        all_logs.append(logs_df)

    df_fold = pd.DataFrame(fold_results)

    if save_fold_results:
        all_fold_results.extend(fold_results)

    # summarize results
    row = {
        "mode": mode,
        "skill_name": skill_name,
        "model": overall_model_type,
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
    }

    results_df = pd.DataFrame([row])

    # output directory
    subdir = os.path.join(output_dir, mode, "sim_results")
    if out_timestamped:
        subdir = os.path.join(subdir, run_id)

    # save results CSV
    if save_csv:
        os.makedirs(subdir, exist_ok=True)

        # save summary results
        out_name = f"{mode}_results_rc{rc_str}_ri{ri_str}{suffix}.csv"
        out_path = os.path.join(subdir, out_name)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")
    
    # save fold results CSV
    if save_fold_results and all_fold_results:
        os.makedirs(subdir, exist_ok=True)
        
        fold_df = pd.DataFrame(all_fold_results)
        fold_out_name = f"{mode}_fold_results_rc{rc_str}_ri{ri_str}{suffix}.csv"
        fold_out_path = os.path.join(subdir, fold_out_name)
        fold_df.to_csv(fold_out_path, index=False)
        print(f"Saved fold results to: {fold_out_path}")
    
    # save user logs
    logs_all_df: Optional[pd.DataFrame] = None
    if save_logs and all_logs:
        logs_all_df = pd.concat(all_logs, ignore_index=True)

        log_subdir = os.path.join(base_log_dir, mode)
        if timestamped:
            log_subdir = os.path.join(log_subdir, run_id)

        os.makedirs(log_subdir, exist_ok=True)

        log_out_name = f"{mode}_user_logs_rc{rc_str}_ri{ri_str}{suffix}.csv"
        log_out_path = os.path.join(log_subdir, log_out_name)
        logs_all_df.to_csv(log_out_path, index=False)
        print(f"Saved user logs to: {log_out_path}")

    print(f"=== {mode.upper()} Simulation Completed ===")
    return results_df, logs_all_df


if __name__ == "__main__":
    args = _parse_args()
    run_simulations(config_path=args.config)