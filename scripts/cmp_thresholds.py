import os
import pandas as pd
import argparse
import yaml

from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation


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


def compare_thresholds(
    cfg: Dict[str, Any],
    output_dir: str = "outputs/results/cmp_thresholds",
    save_user_logs: bool = True,
) -> pd.DataFrame:
    # ---- mode ----
    mode = str(cfg.get("mode", "ws1")).lower()
    if mode not in ("ws1", "ws2"):
        raise ValueError(f"Unsupported mode: {mode}")

    # ---- data ----
    data_cfg = cfg.get("data", {})
    common_cfg = data_cfg.get("common", {})
    ws_cfg = data_cfg.get(mode, {})

    input_path: str = common_cfg.get("input_path", "")
    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input CSV path '{input_path}' does not exist.")

    ignore_items: List[str] = common_cfg.get("ignore_items", [])
    skill_name: str = common_cfg.get("skill_name", "unknown_skill")

    ra_col: str = ws_cfg.get("ra_col", "")
    if not ra_col:
        raise ValueError(f"data.{mode}.ra_col must be specified in config.")

    # ---- model info ----
    model_cfg = cfg.get("model", {})
    overall_model_type: str = model_cfg.get("overall", {}).get("type", "logistic_regression")

    # ---- grid ----
    thresholds_cfg = cfg.get("thresholds", {})
    rc_values = thresholds_cfg.get("rc_grid", [thresholds_cfg.get("RC", 0.80)])
    ri_values = thresholds_cfg.get("ri_grid", [thresholds_cfg.get("RI", 0.70)])
    rc_values = [float(x) for x in rc_values]
    ri_values = [float(x) for x in ri_values]

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

    # fixed splits for all thresholds
    splits = list(splitter.split(df, y))

    # ---- output base ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(output_dir, mode, run_id)
    os.makedirs(out_base, exist_ok=True)

    comparison_rows: List[Dict[str, Any]] = []

    # choose simulation function
    sim_fn = run_ws1_simulation if mode == "ws1" else run_ws2_simulation

    for RC, RI in product(rc_values, ri_values):
        # overwrite thresholds only
        cfg_g = dict(cfg)
        cfg_g["thresholds"] = dict(cfg.get("thresholds", {}))
        cfg_g["thresholds"]["RC"] = float(RC)
        cfg_g["thresholds"]["RI"] = float(RI)

        # prevent per-fold log saving inside sim
        cfg_g["logging"] = dict(cfg.get("logging", {}))
        cfg_g["logging"]["save_logs"] = False

        fold_results: List[Dict[str, Any]] = []
        all_logs: List[pd.DataFrame] = []

        print(f"\n=== {mode.upper()} Grid: RC={RC}, RI={RI} ===")

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
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
            logs_df["RC_THRESHOLD"] = float(RC)
            logs_df["RI_THRESHOLD"] = float(RI)
            all_logs.append(logs_df)

        df_fold = pd.DataFrame(fold_results)

        row = {
            "mode": mode,
            "skill_name": skill_name,
            "model(overall)": overall_model_type,
            "csv_path": input_path,
            "cv_seed": random_seed,
            "RC_THRESHOLD": float(RC),
            "RI_THRESHOLD": float(RI),
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
        comparison_rows.append(row)

        if save_user_logs and save_logs and all_logs:
            logs_all_df = pd.concat(all_logs, ignore_index=True)
            rc_str = str(RC).replace(".", "p")
            ri_str = str(RI).replace(".", "p")

            log_subdir = os.path.join(base_log_dir, mode, "user_logs")
            if timestamped:
                log_subdir = os.path.join(log_subdir, run_id)
            os.makedirs(log_subdir, exist_ok=True)

            log_name = f"{mode}_user_logs_rc{rc_str}_ri{ri_str}.csv"
            logs_all_df.to_csv(os.path.join(log_subdir, log_name), index=False)
            print(f"Saved user logs: {os.path.join(log_subdir, log_name)}")

    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(out_base, f"{mode}_threshold_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to: {comp_path}")
    return comparison_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--outdir", default="outputs/results/cmp_thresholds")
    p.add_argument("--no_user_logs", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    compare_thresholds(
        cfg=cfg,
        output_dir=args.outdir,
        save_user_logs=(not args.no_user_logs),
    )
