import os
import copy
import argparse
import yaml
import pandas as pd

from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--outdir", default="outputs/results/cmp_thresholds")
    p.add_argument("--no_user_logs", action="store_true")
    return p.parse_args()

def _load_config(path: str) -> Dict[str, Any]:
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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _calc_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    """
    calculate the mean of a specified column in a DataFrame
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

def _to_p_str(x: float) -> str:
    # 0.8 -> 0p8, 0.80 -> 0p80
    s = f"{x:.2f}".rstrip("0").rstrip(".") if abs(x * 100 - round(x * 100)) > 1e-9 else f"{x:.2f}"
    return s.replace(".", "p")


def _metrics_over_threshold_by_ri(logs_df: pd.DataFrame, ri: float) -> Dict[str, Optional[float]]:
    """
    calculate accuracy, f1-score, and coverage over a specified RI threshold
    Parameters:
    -----------
        logs_df: pd.DataFrame
            user logs dataframe containing actual and predicted labels along with confidence scores
        ri: float
            re-aggregation confidence threshold
    Returns:    
    -------
        Dict[str, Optional[float]]
            dictionary containing accuracy ("acc"), f1-score ("f1"), and coverage ("cov") over the specified RI threshold
    """
    if logs_df is None or logs_df.empty:
        return {"acc": None, "f1": None, "cov": None}

    required = {"actual_ra", "predicted_ra", "confidence"}
    missing = required - set(logs_df.columns)
    if missing:
        raise ValueError(f"logs_df missing required columns for RI re-aggregation: {missing}")

    confident_df = logs_df[logs_df["confidence"].astype(float) >= float(ri)]
    if confident_df.empty:
        return {"acc": None, "f1": None, "cov": 0.0}

    y_true = confident_df["actual_ra"].astype(int)
    y_pred = confident_df["predicted_ra"].astype(int)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    f1 = float(f1_score(y_true, y_pred, average="macro") * 100.0)
    cov = float(len(confident_df) / len(logs_df) * 100.0)
    return {"acc": acc, "f1": f1, "cov": cov}


def compare_thresholds(
    cfg: Dict[str, Any],
    output_dir: str = "outputs/results/cmp_thresholds",
    save_user_logs: bool = True,
) -> pd.DataFrame:
    """
    compare different RC and RI threshold combinations by running simulations
    Parameters:
    -----------
        cfg: Dict[str, Any]
            configuration dictionary for the simulations
        output_dir: str
            base output directory to save comparison results
        save_user_logs: bool
            whether to save user logs for each threshold combination
    Returns:
    -------
        comparison_df: pd.DataFrame
            DataFrame summarizing the comparison results across threshold combinations
    """
    # execution mode（WS1/WS2)
    mode = str(cfg.get("mode", "ws1")).lower()
    if mode not in ("ws1", "ws2"):
        raise ValueError(f"Unsupported mode: {mode}")

    # model type
    model_cfg = cfg.get("model", {})
    model_type: str = model_cfg.get("overall", {}).get("type", "logistic_regression")

    # data settings
    data_cfg = cfg.get("data", {})
    common_cfg = data_cfg.get("common", {})
    ws_cfg = data_cfg.get(mode, {})

    # input data settings
    input_path: str = ws_cfg.get("input_path", None)
    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")
    
    skill_name = common_cfg.get("skill_name", "unknown_skill")
    ignore_items: List[str] = common_cfg.get("ignore_items", [])

    # label column name
    if mode == "ws1":
        ra_col = ws_cfg.get("overall_col", "")
    else:
        ra_col = ws_cfg.get("current_overall_col", "")
    if not ra_col:
        raise ValueError("overall label column must be specified in config.")

    # CV settings
    cv_cfg = cfg.get("cv", {})
    K: int = int(cv_cfg.get("folds", 5))
    stratified: bool = bool(cv_cfg.get("stratified", True))
    random_seed: int = int(cv_cfg.get("random_seed", 42))

    splitter = (
        StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        if stratified
        else KFold(n_splits=K, shuffle=True, random_state=random_seed)
    )

    # grid settings
    thresholds_cfg = cfg.get("thresholds", {})
    rc_values = thresholds_cfg.get("rc_grid", [thresholds_cfg.get("RC", 0.80)])
    ri_values = thresholds_cfg.get("ri_grid", [thresholds_cfg.get("RI", 0.70)])
    rc_values = [float(x) for x in rc_values]
    ri_values = [float(x) for x in ri_values]

    # results saving settings
    logging_cfg = cfg.get("logging", {})
    save_logs = bool(logging_cfg.get("save_logs", True))
    base_log_dir = logging_cfg.get("log_dir", "outputs/logs")
    timestamped = bool(logging_cfg.get("timestamped", True))

    # load df once
    df = pd.read_csv(input_path)
    drop_cols = [c for c in ignore_items if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if ra_col not in df.columns:
        raise ValueError(f"Label column '{ra_col}' not found in {input_path}")

    y = df[ra_col].values if stratified else None

    # fixed splits for all thresholds
    splits = list(splitter.split(df, y))

    # output base
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(output_dir, mode, run_id)
    os.makedirs(out_base, exist_ok=True)

    comparison_rows: List[Dict[str, Any]] = []

    # choose simulation function
    sim_fn = run_ws1_simulation if mode == "ws1" else run_ws2_simulation

    for RC in rc_values:
        # overwrite RC in cfg
        cfg_g = dict(cfg)
        cfg_g["thresholds"] = dict(cfg.get("thresholds", {}))
        cfg_g["thresholds"]["RC"] = float(RC)

        # prevent per-fold log saving inside sim
        cfg_g["logging"] = dict(cfg.get("logging", {}))
        cfg_g["logging"]["save_logs"] = False

        fold_results: List[Dict[str, Any]] = []
        all_logs: List[pd.DataFrame] = []

        print(f"\n=== {mode.upper()} Grid: RC={RC} ===")

        # CV execution
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
            logs_df["RC_THRESHOLD"] = float(RC)
            all_logs.append(logs_df)

        df_fold = pd.DataFrame(fold_results)
        logs_all_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

        # Base metrics (independent of RI)
        row = {
            "mode": mode,
            "skill_name": skill_name,
            "model(overall)": model_type,
            "csv_path": input_path,
            "cv_seed": random_seed,
            "RC_THRESHOLD": float(RC),
            "num_folds": K,
            "total_questions": _calc_mean(df_fold, "total_questions"),
            "avg_answered_questions": _calc_mean(df_fold, "avg_answered_questions"),
            "avg_complemented_questions": _calc_mean(df_fold, "avg_complemented_questions"),
            "avg_complement_accuracy": _calc_mean(df_fold, "avg_complement_accuracy"),
            "avg_reduction_rate": _calc_mean(df_fold, "reduction_rate"),
            "accuracy_all": _calc_mean(df_fold, "accuracy_all"),
            "f1_macro_all": _calc_mean(df_fold, "f1_macro_all"),
            "avg_response_time": _calc_mean(df_fold, "avg_response_time"),
        }

        # vary RI and compute metrics
        for RI in ri_values:
            ri_key = _to_p_str(float(RI))  # e.g. 0.8 -> 0p80
            m = _metrics_over_threshold_by_ri(logs_all_df, float(RI))
            row[f"acc_ri{ri_key}"] = m["acc"]
            row[f"f1_ri{ri_key}"] = m["f1"]
            row[f"cov_ri{ri_key}"] = m["cov"]

        comparison_rows.append(row)

        if save_user_logs and save_logs and all_logs:
            logs_all_df = pd.concat(all_logs, ignore_index=True)
            rc_str = str(RC).replace(".", "p")
            ri_str = str(RI).replace(".", "p")

            log_subdir = os.path.join(base_log_dir, mode, "user_logs_thresholds")
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
    args = _parse_args()
    cfg = _load_config(args.config)
    compare_thresholds(
        cfg=cfg,
        output_dir=args.outdir,
        save_user_logs=(not args.no_user_logs),
    )