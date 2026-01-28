import os
import pandas as pd
import argparse
import yaml
import copy

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

def _expand_grid(grid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    パラメータグリッドを展開して全組み合わせのリストを返す
    e.g., {"C": [0.1, 1.0], "solver": ["lbfgs", "liblinear"]}  
    -> [{"C": 0.1, "solver": "lbfgs"}, {"C": 0.1, "solver": "liblinear"},
        {"C": 1.0, "solver": "lbfgs"}, {"C": 1.0, "solver": "liblinear"}]
    """
    if not grid:
        return [{}]
    
    keys = list(grid.keys())
    values_list= []
    for k in keys:
        v = grid[k]
        values_list.append(v if isinstance(v, list) else [v])
    
    combos: List[Dict[str, Any]] = []
    for prod in product(*values_list):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos

def _merge_params(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    baseのパラメータにextraのパラメータを上書きでマージして返す
    """
    out = dict(base or {})
    for k, v in extra.items():
        out[k] = v
    return out

def compare_params(
    cfg: Dict[str, Any],
    output_dir: str = "outputs/results/cmp_params",
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

    input_path: str = ws_cfg.get("input_path", "")
    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input CSV path '{input_path}' does not exist.")

    ignore_items: List[str] = common_cfg.get("ignore_items", [])
    skill_name: str = common_cfg.get("skill_name", "unknown_skill")

    ra_col: str = ws_cfg.get("ra_col", "")
    if not ra_col:
        raise ValueError(f"data.{mode}.ra_col must be specified in config.")
    
    # ---- thresholds ----
    thresholds_cfg = cfg.get("thresholds", {})
    RC_THRESHOLD: float = float(thresholds_cfg.get("RC", 0.80))
    RI_THRESHOLD: float = float(thresholds_cfg.get("RI", 0.70))

    # ---- mode ----
    model_cfg = cfg.get("model", {})
    item_cfg = model_cfg.get("item", {})
    overall_cfg = model_cfg.get("overall", {})

    item_type: str = str(item_cfg.get("type", "logistic_regression"))
    overall_type: str = str(overall_cfg.get("type", "logistic_regression"))

    item_params_base = item_cfg.get("params", {}) or {}
    overall_params_base = overall_cfg.get("params", {}) or {}

    item_params_grid_all = item_cfg.get("param_grid", {}) or {}
    overall_params_grid_all = overall_cfg.get("param_grid", {}) or {}

    item_grid = (item_params_grid_all.get(item_type, {}) or {}) if isinstance(item_params_grid_all, dict) else {}
    overall_grid = (overall_params_grid_all.get(overall_type, {}) or {}) if isinstance(overall_params_grid_all, dict) else {}

    item_param_combos = _expand_grid(item_grid)
    overall_param_combos = _expand_grid(overall_grid)

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

    comparison_rows: List[Dict[str, Any]] = []

    total_combos = len(item_param_combos) * len(overall_param_combos)
    combo_no = 0

    for item_var in item_param_combos:
        for overall_var in overall_param_combos:
            combo_no += 1
            # build cfg ro this combo
            cfg_g = copy.deepcopy(cfg)

            cfg_g.setdefault("thresholds", {})
            cfg_g["thresholds"]["RC"] = RC_THRESHOLD
            cfg_g["thresholds"]["RI"] = RI_THRESHOLD

            # set model params
            cfg_g.setdefault("model", {})
            cfg_g["model"].setdefault("item", {})
            cfg_g["model"].setdefault("overall", {})

            cfg_g["model"]["item"]["type"] = item_type
            cfg_g["model"]["overall"]["type"] = overall_type
            cfg_g["model"]["item"]["params"] = _merge_params(item_params_base, item_var)
            cfg_g["model"]["overall"]["params"] = _merge_params(overall_params_base, overall_var)

            # prevent per-fold log saving inside sim
            cfg_g["logging"] = dict(cfg.get("logging", {}))
            cfg_g["logging"]["save_logs"] = False

            print(f"\n=== {mode.upper()} Combo {combo_no}/{total_combos} ===")
            print(f"  thresholds: RC={RC_THRESHOLD}, RI={RI_THRESHOLD}")
            print(f"  item: type={item_type}, params={cfg_g['model']['item']['params']}")
            print(f"  overall: type={overall_type}, params={cfg_g['model']['overall']['params']}")


            fold_results: List[Dict[str, Any]] = []
            all_logs: List[pd.DataFrame] = []

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
                logs_df["RC_THRESHOLD"] = float(RC_THRESHOLD)
                logs_df["RI_THRESHOLD"] = float(RI_THRESHOLD)
                logs_df["combo"] = combo_no
                for k, v in item_var.items():
                    logs_df[f"item_param_{k}"] = v
                for k, v in overall_var.items():
                    logs_df[f"overall_param_{k}"] = v
                all_logs.append(logs_df)

            df_fold = pd.DataFrame(fold_results)

            row: Dict[str, Any] = {
                "mode": mode,
                "skill_name": skill_name,
                "csv_path": input_path,
                "cv_seed": random_seed,
                "RC_THRESHOLD": float(RC_THRESHOLD),
                "RI_THRESHOLD": float(RI_THRESHOLD),
                "num_folds": K,
                "combo_no": combo_no,
                "item_type": item_type,
                "overall_type": overall_type,
                "item_params": str(item_var),
                "overall_params": str(overall_var),
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

            # flatten variable params for easy filtering/sorting
            for k, v in item_var.items():
                row[f"item_param_{k}"] = v
            for k, v in overall_var.items():
                row[f"overall_param_{k}"] = v

            comparison_rows.append(row)

            if save_user_logs and save_logs and all_logs:
                logs_all_df = pd.concat(all_logs, ignore_index=True)

                log_subdir = os.path.join(base_log_dir, mode, "user_logs_params")
                if timestamped:
                    log_subdir = os.path.join(log_subdir, run_id)
                os.makedirs(log_subdir, exist_ok=True)

                log_name = f"{mode}_user_logs_combo{combo_no:03d}.csv"
                logs_all_df.to_csv(os.path.join(log_subdir, log_name), index=False)
                print(f"Saved user logs: {os.path.join(log_subdir, log_name)}")

    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(out_base, f"{mode}_params_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to: {comp_path}")
    return comparison_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--outdir", default="outputs/results/cmp_params")
    p.add_argument("--no_user_logs", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    compare_params(
        cfg=cfg,
        output_dir=args.outdir,
        save_user_logs=(not args.no_user_logs),
    )