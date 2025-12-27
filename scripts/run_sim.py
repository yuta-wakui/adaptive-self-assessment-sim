import os
import pandas as pd
import argparse
import yaml

from datetime import datetime
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold

from adaptive_self_assessment.simulation.ws1 import run_ws1_simulation
from adaptive_self_assessment.simulation.ws2 import run_ws2_simulation

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to the config file")
    return p.parse_args()

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    YAML形式の設定ファイルを読み込む関数
    
    Parameters:
    ----------
    config_path: str
        設定ファイルのパス（デフォルト； "configs/config.yaml"）
    
    Returns:
    -------
    cfg: Dict[str, Any] 
        読み込んだ設定内容を辞書形式で返す
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def _calc_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    """
    データフレームの指定列の平均を計算する関数
    Parameters:
    -----------
        df: pd.DataFrame
            データフレーム
        col: str
            列名
    Returns:
    -------
        Optional[float]
            指定列の平均値。データがない場合はNoneを返す
        """
    if col not in df.columns or df.empty:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return float(s.mean())

def run_simulations(config_path: str) -> pd.DataFrame:
    """
    1回分（WS1）または2回分（WS2）の自己評価データで適応型自己評価のシミュレーションを交差検証で実行する関数
    設定はconfig.yamlから読み込む
    Parameters:
    -----------
        config_path: str
            設定ファイルのパス
    Returns:
    -------
        results_df: pd.DataFrame
            シミュレーション結果のデータフレーム
    """
    # config設定の読み込み
    cfg = load_config(config_path)

    # 実行モード（WS1/WS2)
    mode: str = cfg.get("mode", {}).lower()
    if mode not in ("ws1", "ws2"):
        raise ValueError(f"Unsupported mode: {mode} (expected 'ws1' or 'ws2')")

    # モデルタイプ
    model_cfg = cfg.get("model", {})
    overall_model_type: str = model_cfg.get("overall", {}).get("type", "logistic_regression")

    # 閾値設定
    thresholds_cfg = cfg.get("thresholds", {})
    RC_THRESHOLD: float = float(thresholds_cfg.get("RC", 0.80))
    RI_THRESHOLD: float = float(thresholds_cfg.get("RI", 0.70))
    rc_str = str(RC_THRESHOLD).replace(".", "p")
    ri_str = str(RI_THRESHOLD).replace(".", "p")    

    # データ設定
    data_cfg = cfg.get("data", {})
    common_cfg = data_cfg.get("common", {})
    ws_cfg = data_cfg.get(mode, {})

    # 入力データ設定
    input_path: str = common_cfg.get("input_path", None)
    if not input_path or not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    skill_name = common_cfg.get("skill_name", "unknown_skill")
    ignore_items: List[str] = common_cfg.get("ignore_items", [])

    # ラベル列名
    ra_col: str = ws_cfg.get("ra_col", "")
    if not ra_col:
        raise ValueError("Label column name 'ra_col' must be specified in config.") 
    
    # 交差検証設定
    cv_cfg = cfg.get("cv", {})
    K: int = int(cv_cfg.get("folds", 5))
    stratified: bool = bool(cv_cfg.get("stratified", True))
    random_seed: int = int(cv_cfg.get("random_seed", 42))

    splitter = (
        StratifiedKFold(n_splits=K, shuffle=True, random_state=random_seed)
        if stratified
        else KFold(n_splits=K, shuffle=True, random_state=random_seed)
    )

    # 結果保存設定
    results_cfg = cfg.get("results", {})
    save_csv: bool = bool(results_cfg.get("save_csv", True))
    output_dir: str = results_cfg.get("output_dir", "outputs/results")
    out_timestamped: bool = bool(results_cfg.get("timestamped", True))
    filename_suffix: str = str(results_cfg.get("filename_suffix", "")).strip()
    save_fold_results: bool = bool(results_cfg.get("save_fold_results", False))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{filename_suffix}" if filename_suffix else ""

    print(f"=== {mode.upper()} Simulation Started ===")
    print(f"input: {input_path}")
    print(f"skill: {skill_name}")
    print(f"model(overall): {overall_model_type}")
    print(f"thresholds: RC={RC_THRESHOLD}, RI={RI_THRESHOLD}")
    print(f"cv: folds={K}, stratified={stratified}, seed={random_seed}")

    # データの読み込み
    df = pd.read_csv(input_path)

    # 無視する項目の除去
    drop_cols = [col for col in ignore_items if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped ignore_items columns: {drop_cols}") 

    print(f"\n==== {mode.upper()}: {skill_name} ====")
    print(f"csv: {input_path}")
    print(f"n_rows: {len(df)}, n_cols: {len(df.columns)}")

    if ra_col not in df.columns:
        raise ValueError(f"Label column '{ra_col}' not found in {input_path}.")

    y = df[ra_col].values if stratified else None

    fold_results: List[Dict[str, Any]] = []
    all_fold_results: List[Dict[str, Any]] = []

    # 交差検証でシミュレーションを実行
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, y), start=1):
        print(f"---- Fold {fold}/{K} ----")
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        if mode == "ws1":
            sim = run_ws1_simulation(train_df=train_df, test_df=test_df, cfg=cfg)
        else:
            sim = run_ws2_simulation(train_df=train_df, test_df=test_df, cfg=cfg)

        # fold番号など補助情報だけ付与
        sim = dict(sim)
        sim["fold"] = fold
        sim["mode"] = mode
        sim["csv_path"] = input_path
        fold_results.append(sim)
 
    df_fold = pd.DataFrame(fold_results)

    if save_fold_results:
        all_fold_results.extend(fold_results)

    # 各foldの結果を集計
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
        "avg_reduction_rate": _calc_mean(df_fold, "reduction_rate"),
        "accuracy_over_threshold": _calc_mean(df_fold, "accuracy_over_threshold"),
        "f1_macro_over_threshold": _calc_mean(df_fold, "f1_macro_over_threshold"),
        "coverage_over_threshold": _calc_mean(df_fold, "coverage_over_threshold"),
        "accuracy_all": _calc_mean(df_fold, "accuracy_all"),
        "f1_macro_all": _calc_mean(df_fold, "f1_macro_all"),
        "avg_response_time": _calc_mean(df_fold, "avg_response_time"),
    }

    results_df = pd.DataFrame([row])

    # 結果の保存先ディレクトリ
    subdir = os.path.join(output_dir, mode, "sim_results")
    if out_timestamped:
        subdir = os.path.join(subdir, run_id)

    # 結果の保存
    if save_csv:
        os.makedirs(subdir, exist_ok=True)

        # skill平均結果の保存
        out_name = f"{mode}_results_rc{rc_str}_ri{ri_str}{suffix}.csv"
        out_path = os.path.join(subdir, out_name)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved results to: {out_path}")
    
    if save_fold_results and all_fold_results:
        os.makedirs(subdir, exist_ok=True)
        
        fold_df = pd.DataFrame(all_fold_results)
        fold_out_name = f"{mode}_fold_results_rc{rc_str}_ri{ri_str}{suffix}.csv"
        fold_out_path = os.path.join(subdir, fold_out_name)
        fold_df.to_csv(fold_out_path, index=False)
        print(f"Saved fold results to: {fold_out_path}")

    print(f"=== {mode.upper()} Simulation Completed ===")
    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_simulations(config_path=args.config)