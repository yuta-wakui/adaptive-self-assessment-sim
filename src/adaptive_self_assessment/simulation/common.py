# -*- coding: utf-8 -*-

"""
common utilities for adaptive self-assessment simulations.
This module provides functions for loading configurations, calculating means,
and summarizing simulation metrics.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/simulation/common.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Common utilities for adaptive self-assessment simulations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LOG_REQUIRED_COLS = [
        "actual_ra", "predicted_ra", "is_confident",
        "num_answered_questions", "num_complemented_questions",
        "complement_accuracy", "response_time",
        ]

# ----------------------------
# Config dataclasses
# ----------------------------

@dataclass(frozen=True)
class Thresholds:
    """Confidence thresholds for the adaptive process."""
    rc: float = 0.80
    ri: float = 0.70

@dataclass(frozen=True)
class CVConfig:
    """Configuration for cross-validation."""
    folds: int = 5
    stratified: bool = True
    random_seed: int = 42

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for predictive models."""
    type: str = "logistic_regression"
    params: Dict[str, Any] = field(default_factory=dict)

# dataclass for common data configuration
@dataclass(frozen=True)
class CommonDataConfig:
    """Common data configuration settings."""
    skill_name: str = "unknown_skill"
    id_col: str = "ID"
    ignore_items: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class WS1DataConfig:
    """Data configuration settings specific to WS1."""
    overall_col: str = "" 
    item_cols: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class WS2DataConfig:
    """Data configuration settings specific to WS2."""
    past_overall_col: str = ""
    past_item_cols: List[str] = field(default_factory=list)
    current_overall_col: str = ""
    current_item_cols: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class AppConfig:
    """ Overall application configuration."""
    thresholds: Thresholds = field(default_factory=Thresholds)
    cv: CVConfig = field(default_factory=CVConfig)
    item_model: ModelConfig = field(default_factory=ModelConfig)
    overall_model: ModelConfig = field(default_factory=ModelConfig)
    common_data: CommonDataConfig = field(default_factory=CommonDataConfig)
    ws1_data: WS1DataConfig = field(default_factory=WS1DataConfig)
    ws2_data: WS2DataConfig = field(default_factory=WS2DataConfig)

# ----------------------------
# Config validation
# ----------------------------

def validate_thresholds(t: Thresholds) -> None:
    if not (0.0 <= t.rc <= 1.0 and 0.0 <= t.ri <= 1.0):
        raise ValueError(f"thresholds must be in [0, 1]: rc={t.rc}, ri={t.ri}")

def validate_cv_config(cv: CVConfig) -> None:
    if cv.folds <= 1:
        raise ValueError(f"cv.folds must be >= 2: {cv.folds}")

def validate_ws1_config(ws1: WS1DataConfig) -> None:
    if not ws1.overall_col:
        raise ValueError("data.ws1.overall_col must be provided.")
    if not ws1.item_cols:
        raise ValueError("data.ws1.item_cols must be a non-empty list.")

def validate_ws2_config(ws2: WS2DataConfig) -> None:
    if not ws2.past_overall_col:
        raise ValueError("data.ws2.past_overall_col must be provided.")
    if not ws2.past_item_cols:
        raise ValueError("data.ws2.past_item_cols must be a non-empty list.")
    if not ws2.current_overall_col:
        raise ValueError("data.ws2.current_overall_col must be provided.")
    if not ws2.current_item_cols:
        raise ValueError("data.ws2.current_item_cols must be a non-empty list.")

# ----------------------------
# Loaders
# ----------------------------

def load_thresholds(cfg: Dict[str, Any]) -> Thresholds:
    t = cfg.get("thresholds", {}) or {}
    out = Thresholds(
        rc=float(t.get("RC", 0.80)),
        ri=float(t.get("RI", 0.70)),
    )
    validate_thresholds(out)
    return out

def load_cv_config(cfg: Dict[str, Any]) -> CVConfig:
    cv = cfg.get("cv", {}) or {}
    strat = cv.get("stratified", True)
    if not isinstance(strat, bool):
        raise ValueError("cv.stratified must be a boolean.")
    out = CVConfig(
        folds=int(cv.get("folds", 5)),
        stratified=strat,
        random_seed=int(cv.get("random_seed", 42)),
    )
    validate_cv_config(out)
    return out

def load_model_config(cfg: Dict[str, Any], kind: str) -> ModelConfig:
    """
    kind: "item_model" or "overall_model"
    expects: cfg["model"][kind] = {"type": "...", "params": {...}}
    """
    if kind not in {"item_model", "overall_model"}:
        raise ValueError(f"kind must be 'item_model' or 'overall_model': {kind}")

    model = cfg.get("model", {}) or {}
    sub = model.get(kind, {}) or {}
    model_type = str(sub.get("type", "logistic_regression"))
    params = sub.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError(f"model.{kind}.params must be a dict.")
    return ModelConfig(type=model_type, params=params)

def load_common_data_config(cfg: Dict[str, Any]) -> CommonDataConfig:
    common = (cfg.get("data", {}) or {}).get("common", {}) or {}
    ignore = common.get("ignore_items", []) or []
    if not isinstance(ignore, list):
        raise ValueError("data.common.ignore_items must be a list.")
    return CommonDataConfig(
        skill_name=str(common.get("skill_name", "unknown_skill")),
        id_col=str(common.get("id_col", "ID")),
        ignore_items=[str(x) for x in ignore],
    )

def load_ws1_data_config(cfg: Dict[str, Any]) -> WS1DataConfig:
    ws1 = (cfg.get("data", {}) or {}).get("ws1", {}) or {}
    out = WS1DataConfig(
        overall_col=str(ws1.get("overall_col", "")),
        item_cols=[str(x) for x in ws1.get("item_cols", []) or []],
    )
    validate_ws1_config(out)
    return out

def load_ws2_data_config(cfg: Dict[str, Any]) -> WS2DataConfig:
    ws2 = (cfg.get("data", {}) or {}).get("ws2", {}) or {}
    out = WS2DataConfig(
        past_overall_col=str(ws2.get("past_overall_col", "")),
        past_item_cols=[str(x) for x in ws2.get("past_item_cols", []) or []],
        current_overall_col=str(ws2.get("current_overall_col", "")),
        current_item_cols=[str(x) for x in ws2.get("current_item_cols", []) or []],
    )
    validate_ws2_config(out)
    return out

def load_app_config(cfg: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        thresholds=load_thresholds(cfg),
        cv=load_cv_config(cfg),
        item_model=load_model_config(cfg, "item_model"),
        overall_model=load_model_config(cfg, "overall_model"),
        common_data=load_common_data_config(cfg),
        ws1_data=load_ws1_data_config(cfg),
        ws2_data=load_ws2_data_config(cfg),
    )

# ----------------------------
# Validation helpers
# ----------------------------

def validate_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    """
    Validate that required columns are present in the DataFrame.
    Parameters:
    -----------
        df: pd.DataFrame
            DataFrame to validate
        required: Sequence[str]
            List of required column names
        df_name: str
            Name of the DataFrame (for error messages)
    Returns:
    -------
        None
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")

# (item, predicted, confidence, actual)
ComplementedItem = Tuple[str, int, float, int]

def complement_accuracy(complemented_items: Sequence[ComplementedItem]) -> Tuple[Optional[float], List[str]]:
    """
    Calculate the accuracy of complemented items.
    Parameters:
    -----------
        complemented_items: Sequence[ComplementedItem]
            List of tuples (item, predicted, confidence, actual)
    Returns:
    -------
        accuracy: Optional[float]
            Accuracy of complemented items, or None if no items
        correct_items: List[str]
            List of correctly complemented item names
    """
    if not complemented_items:
        return None, []
    
    correct_items = [item for (item, pred, _, actual) in complemented_items if pred == actual]
    return len(correct_items) / len(complemented_items), correct_items

# ----------------------------
# Metrics summarization
# ----------------------------

def summarize_metrics(
    logs_df: pd.DataFrame,
    total_questions: int,
) -> Dict[str, Any]:
    """
    Summarize simulation metrics from logs DataFrame.
    Parameters:
    -----------
        logs_df: pd.DataFrame
            Logs DataFrame containing simulation results
        total_questions: int
            Total number of questions in the assessment
    Returns:
    -------
        metrics: Dict[str, Any]
            Dictionary of summarized metrics
    """

    # empty guard
    if logs_df is None or logs_df.empty:
        return {
            "accuracy_all": None,
            "f1_macro_all": None,
            "accuracy_over_threshold": None,
            "f1_macro_over_threshold": None,
            "coverage_over_threshold": None,
            "total_questions": int(total_questions),
            "avg_answered_questions": None,
            "avg_complemented_questions": None,
            "avg_complement_accuracy": None,
            "reduction_rate": None,
            "avg_response_time": None,
            "max_response_time": None,
            "min_response_time": None,
        }
    
    validate_columns(logs_df, LOG_REQUIRED_COLS, "logs_df")

    y_true = logs_df["actual_ra"].astype(int)
    y_pred = logs_df["predicted_ra"].astype(int)

    accuracy_all = float(accuracy_score(y_true, y_pred) * 100.0)
    f1_macro_all = float(f1_score(y_true, y_pred, average="macro") * 100.0)

    confident_df = logs_df[logs_df["is_confident"].astype(bool)]
    if not confident_df.empty:
        y_true_c = confident_df["actual_ra"].astype(int)
        y_pred_c = confident_df["predicted_ra"].astype(int)
        accuracy_over = float(accuracy_score(y_true_c, y_pred_c) * 100)
        f1_over = float(f1_score(y_true_c, y_pred_c, average="macro") * 100)
        coverage = float(len(confident_df) / len(logs_df) * 100)
    else:
        accuracy_over = f1_over = coverage = None

    avg_answered = float(logs_df["num_answered_questions"].mean())
    avg_complemented = float(logs_df["num_complemented_questions"].mean())

    if total_questions <= 0:
        reduction_rate = None
    else:
        reduction_rate = float((1.0 - avg_answered / total_questions) * 100.0)

    valid_comp = logs_df[logs_df["num_complemented_questions"] > 0]
    avg_comp_acc = float(valid_comp["complement_accuracy"].mean() * 100.0) if not valid_comp.empty else None

    avg_rt = max_rt = min_rt = None

    rt = logs_df["response_time"].dropna()
    if not rt.empty:
        avg_rt = float(rt.mean())
        max_rt = float(rt.max())
        min_rt = float(rt.min())

    return {
        "accuracy_all": accuracy_all,
        "f1_macro_all": f1_macro_all,
        "accuracy_over_threshold": accuracy_over,
        "f1_macro_over_threshold": f1_over,
        "coverage_over_threshold": coverage,
        "total_questions": int(total_questions),
        "avg_answered_questions": avg_answered,
        "avg_complemented_questions": avg_complemented,
        "avg_complement_accuracy": avg_comp_acc,
        "reduction_rate": reduction_rate,
        "avg_response_time": avg_rt,
        "max_response_time": max_rt,
        "min_response_time": min_rt,
    }