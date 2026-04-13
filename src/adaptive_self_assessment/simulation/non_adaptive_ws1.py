# -*- coding: utf-8 -*-
"""
Simulation module for WS1 (one-session) non-adaptive self-assessment.
This module provides functions to run simulations for WS1 non-adaptive self-assessment.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/simulation/non_adaptive_ws1.py
# Author: Yuta Wakui
# Date: 2026-02-14
# Description: Simulation module for WS1 non-adaptive self-assessment

import pandas as pd
import time
from typing import List, Dict, Any, Tuple

from adaptive_self_assessment.components.model_store import ModelStore
from adaptive_self_assessment.components.predictor import predict_overall_ws1
from adaptive_self_assessment.simulation.common import (
    load_app_config,
    validate_columns,
    summarize_prediction_metrics,
    summarize_log_stats,
)

def run_non_adaptive_ws1_simulation(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int = 0
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    run non-adaptive self-assessment simulation for Ws1.
    Parameters:
    -----------
        train_df: pd.DataFrame
            data for training
        test_df: pd.DataFrame
            data for testing
        cfg: Dict[str, Any]
            simulation configuration
        fold: int
            current fold number
    Returns:
        results: Dict[str, any]
            results summary
        logs_df: pd.DataFrame
            detailed logs for each user
    """

    if cfg is None:
        raise ValueError("config must be provided.")
    if train_df is None or test_df is None:
        raise ValueError("train_df and test_df must be provided.")
    
    # load config settings
    app = load_app_config(cfg)
    
    RI_THRESHOLD: float = app.thresholds.ri

    id_col: str = app.common_data.id_col # user ID
    skill_name: str = app.common_data.skill_name or "unknown_skill"
    
    ra_col: str = app.ws1_data.overall_col # overall score column
    ca_cols: List[str] = app.ws1_data.item_cols # item columns

    # validate columns
    validate_columns(train_df, [id_col, ra_col] + ca_cols, "train_df")
    validate_columns(test_df, [id_col, ra_col] + ca_cols, "test_df")

    # model type (for logging)
    overall_model_type: str = app.overall_model.type

    cv_seed: int = int(app.cv.random_seed)

    store = ModelStore()
    logs: List[Dict[str, Any]] = []

    # run simulation for each user in test set
    for _, user in test_df.iterrows():
        user_id = int(user[id_col]) # get user ID

        # user' item responses (use all actual items for non-adaptive)
        Ca: Dict[str, int] = {c: int(user[c]) for c in ca_cols}

        start_time = time.time()

        # predict overall score using non-adaptive model
        R_pred, Ra_conf = predict_overall_ws1(
            Ca=Ca,
            df_train=train_df,
            cfg=cfg,
            fold=fold,
            store=store,
            random_state=42,
        )
        
        time_log = time.time() - start_time

        actual_Ra = int(user[ra_col]) # actual overall score
        is_confident= (float(Ra_conf) >= RI_THRESHOLD) # whether the overall prediction is confident

        user_log = {
            "user_id": user_id,
            "skill": skill_name,
            "actual_ra": actual_Ra,
            "predicted_ra": int(R_pred),
            "confidence": float(Ra_conf),
            "is_confident": bool(is_confident),
            "correct": int(int(R_pred) == int(actual_Ra)),
            "total_questions": len(ca_cols),
            "num_answered_questions": len(ca_cols), # all items are answered in non-adaptive
            "num_complemented_questions": 0, # all items are not completed in non-adaptive
            "complement_accuracy": None, # not applicable for non-adaptive
            "answered_items": list(sorted(Ca.keys())),
            "complemented_items": [], # no complemented items in non-adaptive
            "response_time": float(time_log),
            "RC_THRESHOLD": None, # not applicable for non-adaptive
            "RI_THRESHOLD": float(RI_THRESHOLD),
            "num_train": len(train_df),
            "model_type": overall_model_type,
            "selector_strategy": None, # not applicable for non-adaptive
            "user_seed": None, # not applicable for non-adaptive
            "cv_seed": cv_seed,
            "fold": fold,
        }

        logs.append(user_log)

    # convert logs to DataFrame
    logs_df = pd.DataFrame(logs)

    # summarize metrics
    pred_metrics = summarize_prediction_metrics(logs_df)
    log_stats = summarize_log_stats(logs_df, total_questions=len(ca_cols))

    sim_results = {
        "skill": skill_name,
        "model_type": overall_model_type,
        "RC_THRESHOLD": None,
        "RI_THRESHOLD": float(RI_THRESHOLD),
        "num_train": len(train_df),
        "num_test": len(test_df),
        "selector_strategy": None,
        **pred_metrics,
        **log_stats,
    }

    return sim_results, logs_df