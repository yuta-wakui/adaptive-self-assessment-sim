# -*- coding: utf-8 -*-

"""
Simulation module for WS1 (one-session) adaptive self-assessment.
This module provides functions to run simulations for WS1 adaptive self-assessment.

Copyright (c) 2026 Yuta Wakui
Licensed under the MIT License.
"""

# File: src/adaptive_self_assessment/simulation/ws1.py
# Author: Yuta Wakui
# Date: 2026-01-29
# Description: Simulation module for WS1 adaptive self-assessment

import pandas as pd
import time
from typing import List, Dict, Any, Tuple

from adaptive_self_assessment.components.rng import make_selector_seed
from adaptive_self_assessment.components.selector import select_question, set_selector_seed
from adaptive_self_assessment.components.model_store import ModelStore
from adaptive_self_assessment.components.predictor import predict_item_ws1, predict_overall_ws1

from adaptive_self_assessment.simulation.common import (
    load_app_config,
    validate_columns,
    complement_accuracy,
    summarize_metrics,
    ComplementedItem,
)

def run_ws1_simulation(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int = 0
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    run adaptive self-assessment simulation for Ws1.
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
        raise ValueError("cfg must be provided.")
    if train_df is None or test_df is None:
        raise ValueError("train_df and test_df must be provided.")
    
    # load config settings
    app = load_app_config(cfg)

    RC_THRESHOLD: float = app.thresholds.rc
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

    logs: List[Dict[str, Any]] = []
    store = ModelStore()

    cv_seed = int(app.cv.random_seed)

    # run simulation for each user in test set
    for _, user in test_df.iterrows():
        user_id = int(user[id_col]) # get user ID

        C: List[str] = ca_cols.copy() # unanswered items
        Ca: Dict[str, int] = {} # answered or complemented items

        answered_items: List[str] = [] # items actually answered by user
        complemented_items: List[ComplementedItem] = [] # complemented items: (item, predicted_value, confidence, actual_value)

        # selector seed
        seed = make_selector_seed(cv_seed=cv_seed, fold=fold, user_id=user_id)
        set_selector_seed(seed)

        start_time = time.time()  

        # repeat until all items are answered or complemented
        while C:
            ci = select_question(C) # select next question

            # ask user and get actual answer
            answer = int(user[ci]) # use actual answer from test_df
            Ca[ci] = answer # add to answered/complemented items
            C.remove(ci) # remove from unanswered items
            answered_items.append(ci) # record answered item

            # predict remaining items
            Rc_preds, Rc_confidences = predict_item_ws1(
                Ca=Ca,
                C=C, 
                df_train=train_df,
                cfg=cfg,
                fold=fold,
                store=store,
                random_state=42
            )

            # if the confidence of prediction exceeds threshold, complement the item
            for item in C[:]: # iterate over a copy of C
                conf = float(Rc_confidences.get(item, 0.0)) # get confidence
                if conf >= RC_THRESHOLD:
                    pred_value = int(Rc_preds[item])
                    Ca[item] = pred_value # add to answered/complemented items
                    C.remove(item) # remove from unanswered items
                    actual_value = int(user[item]) # get actual value
                    complemented_items.append((item, pred_value, conf, actual_value)) # record complemented information
        
        # after all items are answered or complemented, predict overall score
        Ra_pred, Ra_conf = predict_overall_ws1(
            Ca=Ca,
            df_train=train_df,
            cfg=cfg,
            fold=fold,
            store=store,
            random_state=42
        )

        time_log = time.time() - start_time

        actual_Ra = int(user[ra_col])
        is_confident= (Ra_conf >= RI_THRESHOLD) # whether the overall prediction is confident

        comp_acc, correct_comp_items = complement_accuracy(complemented_items)

        # record user log
        user_log = {
            "user_id": user_id,
            "skill": skill_name,
            "total_questions": len(ca_cols),
            "num_answered_questions": len(answered_items),
            "num_complemented_questions": len(complemented_items),
            "predicted_ra": int(Ra_pred),
            "actual_ra": actual_Ra,
            "confidence": float(Ra_conf),
            "is_confident": is_confident,
            "correct": int(int(Ra_pred) == int(actual_Ra)),
            "complement_accuracy": comp_acc,
            "answered_items": answered_items,
            "complemented_items": complemented_items,
            "correct_complement_items": correct_comp_items,
            "response_time": float(time_log),
            "RC_THRESHOLD": float(RC_THRESHOLD),
            "RI_THRESHOLD": float(RI_THRESHOLD),
            "num_train": len(train_df),
            "model_type": overall_model_type,
            "user_seed": seed,
            "cv_seed": cv_seed,
            "fold": fold,
        }

        logs.append(user_log)

    # convert logs to DataFrame
    logs_df = pd.DataFrame(logs)

    # summarize metrics
    metrics = summarize_metrics(logs_df, total_questions=len(ca_cols))

    sim_results: Dict[str, Any] = {
        "skill": skill_name,
        "model_type": overall_model_type,
        "RC_THRESHOLD": RC_THRESHOLD,
        "RI_THRESHOLD": RI_THRESHOLD,
        "num_train": len(train_df),
        "num_test": len(test_df),
        **metrics,
    }
    
    print(f"[DEBUG][WS1][fold={fold}] model cache size = {len(store.models)}")
    return sim_results, logs_df