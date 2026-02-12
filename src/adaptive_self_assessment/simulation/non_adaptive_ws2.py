# import pandas as pd
# import time
# from typing import List, Dict, Any, Tuple

# from adaptive_self_assessment.components.rng import make_selector_seed
# from adaptive_self_assessment.components.selector import select_question, set_selector_seed
# from adaptive_self_assessment.components.model_store import ModelStore
# from adaptive_self_assessment.components.predictor import predict_overall_ws2

# from adaptive_self_assessment.simulation.common import (
#     load_thresholds,
#     validate_columns,
#     complement_accuracy,
#     summarize_metrics,
# )

# def run_non_adaptive_ws2_simulation(
#         train_df: pd.DataFrame = None,
#         test_df: pd.DataFrame = None,
#         cfg: Dict[str, Any] = None,
#         fold: int = 0
#     ) -> Tuple[Dict[str, Any], pd.DataFrame]:
#     """
#     run non-adaptive (baseline) self-assessment simulation for Ws2.
#     All items are answered by the user; no ML-based complementing.
#     After all items are answered, predict overall score using the same ML model.
#     Parameters:
#     -----------
#         train_df: pd.DataFrame
#             data for training
#         test_df: pd.DataFrame
#             data for testing
#         cfg: Dict[str, Any]
#             simulation configuration
#         fold: int
#             current fold number
#     Returns:
#         results: Dict[str, any]
#             results summary
#         logs_df: pd.DataFrame
#             detailed logs for each user
#     """
#     if cfg is None:
#         raise ValueError("cfg must be provided.")
#     if train_df is None or test_df is None:
#         raise ValueError("train_df and test_df must be provided.")

#     # thresholds
#     th = load_thresholds(cfg)
#     RC_THRESHOLD = th.rc
#     RI_THRESHOLD = th.ri

#     # config
#     data_cfg = cfg.get("data", {})
#     common_cfg = data_cfg.get("common", {})
#     ws2_cfg = data_cfg.get("ws2", {})

#     skill_name: str = common_cfg.get("skill_name", "") or "unknown Skill"
#     id_col: str = common_cfg.get("id_col", "ID")

#     pra_col: str = ws2_cfg.get("past_overall_col", "")
#     pca_cols: List[str] = ws2_cfg.get("past_item_cols", [])
#     ra_col: str = ws2_cfg.get("current_overall_col", "")
#     ca_cols: List[str] = ws2_cfg.get("current_item_cols", [])

#     if (not pra_col) or (not pca_cols) or (not ra_col) or (not ca_cols):
#         raise ValueError(
#             "past_overall_col, past_item_cols, current_overall_col, current_item_cols "
#             "must be specified in cfg['data']['ws2']."
#         )

#     validate_columns(train_df, [id_col, pra_col, ra_col] + pca_cols + ca_cols, "train_df")
#     validate_columns(test_df, [id_col, pra_col, ra_col] + pca_cols + ca_cols, "test_df")

#     # model type (for logging)
#     model_cfg = cfg.get("model", {})
#     overall_model_type: str = model_cfg.get("overall_model", {}).get("type", "logistic_regression")

#     logs: List[Dict[str, Any]] = []
#     store = ModelStore()

#     cv_seed = int(cfg.get("cv", {}).get("random_seed", 42))

#     # run simulation for each user in test set
#     for _, user in test_df.iterrows():
#         user_id = int(user[id_col])

#         Pra: int = int(user[pra_col])  # past overall score
#         Pca: Dict[str, int] = {c: int(user[c]) for c in pca_cols}  # past item scores

#         C: List[str] = ca_cols.copy()
#         Ca: Dict[str, int] = {}

#         answered_items: List[str] = []
#         complemented_items: List[tuple] = []  # always empty in non-adaptive

#         # selector seed (same as adaptive for consistency)
#         seed = make_selector_seed(cv_seed=cv_seed, fold=fold, user_id=user_id)
#         set_selector_seed(seed)

#         start_time = time.time()

#         # answer ALL items — no prediction or complementing
#         while C:
#             ci = select_question(C)
#             answer = int(user[ci])
#             Ca[ci] = answer
#             C.remove(ci)
#             answered_items.append(ci)

#         # after all items are answered, predict overall score
#         Ra_pred, Ra_conf = predict_overall_ws2(
#             Pra=Pra,
#             Pca=Pca,
#             Ca=Ca,
#             df_train=train_df,
#             cfg=cfg,
#             fold=fold,
#             store=store,
#             random_state=42
#         )

#         time_log = time.time() - start_time

#         actual_Ra = int(user[ra_col])
#         is_confident = (Ra_conf >= RI_THRESHOLD)

#         comp_acc, correct_comp_items = complement_accuracy(complemented_items)

#         # record user log (same schema as adaptive ws2)
#         user_log = {
#             "user_id": user_id,
#             "skill": skill_name,
#             "total_questions": len(ca_cols),
#             "num_answered_questions": len(answered_items),
#             "num_complemented_questions": len(complemented_items),
#             "predicted_ra": int(Ra_pred),
#             "actual_ra": actual_Ra,
#             "confidence": float(Ra_conf),
#             "is_confident": is_confident,
#             "correct": int(int(Ra_pred) == int(actual_Ra)),
#             "complement_accuracy": comp_acc,
#             "answered_items": answered_items,
#             "complemented_items": complemented_items,
#             "correct_complement_items": correct_comp_items,
#             "response_time": float(time_log),
#             "RC_THRESHOLD": float(RC_THRESHOLD),
#             "RI_THRESHOLD": float(RI_THRESHOLD),
#             "num_train": len(train_df),
#             "model_type": overall_model_type,
#             "user_seed": seed,
#             "cv_seed": cv_seed,
#             "fold": fold,
#         }
#         logs.append(user_log)

#     # convert logs to DataFrame
#     logs_df = pd.DataFrame(logs)

#     metrics = summarize_metrics(logs_df=logs_df, total_questions=len(ca_cols))

#     sim_results: Dict[str, Any] = {
#         "skill": skill_name,
#         "model_type": overall_model_type,
#         "RC_THRESHOLD": float(RC_THRESHOLD),
#         "RI_THRESHOLD": float(RI_THRESHOLD),
#         "num_train": len(train_df),
#         "num_test": len(test_df),
#         **metrics,
#     }

#     print(f"[DEBUG][NON-ADAPTIVE-WS2][fold={fold}] model cache size = {len(store.models)}")
#     return sim_results, logs_df
