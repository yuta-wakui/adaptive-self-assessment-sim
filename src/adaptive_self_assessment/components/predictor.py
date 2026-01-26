import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional, Hashable

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_self_assessment.components.model_store import ModelStore

logger = logging.getLogger(__name__)

FrozenParams = Tuple[Tuple[str, Hashable], ...]

def _freeze_params(params: Dict[str, Any]) -> FrozenParams:
    """
    convert model parameters dictionary to a hashable FrozenParams tuple
    Parameters:
    ----------
        params: Dict[str, Any]
            model parameters dictionary
    Returns:
    -------
        FrozenParams
            hashable tuple-form parameters
    """
    def freeze(v: Any) -> Hashable:
        if isinstance(v, dict):
            return _freeze_params(v)
        elif isinstance(v, (list, tuple)):
            return tuple(freeze(i) for i in v)
        else:
            return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

    return tuple(sorted((k, freeze(params[k])) for k in params))

def _get_model_config(cfg: Dict[str, Any], kind: str) -> Tuple[str, Dict[str, Any]]:
    """
    read model configuration from config dictionary
    Parameters:
    ----------
        cfg: Dict[str, Any]
            configuration dictionary
        kind: str
            model kind ("item_model" or "overall_model")
    Returns:
    -------
        mode_type: str
            model type (e.g., "logistic_regression")
        params: Dict[str, Any]
            model parameter settings
    """
    if kind not in ["item_model", "overall_model"]:
        raise ValueError(f"Unsupported model kind: {kind}")

    model_cfg = cfg.get("model", {})
    sub_cfg = model_cfg.get(kind, {})

    model_type = sub_cfg.get("type", "logistic_regression")
    params = sub_cfg.get("params", {})

    if not isinstance(params, dict):
        raise ValueError(f"model.{kind}.params must be a dictionary.")

    return str(model_type), params

def _build_model(model_name: str, params: Dict[str, Any], random_state: int = 42) -> Pipeline:
    """
    build sklearn Pipeline from model name and parameters
    Parameters:
    ----------
        model_name: str
            model type (e.g., "logistic_regression")
        params: Dict[str, Any]
            model parameter settings
        random_state: int
            random seed
    Returns:
    -------
        model: Pipeline
            constructed model pipeline
    """  

    if model_name == "logistic_regression":
        base_params = {
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": random_state,
        }
        base_params.update(params) # override parameters

        clf = LogisticRegression(**base_params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def _predict_with_model(
        model: Pipeline,
        x_cols: List[str],
        x_values: List[Any]
) -> Tuple[int, float]:
    """
    predict class and confidence using the given model
    Parameters:
    ----------
        model: Pipeline
            model used for prediction
        x_cols: List[str]
            feature column names to use
        x_values: List[Any]
            feature data for prediction
    Returns:
    -------
        pred: int
            predicted class label
        confidence: float
            prediction confidence
    """
    x_pred = pd.DataFrame([x_values], columns=x_cols)
    proba = model.predict_proba(x_pred)[0]
    classes = model.named_steps["clf"].classes_
    pred_idx = int(np.argmax(proba))
    return int(classes[pred_idx]), float(proba[pred_idx])

def _validate_columns_exist(df: pd.DataFrame, cols: List[str], where: str) -> None:
    """
    check that required columns exist in the DataFrame
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing in {where}: {missing}")

def _cache_key(*parts: Any) -> Tuple[Hashable, ...]:
    """
    create a hashable cache key from given parts
    """
    def normalize(v):
        if isinstance(v, (list, tuple)):
            return tuple(normalize(x) for x in v)
        elif isinstance(v, dict):
            return tuple(sorted((k, normalize(val)) for k, val in v.items()))
        else:
            return v

    return tuple(normalize(p) for p in parts)

def predict_overall_ws1(
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42,
) -> Tuple[int, float]:
    """
    predict overall score for WS1 using answered/complemented items Ca
    Parameters:
    ----------
        Ca: Dict[str, int]
            answered or complemented items
        df_train: pd.DataFrame 
            training data used for prediction
        cfg: Dict[str, Any]
            configuration dictionary
        fold: int
            cross-validation fold number
        store: Optional[ModelStore]
            model store (for caching)
        random_state: int
            random seed
    Returns:
    -------
        pred: int
            predicted overall score  
        confidence: float
            prediction confidence
    """
    # if config is None
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # get column names
    data_cfg = cfg.get("data", {})
    ws1_cfg = data_cfg.get("ws1", {})
    item_cols: List[str] = ws1_cfg.get("item_cols", [])
    overall_col: str = ws1_cfg.get("overall_col")

    if overall_col is None or not item_cols:
        raise ValueError("overall_col and item_cols must be specified in config for WS1.")
    
    _validate_columns_exist(df_train, item_cols + [overall_col], "training data (WS1 overall)")

    missing_ca = [c for c in item_cols if c not in Ca]
    if missing_ca:
        raise ValueError(f"Missing features in Ca: {missing_ca}")

    # get model configuration from config
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall_model")
    frozen_params = _freeze_params(model_params)

    # get model from model store
    key = _cache_key(
        "ws1",
        "overall",
        int(fold),
        int(random_state),
        tuple(item_cols),
        str(model_type),
        frozen_params
    )

    model = store.get(key) if store is not None else None
    # if model is not in cache, create and train a new one
    if model is None:
        # create training data
        X_train = df_train[item_cols]
        y_train = df_train[overall_col]
        model = _build_model(model_name=model_type, params=model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if store is not None:
            store.set(key, model)

    # predict overall score
    x_values = [Ca[c] for c in item_cols]
    pred, confidence = _predict_with_model(model, item_cols, x_values)
    logger.info(
        f"[WS1][OVERALL] fold={fold} pred={pred} conf={confidence:.4f} "
        f"x_cols={item_cols} x_values={x_values}"
    )

    return pred, confidence

def predict_overall_ws2(
        Pra: int,
        Pca: Dict[str, int],
        Ca: Dict[str, int],
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42,
) -> Tuple[int, float]:
    """
    predict overall score for WS2 using past overall Pra, past checklist Pca, and answered/complemented items Ca
    Parameters:
    ----------
        Pra: int
            past overall score
        Pca: Dict[str, int]
            past checklist results
        Ca: Dict[str, int]
            answered or complemented items
        df_train: pd.DataFrame 
            training data used for prediction
        cfg: Dict[str, Any]
            configuration dictionary
        fold: int
            cross-validation fold number
        store: Optional[ModelStore]
            model store (for caching)
        random_state: int
            random seed
    Returns:
    -------
        pred: int
            predicted overall score  
        confidence: float
            prediction confidence
    """
    # if config is None
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # get column names
    ws2_cfg = cfg.get("data", {}).get("ws2", {})
    pra_col: str = ws2_cfg.get("past_overall_col")
    pca_cols: List[str] = ws2_cfg.get("past_item_cols", [])
    ca_cols: List[str] = ws2_cfg.get("current_item_cols", [])
    ra_col: str = ws2_cfg.get("current_overall_col")

    if pra_col is None or not pca_cols or not ca_cols or ra_col is None:
        raise ValueError("pra_col, pca_cols, ca_cols and ra_col must be specified in config for WS2.")
    
    X_cols = [pra_col] + pca_cols + ca_cols
    _validate_columns_exist(df_train, X_cols + [ra_col], "training data (WS2 overall)")
    
    missing_pca = [c for c in pca_cols if c not in Pca]
    missing_ca = [c for c in ca_cols if c not in Ca]
    if missing_pca or missing_ca:
        raise ValueError(f"Missing features in Pca/Ca: {missing_pca + missing_ca}")

    # get model configuration from config
    model_type, model_params = _get_model_config(cfg=cfg, kind="overall_model")
    frozen_params = _freeze_params(model_params)

    key  = _cache_key(
        "ws2",
        "overall",
        int(fold),
        int(random_state),
        tuple(X_cols),
        str(model_type),
        frozen_params
    )            

    model = store.get(key) if store is not None else None
    # if model is not in cache, create and train a new one
    if model is None:
        # create training data
        X_train = df_train[X_cols]
        y_train = df_train[ra_col]
        model = _build_model(model_name=model_type, params=model_params, random_state=random_state)
        model.fit(X_train, y_train)
        if store is not None:
            store.set(key, model)
    
    # predict overall score
    x_values: List[int] = []
    for c in X_cols:
        if c == pra_col:
            x_values.append(Pra)
        elif c in pca_cols:
            x_values.append(Pca[c])
        else:
            x_values.append(Ca[c])

    pred, confidence = _predict_with_model(model, X_cols, x_values)
    logger.info(
        f"[WS2][OVERALL] fold={fold} pred={pred} conf={confidence:.4f} "
        f"x_cols={X_cols} x_values={x_values}"
    )

    return pred, confidence

def predict_item_ws1(
        Ca: Dict[str, int], 
        C: List[str], 
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    predict unanswered items C one by one using answered/complemented items Ca
    Parameters:
    ----------
        Ca: Dict[str, int]
            answered or complemented items
        C: List
            remaining questions
        df_train: pd.DataFrame 
            training data used for prediction
        cfg: Dict[str, Any]
            configuration dictionary
        fold: int
            cross-validation fold number
        store: Optional[ModelStore]
            model store (for caching)
        random_state: int
            random seed
    Returns:
    -------
        preds: Dict[str, int]
            predicted question answers
        confidences: Dict[str, float]
            confidence of each prediction
    """
    if not C:
        return {}, {}

    preds: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    # if config is None
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # get model configuration from config
    model_type, model_params = _get_model_config(cfg=cfg, kind="item_model")
    frozen_params = _freeze_params(model_params)

    # only use columns that exist in df_train
    ca_cols_exist = [c for c in df_train.columns if c in Ca]

    # predict each item
    for item in C:
        try:
            if item not in df_train.columns:
                raise ValueError(f"Target column '{item}' not found in training data.")
            
            # columns used as features for training
            x_cols: List[str] = [c for c in ca_cols_exist if c != item]
            if not x_cols:
                raise ValueError("No usable features for prediction (x_cols is empty).")
            
            missing = [c for c in x_cols if c not in Ca]
            if missing:
                raise ValueError(f"Missing features in Ca: {missing}")
            
            key = _cache_key(
                "ws1",
                "item",
                int(fold),
                int(random_state),
                str(item),
                tuple(x_cols),
                model_type,
                frozen_params
            )

            model = store.get(key) if store is not None else None
            # if model is not in cache, create and train a new one
            if model is None:
                X_train = df_train[x_cols]
                y_train = df_train[item]
                model = _build_model(
                    model_name=model_type,
                    params=model_params,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                if store is not None:
                    store.set(key, model)
                
            # predict the question item
            x_values = [Ca[c] for c in x_cols]
            pred, confidence = _predict_with_model(model, x_cols, x_values)
            preds[item] = pred
            confidences[item] = confidence


            logger.info(
                f"[WS1][ITEM] fold={fold} item={item} "
                f"pred={pred} conf={confidence:.4f} "
                f"x_cols={x_cols} x_values={x_values}"
            )

        except Exception as e:
            logger.exception(f"[predict_item_ws1 error] target={item}: {e}")
            preds[item] = df_train[item].mode(dropna=True).iloc[0]
            confidences[item] = 0.5

    return preds, confidences

def predict_item_ws2(
        Pra: int, 
        Pca: Dict[str, int], 
        Ca: Dict[str, int], 
        C: List[str], 
        df_train: pd.DataFrame,
        cfg: Dict[str, Any],
        fold: int,
        store: Optional[ModelStore] = None,
        random_state: int = 42
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    predict unanswered items C one by one using past overall Pra, past checklist Pca, and answered/complemented items Ca
    Parameters:
    ----------
        Pra: int
            past overall rating value
        Pca: Dict[str, int]
            past checklist results
        Ca: Dict[str, int]
            answered or complemented items
        C: List
            remaining question items
        df_train: pd.DataFrame 
            training data used for prediction
        cfg: Dict[str, Any]
            configuration dictionary
        fold: int
            cross-validation fold number
        store: Optional[ModelStore]
            model store (for caching)
        random_state: int
            random seed
    Returns:
    -------
        preds: Dict[str, int]
            predicted question answers
        confidences: Dict[str, float]
            confidence of each prediction
    """
    if not C:
        return {}, {}

    preds: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    # if cfg is None
    if cfg is None:
        raise ValueError("cfg must be provided.")

    # get past overall rating and past checklist column names
    ws2_cfg = cfg.get("data", {}).get("ws2", {})
    pra_col: str = ws2_cfg.get("past_overall_col")
    pca_cols: List[str] = ws2_cfg.get("past_item_cols", [])

    if Pra is None:
        raise ValueError("Pra must be provided.")

    if not pra_col or not pca_cols:
        raise ValueError("pra_col and pca_cols must be specified in config for WS2.")

    _validate_columns_exist(df_train, [pra_col] + pca_cols, "training data (WS2 items)")

    missing_pca = [c for c in pca_cols if c not in Pca]
    if missing_pca:
        raise ValueError(f"Missing features in Pca: {missing_pca}")

    # get model configuration from config
    model_type, model_params = _get_model_config(cfg=cfg, kind="item_model")
    frozen_params = _freeze_params(model_params)

    # use only keys in Ca that exist as columns in df_train
    ca_cols_exist = [c for c in df_train.columns if c in Ca]

    # predict each item
    for item in C:
        try:
            if item not in df_train.columns:
                raise ValueError(f"Target column '{item}' not found in training data.")

            # features used for training
            x_cols: List[str] = [pra_col] + pca_cols + [c for c in ca_cols_exist if c != item]
            if not x_cols or item not in df_train.columns:
                raise ValueError("No usable features or target column missing.")
            
            missing: List[str] = []

            # columns needed in Pca
            missing += [c for c in pca_cols if c not in Pca]

            # columns needed in Ca
            ca_needed = [c for c in x_cols if c != pra_col and c not in pca_cols]
            missing += [c for c in ca_needed if c not in Ca]

            if missing:
                raise ValueError(f"Missing features in Pca/Ca: {missing}")
            
            key = _cache_key(
                "ws2",
                "item",
                int(fold),
                int(random_state),
                str(item),
                tuple(x_cols),
                str(model_type),
                frozen_params
            )

            model = store.get(key) if store is not None else None
            # if model is not in cache, create and train a new one
            if model is None:
                X_train = df_train[x_cols]
                y_train = df_train[item]
                model = _build_model(
                    model_name=model_type,
                    params=model_params,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                if store is not None:
                    store.set(key, model)
                
            # predict the question item
            x_values: List[int] = []
            for c in x_cols:
                if c == pra_col:
                    x_values.append(Pra)
                elif c in pca_cols:
                    x_values.append(Pca[c])
                else:
                    x_values.append(Ca[c])

            pred, confidence = _predict_with_model(model, x_cols, x_values)
            preds[item] = pred
            confidences[item] = confidence

            
            logger.info(
                f"[WS2][ITEM] fold={fold} item={item} "
                f"pred={pred} conf={confidence:.4f} "
                f"x_cols={x_cols} x_values={x_values}"
)

        except Exception as e:
            logger.exception(f"[predict_item_ws2 error] target={item}: {e}")
            preds[item] = df_train[item].mode(dropna=True).iloc[0]
            confidences[item] = 0.5
        
    return preds, confidences