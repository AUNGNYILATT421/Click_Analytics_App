# cross_validation.py
# Checks how stable the trained model is: the best model type and settings FLAML found are retrained on
# k parts of the training set and scored on the remaining part (k-fold cross-validation). No new search is
# run and the test set is never used, so it costs about k fits of one model.

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

LARGE_TRAINING_SET = 50_000    # above this, 3 folds instead of 5 to keep the time and memory down

# FLAML metric -> scikit-learn scorer (scorers are "higher is better": errors are negated)
SCORERS = {"accuracy": "accuracy", "roc_auc": "roc_auc", "roc_auc_ovr": "roc_auc_ovr", "f1": "f1", "macro_f1": "f1_macro",
           "log_loss": "neg_log_loss", "r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
           "rmse": "neg_root_mean_squared_error"}


def untrained_copy(estimator):
    # Same model type and settings, not trained yet
    if type(estimator).__name__.startswith("CatBoost"):
        # FLAML trains CatBoost with early stopping (up to 8,192 trees, stopping when the validation score stops improving):
        # use the number of trees it kept, and don't let CatBoost write training files to the app folder
        params = estimator.get_params()
        for key in ("early_stopping_rounds", "train_dir"):
            params.pop(key, None)
        params.update(n_estimators=int(estimator.tree_count_), allow_writing_files=False)
        return type(estimator)(**params)
    return clone(estimator)


def best_config_estimator(model):
    # An untrained copy of the best model FLAML found (same type and settings), with the same training-time preprocessing
    automl = model.named_steps["model"] if isinstance(model, Pipeline) else model
    estimator = untrained_copy(automl.model.estimator)
    if isinstance(model, Pipeline):
        return Pipeline([(name, clone(step)) for name, step in model.steps[:-1]] + [("model", estimator)])
    return estimator


def run_cross_validation(model, X, y, metric, problem_type, balanced):
    # Returns (scores per fold, number of folds) or (None, reason)
    folds = 3 if len(X) > LARGE_TRAINING_SET else 5
    if problem_type == "Classification":
        y = LabelEncoder().fit_transform(y)    # some model types (e.g. XGBoost) need classes numbered 0, 1, 2, ...
        smallest = int(np.bincount(y).min())
        if smallest < 2:
            return None, "a class has fewer than 2 rows in the training set"
        folds = min(folds, smallest)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    weights = None
    if balanced:
        from sklearn.utils.class_weight import compute_sample_weight
        weights = compute_sample_weight("balanced", y)
    scorer = get_scorer(SCORERS[metric])
    y = np.asarray(y)
    scores = []
    # One fold at a time with a fresh copy of the model: the model itself already uses the CPU cores, and only one
    # copy is in memory at once (scikit-learn's cross_validate can't copy some FLAML models, e.g. CatBoost)
    for train_rows, test_rows in splitter.split(X, y):
        estimator = best_config_estimator(model)
        fit = {}
        if weights is not None:
            fit[("model__" if isinstance(estimator, Pipeline) else "") + "sample_weight"] = weights[train_rows]
        estimator.fit(X.iloc[train_rows], y[train_rows], **fit)
        scores.append(scorer(estimator, X.iloc[test_rows], y[test_rows]))
    scores = np.array(scores, dtype=float)
    return (-scores if SCORERS[metric].startswith("neg_") else scores), folds


def show_cross_validation(model, X_train, y_train):
    trained = st.session_state.get('trained_settings')
    if trained is None:
        return
    metric_label, metric, problem_type, balanced = trained
    st.markdown("#### Cross-Validation (optional)")
    st.write(f"Checks how stable the model is: the best model FLAML found is retrained on 5 parts of the training set and scored "
             f"({metric_label}) on the part left out each time. The test set isn't used. Scores that vary a lot between parts mean "
             "a single test score could be luck.")
    col1, col2, col3 = st.columns([1, 0.9, 1])
    if col2.button("Run Cross-Validation", use_container_width=True, key="cv_apply"):
        try:
            with st.spinner("Cross-validating the best model..."):
                scores, folds = run_cross_validation(model, X_train, y_train, metric, problem_type, balanced)
        except Exception:
            import logging
            logging.exception("Cross-validation failed")
            scores, folds = None, "this model type couldn't be retrained outside the search"
        st.session_state['cv_result'] = (id(model), metric_label, scores, folds)

    saved = st.session_state.get('cv_result')
    if saved and saved[0] == id(model):
        _, label, scores, folds = saved
        if scores is None:
            st.info(f"Cross-validation isn't available here: {folds}.")
            return
        table = pd.DataFrame({"Part": [f"{i + 1}" for i in range(folds)] + ["Mean", "Spread (std)"],
                              label: list(np.round(scores, 4)) + [round(float(scores.mean()), 4), round(float(scores.std()), 4)]})
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(f"{folds}-fold cross-validation on the {len(X_train):,} training rows: {label} = {scores.mean():.4f} ± {scores.std():.4f}.")
