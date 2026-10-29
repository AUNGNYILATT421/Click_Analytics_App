# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from PIL import Image
from wordcloud import WordCloud
from utils import new_line, short_list
from data_check import class_balance
from config import set_page_config
from session_state import initial_state
from data_loading import load_data
import eda_module
from missing_values_handler import handle_missing_values
from CTGD import handle_categorical_data
from scaling_functions import display_scaling_options
from transformation_functions import display_transformation_options
from feature_engineering import extract_feature, transform_feature, select_feature, show_dataframe
from data_splitting import split_data
from progress_bar import progress_bar
from flaml import AutoML
import io
import re
import logging
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

IMPUTE_STRATEGIES = {"Mean": "mean", "Median": "median", "Most Frequent": "most_frequent"}
SCALERS = {"Standard Scaling": StandardScaler, "MinMax Scaling": MinMaxScaler, "Robust Scaling": RobustScaler}

def add_training_preprocessing(model):
    # Wrap the model in a Pipeline so filling and scaling are learned from X_train only (no test data leaks in)
    # and are saved inside the downloaded model
    impute = st.session_state.get('train_impute', "None")
    scale = st.session_state.get('train_scale', "None")

    steps = []
    if impute in IMPUTE_STRATEGIES:
        steps.append(("imputer", SimpleImputer(strategy=IMPUTE_STRATEGIES[impute])))
    if scale in SCALERS:
        steps.append(("scaler", SCALERS[scale]()))
    if not steps:
        return model

    step_code = ", ".join(f"('{name}', {step!r})" for name, step in steps)
    st.session_state.all_the_process += f"""
# Training-time preprocessing (fitted on X_train only)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
model = Pipeline([{step_code}, ('model', model)])
model.fit(X_train, y_train)
\n """
    # No set_output(transform="pandas") here: with pandas 3 it would give the data text-typed column names again,
    # which scikit-learn 1.4.2 (inside FLAML) can't read. FLAML works fine with the plain arrays instead
    return Pipeline(steps + [("model", model)])

def split_data_parts():
    # All the split data (train + validation + test) that the model will be trained and evaluated on
    X_parts = [st.session_state[k] for k in ('X_train', 'X_val', 'X_test') if st.session_state.get(k) is not None]
    y_parts = [st.session_state[k] for k in ('y_train', 'y_val', 'y_test') if st.session_state.get(k) is not None]
    X, y = pd.concat(X_parts), pd.concat(y_parts)
    return X, y, pd.concat([X, y], axis=1)

def count_duplicate_rows():
    return int(split_data_parts()[2].duplicated().sum())

def find_training_problems():
    X, y, data = split_data_parts()
    problems = []

    # Duplicate rows block training unless the user chose to train with them
    n_dup = int(data.duplicated().sum())
    if n_dup and not st.session_state.get('allow_duplicates', False):
        problems.append(f"**Duplicate rows:** {n_dup} rows are exact copies of another row. "
                        "Remove them in **🕵️‍♂️ Exploratory Data Analysis → Remove Duplicate Rows**, "
                        "or tick **Train with duplicate rows** above.")

    # Duplicate columns: same name. When a file has a repeated header,
    # pandas renames the copies "age" -> "age.1", "age.2", so those count as the same name
    names = [str(c) for c in data.columns]
    name_set = set(names)
    base = [re.sub(r"\.\d+$", "", n) if re.sub(r"\.\d+$", "", n) in name_set else n for n in names]
    # One pass with a dict instead of comparing every pair of columns (slow for thousands of columns)
    first_with_name, dup_pairs = {}, []
    for name, b in zip(names, base):
        if b in first_with_name:
            dup_pairs.append(f"`{name}` and `{first_with_name[b]}`")
        else:
            first_with_name[b] = name
    if dup_pairs:
        problems.append(f"**Duplicate columns:** {', '.join(dup_pairs)} have the same name. "
                        "Remove one column of each pair in **🕵️‍♂️ Exploratory Data Analysis → Delete Columns**.")

    text_cols = X.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
    if text_cols:
        problems.append(f"**Non-numeric features:** **`{text_cols}`** contain text. "
                        "Encode them in **🔠 Handling Categorical Data**, or remove them.")

    missing_cols = X.columns[X.isnull().any()].tolist()
    if missing_cols and st.session_state.get('train_impute', "None") == "None":
        problems.append(f"**Missing values:** **`{missing_cols}`** have missing values. "
                        "Handle them in **⚠️ Missing Values**, or choose **Fill Missing Values** above.")
    if y.isnull().any():
        problems.append("**Missing values in the target:** please fill or drop them in **⚠️ Missing Values**.")

    return problems

def model_bytes(model):
    # Serialize the model together with everything needed to prepare raw new data (in memory, not to a shared file on disk).
    # Use it outside the app with predict_with_bundle.py.
    from importlib.metadata import version
    from recipe import required_inputs
    ss = st.session_state
    schema, recipe, model_columns = ss.get('input_schema') or {}, ss.get('recipe', []), list(ss['X_train'].columns)
    target = ss.get('target_variable')
    bundle = {"format": "click-analytics-model", "version": 1, "model": model, "problem_type": ss.get('problem_type'),
              "target": target, "model_columns": model_columns, "input_schema": schema, "recipe": recipe,
              "needed_inputs": [c for c in required_inputs(recipe, list(schema), [str(c) for c in model_columns]) if c != target],
              "packages": {p: version(p) for p in ("flaml", "scikit-learn", "pandas", "numpy", "lightgbm", "xgboost", "catboost")}}
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()

# Automated model training with FLAML: model types users can choose from (label -> FLAML name)
AUTOML_MODELS = {
    "Classification": {"LightGBM": "lgbm", "XGBoost": "xgboost", "XGBoost (limited depth)": "xgb_limitdepth",
                       "Random Forest": "rf", "Extra Trees": "extra_tree", "CatBoost": "catboost",
                       "Logistic Regression (L1)": "lrl1", "Logistic Regression (L2)": "lrl2", "K-Nearest Neighbors": "kneighbor"},
    "Regression": {"LightGBM": "lgbm", "XGBoost": "xgboost", "XGBoost (limited depth)": "xgb_limitdepth",
                   "Random Forest": "rf", "Extra Trees": "extra_tree", "CatBoost": "catboost", "K-Nearest Neighbors": "kneighbor"},
}
# FLAML's usual selection; slower K-Nearest Neighbors and L2 Logistic Regression are off by default
DEFAULT_AUTOML_MODELS = ["LightGBM", "XGBoost", "XGBoost (limited depth)", "Random Forest", "Extra Trees", "CatBoost", "Logistic Regression (L1)"]
AUTOML_METRICS = {
    "Classification": {"Accuracy": "accuracy", "ROC AUC": "roc_auc", "F1 Score": "f1", "Log Loss": "log_loss"},
    "Regression": {"R2 Score": "r2", "Mean Absolute Error (MAE)": "mae", "Mean Squared Error (MSE)": "mse",
                   "Root Mean Squared Error (RMSE)": "rmse"},
}
# FLAML stores results as a loss: 1 - score for these "higher is better" metrics, the metric itself for the others
LOWER_IS_BETTER = {"log_loss", "mae", "mse", "rmse"}

def flaml_metric(metric, y_train):
    # ROC AUC and F1 need their multiclass versions when the target has more than two classes
    if pd.Series(y_train).nunique() > 2:
        return {"roc_auc": "roc_auc_ovr", "f1": "macro_f1"}.get(metric, metric)
    return metric

def automl_leaderboard(automl, metric, metric_label, problem_type):
    # Best validation score of each model type FLAML tried, best first
    names = {v: k for k, v in AUTOML_MODELS[problem_type].items()}
    rows = []
    for estimator, loss in automl.best_loss_per_estimator.items():
        if loss is None or not np.isfinite(loss):
            continue
        score = loss if metric in LOWER_IS_BETTER else 1 - loss
        rows.append({"Model": names.get(estimator, estimator) + (" 🏆" if estimator == automl.best_estimator else ""),
                     f"Best validation {metric_label}": round(float(score), 4)})
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(board.columns[1], ascending=metric in LOWER_IS_BETTER).reset_index(drop=True)
        board.index += 1
    return board

def fit_and_store(X_train, y_train, problem_type, metric_label, model_labels, time_budget):
    # Mark the model as trained only after the search succeeds, so evaluation never uses a stale model
    if X_train is None or y_train is None:
        st.session_state['trained_model_bool'] = False
        st.error("Please split the data before training a model.")
        return

    metric = flaml_metric(AUTOML_METRICS[problem_type][metric_label], y_train)
    estimators = [AUTOML_MODELS[problem_type][label] for label in model_labels]
    settings = dict(task=problem_type.lower(), metric=metric, estimator_list=estimators, time_budget=time_budget, seed=42, verbose=0)

    st.session_state.all_the_process += f"""
# Model Building --> Automated machine learning (FLAML)
from flaml import AutoML
model = AutoML()
model.fit(X_train, y_train, {", ".join(f"{k}={v!r}" for k, v in settings.items())})
\n """
    # Balanced class weights: rows of rare classes count more (computed from the training set only)
    if problem_type == "Classification" and st.session_state.get('balance_classes') and class_balance(y_train)[0]:
        from sklearn.utils.class_weight import compute_sample_weight
        settings["sample_weight"] = compute_sample_weight("balanced", y_train)
        st.session_state.all_the_process += """
# Balance classes: rows of rare classes count more in training
from sklearn.utils.class_weight import compute_sample_weight
sample_weight = compute_sample_weight("balanced", y_train)   # passed to model.fit(..., sample_weight=sample_weight)
\n """
    automl = AutoML()
    model = add_training_preprocessing(automl)
    # A Pipeline passes the search settings to its last step (the AutoML search) as "model__<setting>"
    fit_settings = {f"model__{k}": v for k, v in settings.items()} if isinstance(model, Pipeline) else settings

    try:
        with st.spinner(f"Searching for the best model (up to {time_budget} seconds)..."):
            model.fit(X_train, y_train, **fit_settings)
    except Exception:
        # Full details go to the server logs; users see a friendly message
        logging.exception("Model training failed")
        st.session_state['trained_model'] = None
        st.session_state['trained_model_bool'] = False
        st.session_state['automl_leaderboard'] = None
        st.session_state['train_message'] = ("error", "Sorry, the model couldn't be trained with the current data and settings. "
                                                      "Please check your data and model options, then try again.")
        st.rerun()

    st.session_state['trained_model'] = model
    st.session_state['trained_model_bool'] = True
    st.session_state['automl_leaderboard'] = automl_leaderboard(automl, metric, metric_label, problem_type)

    # Forget cached predictions and metrics so the new model is evaluated from scratch
    st.session_state['lst_models_predictions'] = []
    st.session_state['models_with_eval'] = {}
    st.session_state['metrics_df'] = pd.DataFrame()
    st.session_state['perm_importance'] = None

    # Rerun so the Train Model button is enabled again; the message is shown after the rerun
    best = {v: k for k, v in AUTOML_MODELS[problem_type].items()}.get(automl.best_estimator, automl.best_estimator)
    st.session_state['train_message'] = ("success", f"Model Trained Successfully! The best model is **{best}**.")
    st.rerun()

def auc_score(model, X, y):
    # AUC is computed from predicted probabilities (not predicted labels); one-vs-rest for more than two classes
    from sklearn.metrics import roc_auc_score
    proba = model.predict_proba(X)
    try:
        if proba.shape[1] == 2:
            return roc_auc_score(y, proba[:, 1])
        return roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_)
    except ValueError:
        return np.nan   # e.g. a set that contains only one class: AUC isn't defined

def plot_multiclass_roc(model, X_test, y_test):
    # One ROC curve per class (that class vs. all others) on the test set
    from sklearn.metrics import RocCurveDisplay
    proba = model.predict_proba(X_test)
    fig, ax = plt.subplots()
    for i, label in enumerate(model.classes_):
        RocCurveDisplay.from_predictions(y_test == label, proba[:, i], name=f"Class {label}", ax=ax)
    ax.set_title("ROC curves on the test set (each class vs. the rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    return fig

def show_permutation_importance(model, X_test, y_test):
    # Model-agnostic importance: how much the test score drops when one feature's values are shuffled
    from sklearn.inspection import permutation_importance
    score_name = "accuracy" if st.session_state.get('problem_type') == "Classification" else "R2 score"

    new_line()
    st.markdown("### Feature Importance")
    st.write(f"Permutation importance: how much the model's test {score_name} drops when one feature's values are randomly shuffled "
             "(10 repeats; the gray lines show the spread). Higher means the model relies on the feature more. "
             "Note: features that carry the same information can both look less important.")
    col1, col2, col3 = st.columns([1, 0.8, 1])
    if col2.button("Compute Feature Importance", use_container_width=True, key="perm_importance_apply"):
        with st.spinner("Computing feature importance..."):
            scoring = "accuracy" if st.session_state.get('problem_type') == "Classification" else "r2"
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring=scoring)
        st.session_state['perm_importance'] = pd.DataFrame({"feature": X_test.columns.astype(str),
                                                            "mean": result.importances_mean,
                                                            "std": result.importances_std}).sort_values("mean", ascending=False)

    importance = st.session_state.get('perm_importance')
    if importance is not None:
        top = importance.head(20).iloc[::-1]   # most important feature at the top of the plot
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(top) + 1)))
        ax.barh(top["feature"], top["mean"], xerr=top["std"], color="skyblue", ecolor="gray", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Drop in test {score_name}")
        ax.set_title("Permutation Feature Importance" + (f" (top 20 of {len(importance)})" if len(importance) > 20 else ""))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def display_model_building_options(X_train, y_train):

    new_line()
    st.markdown("### 🤖 Building the Model")
    new_line()
    with st.expander(" Model Building"):
        st.write("Automated machine learning (FLAML): the app tries several model types, tunes their settings within the "
                 "time budget, and keeps the model with the best validation score.")

        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Target Variable", [st.session_state['target_variable']], key='target_ml', help="The target variable is the variable that you want to predict")
        with col2:
            problem_type = st.selectbox("Problem Type", ["Select", "Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")

        if problem_type not in ("Classification", "Regression"):
            return

        # Search settings (separate keys per problem type, since their options differ)
        col1, col2 = st.columns(2)
        with col1:
            metric_label = st.selectbox("Optimization Metric", list(AUTOML_METRICS[problem_type]), key=f'automl_metric_{problem_type}',
                                        help="The score FLAML tries to improve while comparing models on validation data.")
        with col2:
            time_budget = st.slider("Time Budget (seconds)", min_value=10, max_value=300, value=60, step=10, key='automl_time_budget',
                                    help="How long FLAML searches. More time lets it try more models and settings.")
        options = list(AUTOML_MODELS[problem_type])
        model_labels = st.multiselect("Models to Try", options, default=[m for m in DEFAULT_AUTOML_MODELS if m in options],
                                      key=f'automl_models_{problem_type}', help="FLAML only tries the model types selected here.")

        # Preprocessing learned from the training set only and saved inside the model
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Fill Missing Values (Optional)", ["None"] + list(IMPUTE_STRATEGIES), key='train_impute',
                         help="Fills missing values using statistics from the training set only, so no test data leaks into training.")
        with col2:
            st.selectbox("Scale Features (Optional)", ["None"] + list(SCALERS), key='train_scale',
                         help="Scales features using statistics from the training set only, so no test data leaks into training.")

        # Imbalanced classes: accuracy can look good while the small classes are never predicted
        if problem_type == "Classification" and y_train is not None:
            minority, share = class_balance(y_train)
            if minority:
                st.info(f"**Imbalanced classes:** {short_list([str(c) for c in minority])} "
                        f"{'is' if len(minority) == 1 else 'are'} only {share:.0%} of the training rows"
                        f"{' (the smallest)' if len(minority) > 1 else ''}. A model can then score a high accuracy while rarely predicting "
                        f"{'it' if len(minority) == 1 else 'them'}. "
                        + ("Consider **F1 Score** or **ROC AUC** as the optimization metric, and " if metric_label == "Accuracy" else "You can ")
                        + "tick **Balance classes** so every class counts equally in training.")
                st.checkbox("Balance classes", value=False, key='balance_classes',
                            help="Gives the rows of smaller classes more weight in training (scikit-learn's 'balanced' weights), "
                                 "so the model doesn't ignore them. Nothing is added to or removed from the data.")

        # Block training until the split data is clean, and tell the user why
        n_dup = count_duplicate_rows()
        if n_dup:
            st.checkbox(f"Train with duplicate rows ({n_dup} rows are exact copies of another row)", value=False, key='allow_duplicates',
                        help="Duplicate rows can be genuine (e.g. two records with the same details). Note: copies of the same row "
                             "may end up in both the train and test sets, which can make the test score look better than it really is.")
        training_problems = find_training_problems()
        if training_problems:
            st.warning("**Training is disabled** until these issues are fixed:\n\n"
                       + "\n".join(f"- {problem}" for problem in training_problems)
                       + "\n\nAfter fixing them, please **split the data again**.")
        if not model_labels:
            st.info("Please select at least one model type in **Models to Try**.")

        # A click on Train Model starts a new run. Read the click from session state before drawing the button,
        # so the button can be drawn disabled for the whole training (a disabled button itself returns False)
        blocked = bool(training_problems) or not model_labels
        train_clicked = bool(st.session_state.get('train_model', False)) and not blocked
        col1, col2, col3 = st.columns([1, 0.7, 1])
        col2.button("Train Model", disabled=blocked or train_clicked, key='train_model', use_container_width=True)
        if train_clicked:
            fit_and_store(X_train, y_train, problem_type, metric_label, model_labels, time_budget)

        # Shown after training reruns the page: the result, the leaderboard, and the trained model's download
        message = st.session_state.pop('train_message', None)
        if message:
            kind, text = message
            if kind == "success":
                st.success(text)
            else:
                st.error(text, icon="⚠️")
        trained = st.session_state.get('trained_model')
        board = st.session_state.get('automl_leaderboard')
        if trained is not None and board is not None and not board.empty:
            st.markdown("#### Leaderboard")
            st.write("The best validation score of each model type FLAML tried (🏆 = the model used for evaluation).")
            st.dataframe(board, use_container_width=True)
        if trained is not None:
            col1, col2, col3 = st.columns([1, 0.7, 1])
            col2.download_button("Download Model", lambda: model_bytes(trained), "model.pkl", use_container_width=True, key='save_model',
                                 help="The model together with the recorded preparation steps. Use predict_with_bundle.py from this app's folder to predict raw new data with it.")
