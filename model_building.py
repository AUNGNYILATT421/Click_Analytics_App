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
from utils import new_line
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
    return Pipeline(steps + [("model", model)]).set_output(transform="pandas")

def find_training_problems():
    # Check all the split data (train + validation + test) that the model will be trained and evaluated on
    X_parts = [st.session_state[k] for k in ('X_train', 'X_val', 'X_test') if st.session_state.get(k) is not None]
    y_parts = [st.session_state[k] for k in ('y_train', 'y_val', 'y_test') if st.session_state.get(k) is not None]
    X, y = pd.concat(X_parts), pd.concat(y_parts)
    data = pd.concat([X, y], axis=1)
    problems = []

    n_dup = int(data.duplicated().sum())
    if n_dup:
        problems.append(f"**Duplicate rows:** {n_dup} rows are exact copies of another row. "
                        "Remove them in **🕵️‍♂️ Exploratory Data Analysis → Remove Duplicate Rows**.")

    # Duplicate columns: same name. When a file has a repeated header,
    # pandas renames the copies "age" -> "age.1", "age.2", so those count as the same name
    names = [str(c) for c in data.columns]
    base = [re.sub(r"\.\d+$", "", n) if re.sub(r"\.\d+$", "", n) in names else n for n in names]
    dup_pairs = [f"`{names[i]}` and `{names[j]}`" for i in range(len(names)) for j in range(i) if base[i] == base[j]]
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

def fit_and_store(model, X_train, y_train, col):
    # Mark the model as trained only after fit succeeds, so evaluation never uses a stale model
    if X_train is None or y_train is None:
        st.session_state['trained_model_bool'] = False
        st.error("Please split the data before training a model.")
        return

    model = add_training_preprocessing(model)

    try:
        model.fit(X_train, y_train)
    except Exception:
        # Full details go to the server logs; users see a friendly message
        logging.exception("Model training failed")
        st.session_state['trained_model'] = None
        st.session_state['trained_model_bool'] = False
        st.error("Sorry, the model couldn't be trained with the current data and settings. "
                 "Please check your data and model options, then try again.", icon="⚠️")
        return

    st.session_state['trained_model'] = model
    st.session_state['trained_model_bool'] = True

    # Forget cached predictions and metrics so the new model is evaluated from scratch
    st.session_state['lst_models_predictions'] = []
    st.session_state['models_with_eval'] = {}
    st.session_state['metrics_df'] = pd.DataFrame()
    st.success("Model Trained Successfully!")

    # Download the model (serialized in memory, not to a shared file on disk)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    col.download_button("Download Model", buffer.getvalue(), "model.pkl", use_container_width=True, key='save_model')

def display_model_building_options(X_train, y_train):

    new_line()
    st.markdown("### 🤖 Building the Model")
    new_line()
    problem_type = ""
    with st.expander(" Model Building"):    
        
        target, problem_type, model = "", "", ""
        col1, col2, col3 = st.columns(3)

        with col1:
            target = st.selectbox("Target Variable", [st.session_state['target_variable']] , key='target_ml', help="The target variable is the variable that you want to predict")
            new_line()

        with col2:
            problem_type = st.selectbox("Problem Type", ["Select", "Classification", "Regression"], key='problem_type', help="The problem type is the type of problem that you want to solve")

        with col3:

            if problem_type == "Classification":
                model = st.selectbox("Model", ["Select", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()

            elif problem_type == "Regression":
                model = st.selectbox("Model", ["Linear Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                     key='model', help="The model is the algorithm that you want to use to solve the problem")
                new_line()

        # Preprocessing learned from the training set only and saved inside the model
        if problem_type in ("Classification", "Regression") and model != "Select":
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Fill Missing Values (Optional)", ["None"] + list(IMPUTE_STRATEGIES), key='train_impute',
                             help="Fills missing values using statistics from the training set only, so no test data leaks into training.")
            with col2:
                st.selectbox("Scale Features (Optional)", ["None"] + list(SCALERS), key='train_scale',
                             help="Scales features using statistics from the training set only, so no test data leaks into training.")

        # Block training until the split data is clean, and tell the user why
        training_problems = []
        if problem_type in ("Classification", "Regression") and model != "Select":
            training_problems = find_training_problems()
            if training_problems:
                st.warning("**Training is disabled** until these issues are fixed:\n\n"
                           + "\n".join(f"- {problem}" for problem in training_problems)
                           + "\n\nAfter fixing them, please **split the data again**.")


        if target != "Select" and problem_type and model:
            
            if problem_type == "Classification":
                 
                if model == "Logistic Regression":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        penalty = st.selectbox("Penalty (Optional)", ["l2", "l1", None], key='penalty')

                    with col2:
                        # Only offer the solvers that support the chosen penalty
                        solvers = {"l2": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                                   "l1": ["liblinear", "saga"],
                                   None: ["lbfgs", "newton-cg", "sag", "saga"]}[penalty]
                        solver = st.selectbox("Solver (Optional)", solvers, key='solver')

                    with col3:
                        C = st.number_input("C (Optional)", min_value=1e-5, max_value=1.0, value=1.0, step=0.05, key='C')

                    
                    col1, col2, col3 = st.columns([1,1,1])
                    if col2.button("Train Model", disabled=bool(training_problems), use_container_width=True):
                        
                        
                        progress_bar()

            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='{penalty}', solver='{solver}', C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors **Required**", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')

                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')

                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')

                    
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems), use_container_width=True):
                        progress_bar()


                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "Support Vector Machine":
                        
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["rbf", "poly", "linear", "sigmoid"], key='kernel')
    
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=100, value=3, step=1, key='degree')
    
                    with col3:
                        C = st.number_input("C (Optional)", min_value=1e-5, max_value=1.0, value=1.0, step=0.05, key='C')
    
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems), use_container_width=True):

                        progress_bar()
    
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVC
model = SVC(kernel='{kernel}', degree={degree}, C={C}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.svm import SVC
                        model = SVC(kernel=kernel, degree=degree, C=C, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "Decision Tree":
                            
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
        
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
        
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=2, max_value=100, value=2, step=1, key='min_samples_split')
                            
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems), use_container_width=True):
                        progress_bar()
        
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "Random Forest":
                                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["gini", "entropy", "log_loss"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=2, max_value=100, value=2, step=1, key='min_samples_split')
                                
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems), use_container_width=True):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split}, random_state=42)
model.fit(X_train, y_train)
\n """
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "XGBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.01, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}', random_state=42)
model.fit(X_train, y_train)
\n """
                        from xgboost import XGBClassifier
                        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == 'LightGBM':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.01, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss"], key='boosting_type')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> LightGBM
from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
model.fit(X_train, y_train)
\n """
                        from lightgbm import LGBMClassifier
                        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

                if model == 'CatBoost':

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=5, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.01, max_value=1.0, value=0.1, step=0.05, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["Ordered", "Plain"], key='boosting_type')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> CatBoost
from catboost import CatBoostClassifier
model = CatBoostClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}', random_state=42)
model.fit(X_train, y_train)
\n """
                        from catboost import CatBoostClassifier
                        model = CatBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type, random_state=42)
                        fit_and_store(model, X_train, y_train, col2)

            if problem_type == "Regression":
                 
                if model == "Linear Regression":
                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fit_intercept = st.selectbox("Fit Intercept (Optional)", [True, False], key='normalize')
            
                    with col2:
                        positive = st.selectbox("Positve (Optional)", [True, False], key='positive')
            
                    with col3:
                        copy_x = st.selectbox("Copy X (Optional)", [True, False], key='copy_x')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept={fit_intercept}, positive={positive}, copy_X={copy_x})
model.fit(X_train, y_train)
\n """
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression(fit_intercept=fit_intercept, positive=positive, copy_X=copy_x)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "K-Nearest Neighbors":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.number_input("N Neighbors (Optional)", min_value=1, max_value=100, value=5, step=1, key='n_neighbors')
            
                    with col2:
                        weights = st.selectbox("Weights (Optional)", ["uniform", "distance"], key='weights')
            
                    with col3:
                        algorithm = st.selectbox("Algorithm (Optional)", ["auto", "ball_tree", "kd_tree", "brute"], key='algorithm')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}')
model.fit(X_train, y_train)
\n """
                        from sklearn.neighbors import KNeighborsRegressor
                        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "Support Vector Machine":
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        kernel = st.selectbox("Kernel (Optional)", ["linear", "poly", "rbf", "sigmoid"], key='kernel')
            
                    with col2:
                        degree = st.number_input("Degree (Optional)", min_value=1, max_value=10, value=3, step=1, key='degree')
            
                    with col3:
                        gamma = st.selectbox("Gamma (Optional)", ["scale", "auto"], key='gamma')
                        
                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Support Vector Machine
from sklearn.svm import SVR
model = SVR(kernel='{kernel}', degree={degree}, gamma='{gamma}')
model.fit(X_train, y_train)
\n """
                        from sklearn.svm import SVR
                        model = SVR(kernel=kernel, degree=degree, gamma=gamma)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "Decision Tree":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col2:
                        splitter = st.selectbox("Splitter (Optional)", ["best", "random"], key='splitter')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=2, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Decision Tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(criterion='{criterion}', splitter='{splitter}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
                        from sklearn.tree import DecisionTreeRegressor
                        model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split)
                        fit_and_store(model, X_train, y_train, col2)
                
                if model == "Random Forest":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        criterion = st.selectbox("Criterion (Optional)", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key='criterion')
            
                    with col3:
                        min_samples_split = st.number_input("Min Samples Split (Optional)", min_value=2, max_value=10, value=2, step=1, key='min_samples_split')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators={n_estimators}, criterion='{criterion}', min_samples_split={min_samples_split})
model.fit(X_train, y_train)
\n """
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "XGBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.0001, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        booster = st.selectbox("Booster (Optional)", ["gbtree", "gblinear", "dart"], key='booster')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, booster='{booster}')
model.fit(X_train, y_train)
\n """
                        from xgboost import XGBRegressor
                        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, booster=booster)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "LightGBM":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["gbdt", "dart", "goss"], key='boosting_type')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> LightGBM
from lightgbm import LGBMRegressor
model = LGBMRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')
model.fit(X_train, y_train)
\n """
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type)
                        fit_and_store(model, X_train, y_train, col2)

                if model == "CatBoost":

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("N Estimators (Optional)", min_value=1, max_value=1000, value=100, step=1, key='n_estimators')
            
                    with col2:
                        learning_rate = st.number_input("Learning Rate (Optional)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, key='learning_rate')
            
                    with col3:
                        boosting_type = st.selectbox("Boosting Type (Optional)", ["Ordered", "Plain"], key='boosting_type')

                    col1, col2, col3 = st.columns([1,0.7,1])
                    if col2.button("Train Model", disabled=bool(training_problems)):
                        progress_bar()
            
                        # Train the model
                        st.session_state.all_the_process += f"""
# Model Building --> CatBoost
from catboost import CatBoostRegressor
model = CatBoostRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, boosting_type='{boosting_type}')
model.fit(X_train, y_train)
\n """
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, boosting_type=boosting_type)
                        fit_and_store(model, X_train, y_train, col2)

