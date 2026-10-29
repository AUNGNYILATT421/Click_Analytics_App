# Shared helpers for the tests. The app is driven through Streamlit's AppTest, like a user clicking in the browser.
# Run from the app folder:  python -m pytest

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "app.py")
warnings.simplefilter("ignore")
logging.disable(logging.CRITICAL)


# The app opens ./assets/logo.png, so the tests run from its folder
os.chdir(ROOT)


def titanic_like(n=600, seed=0):
    # Synthetic passenger data with the same kinds of problems as Titanic: IDs, names, text categories, missing values
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5])
    age = rng.normal(30, 12, n).clip(1, 80).round()
    age[rng.random(n) < 0.15] = np.nan
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]).astype(object)
    embarked[:2] = None
    fare = (rng.gamma(2, 15, n) * (4 - pclass)).round(2)
    survived = ((sex == "female") * 0.6 + (pclass == 1) * 0.25 + rng.random(n) * 0.5 > 0.6).astype(int)
    cabin = np.where(rng.random(n) < 0.8, None, [f"C{i}" for i in range(n)]).astype(object)
    return pd.DataFrame({"PassengerId": np.arange(1, n + 1), "Survived": survived, "Pclass": pclass,
                         "Name": [f"Passenger {i}" for i in range(n)], "Sex": sex, "Age": age,
                         "SibSp": rng.integers(0, 4, n), "Parch": rng.integers(0, 3, n), "Fare": fare,
                         "Cabin": cabin, "Embarked": embarked})


def new_app(df=None, timeout=600):
    at = AppTest.from_file(APP, default_timeout=timeout)
    if df is not None:
        at.session_state["df"] = df.copy()
    at.run()
    return at


def checkbox(at, label):
    return [c for c in at.checkbox if c.label == label][0]


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def errors(at):
    return [e.value for e in at.exception]


def delete_columns(at, cols):
    checkbox(at, "Delete Columns").check().run()
    [m for m in at.multiselect if m.label == "Select Columns to Delete"][0].set_value(cols).run()
    [b for b in at.button if b.label == "Delete" and b.key != "delete_rows_apply"][0].click().run()
    checkbox(at, "Delete Columns").uncheck().run()


def split(at, target, sets="Train and Test"):
    at.selectbox(key="target").select(target).run()
    at.selectbox(key="sets").select(sets).run()
    button(at, "Split Data").click().run()
    at.run()


def train(at, problem_type, budget=10, impute=None, models=None, balance=False):
    at.selectbox(key="problem_type").select(problem_type).run()
    at.slider(key="automl_time_budget").set_value(budget).run()
    if impute:
        at.selectbox(key="train_impute").select(impute).run()
    if models:
        at.multiselect(key=f"automl_models_{problem_type}").set_value(models).run()
    if [c for c in at.checkbox if c.key == "allow_duplicates"]:
        at.checkbox(key="allow_duplicates").check().run()
    if balance:
        at.checkbox(key="balance_classes").check().run()
    at.button(key="train_model").click().run()
    at.run()


def prepare_titanic(at):
    # Nine kinds of recorded steps, applied through the user interface
    delete_columns(at, ["PassengerId", "Name"])
    for cols, strategy in ((["Age", "Embarked"], "Fill with Median"), (["Cabin"], "Drop Columns")):
        at.run()
        m = [m for m in at.multiselect if m.label == "Select Features" and any("Default Feature" in o for o in m.options)][0]
        m.set_value(cols).run()
        [s for s in at.selectbox if s.label == "Select Missing Values Strategy"][0].select(strategy).run()
        at.button(key="missing_apply").click().run()
    checkbox(at, "Change Data Type").check().run()
    at.multiselect(key="convert_cols").set_value(["Pclass"]).run()
    at.selectbox(key="convert_type").select("Text").run()
    at.button(key="convert_apply").click().run()
    checkbox(at, "Change Data Type").uncheck().run()
    for cols, method in ((["Sex", "Embarked"], "One Hot Encoding"), (["Pclass"], "Ordinal Encoding")):
        at.multiselect(key="encoding_feat").set_value(cols).run()
        at.selectbox(key="encoding").select(method).run()
        at.button(key="encoding_apply").click().run()
        at.run()
    scale = [m for m in at.multiselect if m.label == "Select Features" and m.key not in ("encoding_feat", "transformation features")
             and "Age" in m.options and not any("Default Feature" in o for o in m.options)][0]
    scale.set_value(["Age"]).run()
    [s for s in at.selectbox if s.label == "Select Scaling"][0].select("MinMax Scaling").run()
    at.button(key="scaling_apply").click().run()
    at.multiselect(key="transformation features").set_value(["Fare"]).run()
    at.selectbox(key="transformation").select("Log Transformation").run()
    at.button(key="trans_apply").click().run()
    at.selectbox(key="feat_ex1").select("SibSp").run()
    at.selectbox(key="feat_ex_op").select("Addition +").run()
    at.selectbox(key="feat_ex2").select("Parch").run()
    at.text_input(key="feat_name").input("family").run()
    button(at, "Extract Feature").click().run()
    at.run()


@pytest.fixture(scope="session")
def trained_titanic():
    # One prepared and trained classification app, shared by several tests (training once keeps the suite fast)
    raw = titanic_like()
    at = new_app(raw)
    prepare_titanic(at)
    split(at, "Survived")
    train(at, "Classification")
    assert at.session_state["trained_model"] is not None, errors(at)
    return at, raw
