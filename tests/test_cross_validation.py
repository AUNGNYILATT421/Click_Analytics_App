# Cross-validation of the trained model: retrains the best model FLAML found on parts of the training set
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from conftest import ROOT, errors, new_app, split, train


def run_cv(at):
    at.button(key="cv_apply").click().run()
    assert not errors(at)
    tables = [d.value for d in at.dataframe if "Part" in d.value.columns]
    assert tables, [i.value for i in at.info]
    return tables[-1]


def test_cross_validation_in_the_app(trained_titanic):
    at, _ = trained_titanic
    table = run_cv(at)
    scores = table.iloc[:5, 1].astype(float)
    assert list(table["Part"]) == ["1", "2", "3", "4", "5", "Mean", "Spread (std)"]
    assert ((scores > 0.5) & (scores <= 1)).all()
    assert np.isclose(float(table.iloc[5, 1]), scores.mean(), atol=1e-4)


def test_cross_validation_with_preprocessing_balanced_classes_and_catboost():
    # Training-time filling (a Pipeline), balanced class weights, and CatBoost (which writes files unless told not to)
    data = load_breast_cancer(as_frame=True).frame
    data = pd.concat([data[data.target == 1].head(300), data[data.target == 0].head(40)]).reset_index(drop=True)   # 12% minority
    data.loc[data.index[:20], "mean texture"] = np.nan
    before = set(os.listdir(ROOT))
    at = new_app(data)
    split(at, "target")
    train(at, "Classification", impute="Median", models=["CatBoost"], balance=True)
    assert at.session_state["trained_model"] is not None, errors(at)
    run_cv(at)
    assert set(os.listdir(ROOT)) - before == set(), "files were written to the app folder"


def test_cross_validation_for_regression_reports_errors_as_positive_numbers():
    at = new_app(load_diabetes(as_frame=True).frame)
    split(at, "target")
    at.selectbox(key="problem_type").select("Regression").run()
    at.selectbox(key="automl_metric_Regression").select("Root Mean Squared Error (RMSE)").run()
    train(at, "Regression")
    table = run_cv(at)
    assert (table.iloc[:5, 1].astype(float) > 0).all()     # RMSE, not a negated score
