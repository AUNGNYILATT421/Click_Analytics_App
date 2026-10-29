# Prediction on new data: recorded steps, one row, files, the downloaded model, and "Why this prediction?"
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from conftest import ROOT, errors, new_app, split, train, delete_columns
from recipe import required_inputs, prepare_row, prepare_rows
from prediction import build_row, build_rows, predict_table, explain_row, typical_values


def parts(at):
    ss = at.session_state
    cols = list(ss["X_test"].columns)
    needed = [c for c in required_inputs(ss["recipe"], list(ss["input_schema"]), [str(c) for c in cols]) if c != "Survived"]
    return ss["trained_model"], ss["X_test"], ss["recipe"], ss["input_schema"], cols, needed


def test_new_rows_are_prepared_exactly_like_the_training_data(trained_titanic):
    at, raw = trained_titanic
    model, X_test, recipe, schema, cols, needed = parts(at)
    complete = [i for i in X_test.index if not raw.loc[i, needed].isna().any()]
    rows, problems = build_rows(raw.loc[complete, needed], needed, schema, recipe)
    prepared, step_problems, problem = prepare_rows(rows, recipe, cols)
    assert not problems and not step_problems and problem is None
    assert np.abs(prepared.astype(float).values - X_test.loc[complete].astype(float).values).max() == 0
    assert (model.predict(prepared) == model.predict(X_test.loc[complete])).all()
    for i in complete[:20]:     # one row at a time gives the same features
        one, _ = prepare_row(build_row(raw.loc[i, needed].to_dict(), schema), recipe, cols)
        assert np.array_equal(one.astype(float).values, prepared.loc[[i]].astype(float).values)


def test_file_rows_that_cant_be_predicted_get_a_reason(trained_titanic):
    at, raw = trained_titanic
    model, X_test, recipe, schema, cols, needed = parts(at)
    good = raw.loc[X_test.index[0], needed].to_dict()
    good["Age"] = 30.0
    rows = [good, dict(good, Embarked="X"), dict(good, Age="abc"), dict(good, Pclass=2.5), dict(good, Pclass=None),
            dict(good, Age=-4), dict(good, Pclass=4)]
    result, problem = predict_table(pd.DataFrame(rows), model, needed, schema, recipe, cols, "Classification")
    expected = [None, "Embarked = X wasn't in the training data", "Age = abc isn't a number", "Pclass = 2.5 isn't a whole number",
                "Pclass is empty", "Age = -4 is negative", "Pclass = 4 wasn't in the training data"]
    assert problem is None
    for want, got in zip(expected, result["Problem"]):
        assert (want is None and pd.isna(got)) or (want in got), (want, got)
    assert result["Prediction"].notna().tolist() == [want is None for want in expected]
    assert list(result.columns[:4]) == ["Prediction", "Probability 0", "Probability 1", "Problem"]


def test_downloaded_model_predicts_raw_data_outside_the_app(trained_titanic, tmp_path):
    at, raw = trained_titanic
    model, X_test, recipe, schema, cols, needed = parts(at)
    bundle = tmp_path / "model.pkl"

    def make_bundle():
        import os
        import streamlit as st
        from model_building import model_bundle, model_bytes
        open(os.environ["BUNDLE"], "wb").write(model_bytes(model_bundle(st.session_state["trained_model"])))

    os.environ["BUNDLE"] = str(bundle)
    bt = AppTest.from_function(make_bundle, default_timeout=120)
    for key in ("trained_model", "X_train", "recipe", "input_schema", "target_variable", "problem_type"):
        bt.session_state[key] = at.session_state[key]
    bt.run()
    assert not errors(bt) and bundle.stat().st_size > 0
    complete = [i for i in X_test.index if not raw.loc[i, needed].isna().any()]
    raw.loc[complete].to_csv(tmp_path / "new.csv", index=False)
    run = subprocess.run([sys.executable, "predict_with_bundle.py", str(bundle), str(tmp_path / "new.csv"), str(tmp_path / "out.csv")],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (pd.read_csv(tmp_path / "out.csv")["Prediction"].values == model.predict(X_test.loc[complete])).all()


def test_why_this_prediction(trained_titanic):
    at, raw = trained_titanic
    model, X_test, recipe, schema, cols, needed = parts(at)
    X_train = at.session_state["X_train"]
    row = X_test.iloc[[0]]
    prediction = model.predict(row)[0]
    effects = explain_row(model, row, X_train, recipe, schema, needed, "Classification", prediction)
    assert set(effects["input"]) == set(needed)
    # A row made of typical values: replacing an input by its typical value changes nothing
    typical = typical_values(X_train)
    plain = pd.DataFrame({c: pd.Series([typical[c]]).astype(X_train[c].dtype) for c in X_train.columns})
    plain.columns = row.columns
    flat = explain_row(model, plain, X_train, recipe, schema, needed, "Classification", model.predict(plain)[0])
    assert np.allclose(flat["effect"], 0)


def test_prediction_form_and_explanation_in_the_app(trained_titanic):
    at, raw = trained_titanic
    model, X_test, recipe, schema, cols, needed = parts(at)
    values = raw.loc[X_test.index[0], needed].to_dict()
    values["Age"] = 30.0
    for col, v in values.items():
        key = f"predict_input_{col}"
        if [s for s in at.selectbox if s.key == key]:
            at.selectbox(key=key).select(v).run()
        else:
            at.number_input(key=key).set_value(v).run()
    at.button(key="predict_apply").click().run()
    assert any("Predicted" in s.value for s in at.success) and not errors(at)
    assert any(m.value == "#### Why this prediction?" for m in at.markdown)


def test_numbers_stored_as_text_are_entered_as_numbers():
    # Telco-style data: TotalCharges is text in the file (blank for new customers), converted to Float in the app
    rng = np.random.default_rng(0)
    n = 800
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = np.where(tenure == 0, " ", (tenure * monthly).round(2).astype(str))
    churn = np.where(rng.random(n) < 0.3 + 0.004 * (72 - tenure), "Yes", "No")
    raw = pd.DataFrame({"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": pd.array(total, dtype="str"), "Churn": churn})
    at = new_app(raw)
    at.checkbox(key="treat_blank").check().run()
    at.multiselect(key="blank_cols").set_value(["TotalCharges"]).run()
    at.button(key="blank_apply").click().run()
    at.checkbox(key="treat_blank").uncheck().run()
    [c for c in at.checkbox if c.label == "Change Data Type"][0].check().run()
    at.multiselect(key="convert_cols").set_value(["TotalCharges"]).run()
    at.selectbox(key="convert_type").select("Float").run()
    at.button(key="convert_apply").click().run()
    at.run()
    m = [m for m in at.multiselect if m.label == "Select Features" and any("Default Feature" in o for o in m.options)][0]
    m.set_value(["TotalCharges"]).run()
    [s for s in at.selectbox if s.label == "Select Missing Values Strategy"][0].select("Fill with Median").run()
    at.button(key="missing_apply").click().run()
    at.multiselect(key="encoding_feat").set_value(["Churn"]).run()
    at.selectbox(key="encoding").select("Ordinal Encoding").run()
    at.button(key="encoding_apply").click().run()
    at.run()
    split(at, "Churn")
    train(at, "Classification")
    boxes = {b.key: b for b in at.number_input}
    assert "predict_input_TotalCharges" in boxes and "predict_input_TotalCharges" not in {s.key for s in at.selectbox}
    assert all(boxes[f"predict_input_{c}"].min == 0 for c in ("tenure", "MonthlyCharges", "TotalCharges"))
    for col, v in {"tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 844.2}.items():
        at.number_input(key=f"predict_input_{col}").set_value(v).run()
    at.button(key="predict_apply").click().run()
    assert any("Predicted" in s.value for s in at.success) and not errors(at)
