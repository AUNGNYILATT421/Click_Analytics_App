# Undo, the data check, the near-leak and imbalance warnings, and the workflow guide (no model training needed)
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from conftest import button, checkbox, delete_columns, errors, new_app, split, titanic_like


def sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown) + " " + " ".join(c.value for c in at.sidebar.caption)


def can_undo(at):
    return any(b.key == "undo_last_step" for b in at.sidebar.button)


def test_undo_restores_the_data_and_steps():
    raw = titanic_like()
    at = new_app(raw)
    delete_columns(at, ["PassengerId", "Name"])
    checkbox(at, "Delete Rows").check().run()
    at.text_input(key="rows_to_delete").input("0, 1, 2").run()
    at.button(key="delete_rows_apply").click().run()     # changes the data in place
    assert len(at.session_state["recipe"]) == 2 and "Deleted columns" in sidebar_text(at)
    at.button(key="undo_last_step").click().run()
    assert len(at.session_state["df"]) == len(raw) and len(at.session_state["recipe"]) == 1
    at.button(key="undo_last_step").click().run()
    assert at.session_state["df"].equals(raw) and at.session_state["recipe"] == [] and not can_undo(at)


def test_at_most_five_steps_can_be_undone():
    at = new_app(titanic_like())
    for col in ["PassengerId", "Name", "Cabin", "SibSp", "Parch", "Fare", "Embarked"]:
        delete_columns(at, [col])
    undone = 0
    while can_undo(at) and undone < 10:
        at.button(key="undo_last_step").click().run()
        undone += 1
    assert undone == 5 and len(at.session_state["recipe"]) == 2


def test_undo_after_a_split_discards_the_split():
    at = new_app(load_breast_cancer(as_frame=True).frame)
    delete_columns(at, ["mean radius"])
    split(at, "target")
    assert at.session_state["X_train"] is not None
    at.button(key="undo_last_step").click().run()
    assert at.session_state["X_train"] is None and any("changed since it was split" in w.value for w in at.warning)


def test_data_check_names_common_problems():
    at = new_app(titanic_like())
    text = " ".join(w.value for w in at.warning)
    assert "Identifier columns" in text and "PassengerId" in text and "Name" in text
    assert "Mostly empty columns" in text and "Cabin" in text
    clean = new_app(load_breast_cancer(as_frame=True).frame)
    assert any("No common problems" in s.value for s in clean.success)


def test_near_leak_is_a_warning_only():
    data = load_breast_cancer(as_frame=True).frame
    data["score"] = data["target"] * 0.97 + np.random.default_rng(1).normal(0, 0.05, len(data))
    at = new_app(data)
    at.selectbox(key="target").select("target").run()
    assert any("score" in w.value and "correlated" in w.value for w in at.warning)
    assert "score" in at.session_state["df"].columns


def test_imbalanced_classes_are_flagged():
    data = load_breast_cancer(as_frame=True).frame
    data = pd.concat([data[data.target == 1], data[data.target == 0].head(30)]).reset_index(drop=True)
    at = new_app(data)
    split(at, "target")
    at.selectbox(key="problem_type").select("Classification").run()
    assert any("Imbalanced classes" in i.value for i in at.info)
    assert any(c.key == "balance_classes" for c in at.checkbox)


def test_workflow_guide_follows_the_data():
    at = new_app(titanic_like())
    guide = [m.value for m in at.sidebar.markdown if m.value.startswith("1. ")][0].split("\n")
    assert "✓ Load data" in guide[0] and "○ Handle missing values" in guide[2]
    assert any("handle missing values" in i.value for i in at.sidebar.info)
    assert not errors(at)
