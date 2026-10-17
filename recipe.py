# recipe.py
# Records every change applied to the data (and the values learned from it), so a new row of data
# can be prepared exactly the same way before prediction. Nothing is re-learned from the new data.

import numpy as np
import pandas as pd
import streamlit as st


def build_input_schema(df):
    # The columns of the data as it was loaded, and what kind of value each one takes (used for the prediction form)
    schema = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            schema[col] = {"kind": "bool", "options": [True, False]}
        elif pd.api.types.is_integer_dtype(s):
            schema[col] = {"kind": "int"}
        elif pd.api.types.is_numeric_dtype(s):
            schema[col] = {"kind": "float"}
        else:
            values = [v for v in s.dropna().unique() if not (isinstance(v, str) and v.strip() == "")]
            schema[col] = {"kind": "category", "options": sorted(values, key=str), "dtype": s.dtype}
    return schema


def record_step(step):
    # Called by every operation that changes the data, right after it is applied
    st.session_state.setdefault('recipe', []).append(step)


def required_inputs(recipe, original_columns, model_columns):
    # The original columns a new row needs: those the model's features are built from (directly or through other steps)
    sources = feature_sources(recipe, original_columns)
    needed = set().union(*(sources.get(col, set()) for col in model_columns))
    return [col for col in original_columns if col in needed]


def feature_sources(recipe, original_columns):
    # For every column at the end of the recorded steps: the original columns it is built from
    deps = {col: {col} for col in original_columns}
    for step in recipe:
        op = step["op"]
        if op == "drop_columns":
            for col in step["columns"]:
                deps.pop(col, None)
        elif op == "select_columns":
            deps = {col: deps.get(col, set()) for col in step["columns"]}
        elif op == "one_hot":
            for col, (_, dummies) in step["columns"].items():
                source = deps.pop(col, set())
                for dummy, _ in dummies:
                    deps[dummy] = source
        elif op == "extract":
            deps[step["name"]] = deps.get(step["a"], set()) | deps.get(step["b"], set())
    return deps


def target_steps(recipe, target):
    # Recorded steps that changed the target column itself (its predictions are then on that changed scale)
    changing = ("convert", "ordinal", "count_frequency", "scale", "transform", "constant")
    return [step["label"] for step in recipe
            if step["op"] in changing and target in (step.get("columns") or step.get("mappings") or [])]


class StepProblem(Exception):
    pass


def _present(row, cols):
    return [c for c in cols if c in row.columns]


def apply_step(row, step):
    # Apply one recorded step to the new row, with the values learned from the training data
    op = step["op"]
    if op in ("rows", "ffill_bfill", "blank_to_missing"):
        return row      # row cleaning applies to the training data only; a complete new row has no blanks to fill
    if op == "drop_columns":
        return row.drop(columns=_present(row, step["columns"]))
    if op == "select_columns":
        return row[_present(row, step["columns"])]
    if op == "convert":
        from eda_module import convert_column
        for col in _present(row, step["columns"]):
            result, problem = convert_column(row[col], step["to"])
            if problem:
                raise StepProblem(f"**`{col}`** can't be converted to {step['to']}: {problem}.")
            row[col] = result
        return row
    if op == "fill":
        values = {col: v for col, v in step["values"].items() if col in row.columns}
        return row.fillna(values)
    if op in ("ordinal", "count_frequency"):
        for col in _present(row, step["mappings"]):
            mapping, value = step["mappings"][col], row[col].iloc[0]
            if value not in mapping:
                raise StepProblem(f"**`{col}`** = `{value}` wasn't in the training data, so the model can't use it.")
            row[col] = float(mapping[value])
        return row
    if op == "one_hot":
        new_columns = {}
        for col in _present(row, step["columns"]):
            categories, dummies = step["columns"][col]
            value = str(row[col].iloc[0])
            if value not in categories:
                raise StepProblem(f"**`{col}`** = `{value}` wasn't in the training data, so the model can't use it.")
            for dummy, category in dummies:
                new_columns[dummy] = [value == category]
        row = row.drop(columns=_present(row, step["columns"]))
        return pd.concat([row, pd.DataFrame(new_columns, index=row.index)], axis=1)
    if op == "scale":
        cols = step["columns"]
        if not _present(row, cols):
            return row
        # The scaler was fitted on all these columns; columns the model doesn't need are filled with 0 and discarded
        frame = pd.DataFrame({c: row[c] if c in row.columns else 0.0 for c in cols}, index=row.index)
        scaled = step["scaler"].transform(frame)
        for i, c in enumerate(cols):
            if c in row.columns:
                row[c] = scaled[:, i]
        return row
    if op == "transform":
        cols = _present(row, step["columns"])
        x = row[cols].astype(float)
        row[cols] = {"Log Transformation": np.log1p, "Square Root Transformation": lambda v: np.sign(v) * np.sqrt(np.abs(v)),
                     "Cube Root Transformation": np.cbrt, "Exponential Transformation": np.exp}[step["method"]](x)
        return row
    if op == "constant":
        cols = _present(row, step["columns"])
        v = step["value"]
        row[cols] = {"+": row[cols] + v, "-": row[cols] - v, "*": row[cols] * v, "/": row[cols] / v}[step["sign"]]
        return row
    if op == "extract":
        if step["a"] in row.columns and step["b"] in row.columns:
            a, b = row[step["a"]], row[step["b"]]
            row[step["name"]] = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[step["sign"]]
        return row
    raise StepProblem(f"Unknown step: {op}")


def prepare_row(row, recipe, model_columns):
    # Replay every recorded step in order; returns (features for the model, problem message or None)
    try:
        for step in recipe:
            row = apply_step(row, step)
        missing = [c for c in model_columns if c not in row.columns]
        if missing:
            return None, f"These model features couldn't be built from the inputs: {missing}."
        prepared = row[list(model_columns)]
        prepared.columns = pd.Index(list(prepared.columns), dtype=object)   # same as the split data (see data_splitting.py)
        return prepared, None
    except StepProblem as e:
        return None, str(e)
