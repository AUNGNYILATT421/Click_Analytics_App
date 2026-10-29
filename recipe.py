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
        # New values may not be negative when the loaded data had none (e.g. tenure, charges, age)
        if not pd.api.types.is_bool_dtype(s):
            num = pd.to_numeric(s, errors='coerce').dropna()
            schema[col]["non_negative"] = bool(len(num)) and bool((num >= 0).all())
    return schema


def input_kind(col, schema, recipe):
    # The kind of value to ask for: a column stored as text but converted to numbers before training
    # (e.g. Telco's TotalCharges, which has blank values) is entered as a number, then converted the same way
    for step in recipe:
        if step["op"] == "convert" and col in step["columns"]:
            return {"Integer": "int", "Float": "float"}.get(step["to"], schema[col]["kind"])
    return schema[col]["kind"]


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


def _present(rows, cols):
    return [c for c in cols if c in rows.columns]


def _drop_bad(rows, bad, message, problems):
    # Remove rows that can't be prepared, remembering why (the first problem of each row is kept)
    for i in rows.index[bad]:
        problems.setdefault(i, message(i))
    return rows[~bad]


def _convert_rows(s, to):
    # Vectorised version of eda_module.convert_column: returns (converted values, mask of rows that can't be converted)
    if to in ("Integer", "Float"):
        num = pd.to_numeric(s, errors='coerce')
        bad = num.isna() | (num % 1 != 0) if to == "Integer" else num.isna() & s.notna()
        return (num.where(~bad, 0).astype("int64") if to == "Integer" else num.astype(float)), bad
    if to == "Text":
        return s.astype(object).where(s.isna(), s.astype(str)), pd.Series(False, index=s.index)
    if to == "Boolean":
        mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False, "1.0": True, "0.0": False}
        mapped = s.astype(str).str.strip().str.lower().map(mapping)
        return mapped.fillna(False).astype(bool), mapped.isna()
    if to == "Datetime":
        dates = pd.to_datetime(s, errors='coerce', format='mixed')
        return dates, dates.isna()
    raise StepProblem(f"Unknown conversion: {to}")


def apply_step(rows, step, problems):
    # Apply one recorded step to new rows, with the values learned from the training data.
    # Rows that can't be prepared are removed and their reason is stored in `problems` (row index -> message).
    op = step["op"]
    if op in ("rows", "ffill_bfill", "blank_to_missing"):
        return rows     # row cleaning applies to the training data only; complete new rows have no blanks to fill
    if op == "drop_columns":
        return rows.drop(columns=_present(rows, step["columns"]))
    if op == "select_columns":
        return rows[_present(rows, step["columns"])]
    if op == "convert":
        for col in _present(rows, step["columns"]):
            original = rows[col]
            result, bad = _convert_rows(original, step["to"])
            rows = rows.assign(**{col: result})
            rows = _drop_bad(rows, bad, lambda i, c=col, v=original: f"**`{c}`** = `{v[i]}` can't be converted to {step['to']}.", problems)
        return rows
    if op == "fill":
        values = {col: v for col, v in step["values"].items() if col in rows.columns}
        return rows.fillna(values)
    if op in ("ordinal", "count_frequency"):
        for col in _present(rows, step["mappings"]):
            mapping, s = step["mappings"][col], rows[col]
            bad = ~s.isin(list(mapping))
            rows = _drop_bad(rows, bad, lambda i, c=col, v=s: f"**`{c}`** = `{v[i]}` wasn't in the training data, so the model can't use it.", problems)
            rows = rows.assign(**{col: rows[col].map(mapping).astype(float)})
        return rows
    if op == "one_hot":
        new_columns = {}
        for col in _present(rows, step["columns"]):
            categories, dummies = step["columns"][col]
            values = rows[col].astype(str)
            bad = ~values.isin(categories)
            rows = _drop_bad(rows, bad, lambda i, c=col, v=values: f"**`{c}`** = `{v[i]}` wasn't in the training data, so the model can't use it.", problems)
            values = values[rows.index]
            for dummy, category in dummies:
                new_columns[dummy] = values == category
        rows = rows.drop(columns=_present(rows, step["columns"]))
        return pd.concat([rows, pd.DataFrame({k: v[rows.index] for k, v in new_columns.items()}, index=rows.index)], axis=1)
    if op == "scale":
        cols = step["columns"]
        if not _present(rows, cols) or rows.empty:
            return rows
        # The scaler was fitted on all these columns; columns the model doesn't need are filled with 0 and discarded
        frame = pd.DataFrame({c: rows[c] if c in rows.columns else 0.0 for c in cols}, index=rows.index)
        scaled = step["scaler"].transform(frame)
        for i, c in enumerate(cols):
            if c in rows.columns:
                rows[c] = scaled[:, i]
        return rows
    if op == "transform":
        cols = _present(rows, step["columns"])
        x = rows[cols].astype(float)
        rows[cols] = {"Log Transformation": np.log1p, "Square Root Transformation": lambda v: np.sign(v) * np.sqrt(np.abs(v)),
                      "Cube Root Transformation": np.cbrt, "Exponential Transformation": np.exp}[step["method"]](x)
        return rows
    if op == "constant":
        cols = _present(rows, step["columns"])
        v = step["value"]
        rows[cols] = {"+": rows[cols] + v, "-": rows[cols] - v, "*": rows[cols] * v, "/": rows[cols] / v}[step["sign"]]
        return rows
    if op == "extract":
        if step["a"] in rows.columns and step["b"] in rows.columns:
            a, b = rows[step["a"]], rows[step["b"]]
            rows[step["name"]] = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[step["sign"]]
        return rows
    raise StepProblem(f"Unknown step: {op}")


def prepare_rows(rows, recipe, model_columns):
    # Replay every recorded step in order on many rows at once.
    # Returns (features for the rows that could be prepared, {row index: problem}, problem for all rows or None)
    problems = {}
    try:
        for step in recipe:
            rows = apply_step(rows, step, problems)
        missing = [c for c in model_columns if c not in rows.columns]
        if missing:
            return None, problems, f"These model features couldn't be built from the inputs: {missing}."
        prepared = rows[list(model_columns)]
        prepared.columns = pd.Index(list(prepared.columns), dtype=object)   # same as the split data (see data_splitting.py)
        return prepared, problems, None
    except StepProblem as e:
        return None, problems, str(e)


def prepare_row(row, recipe, model_columns):
    # One row: returns (features for the model, problem message or None)
    prepared, problems, problem = prepare_rows(row, recipe, model_columns)
    if problem or problems:
        return None, problem or next(iter(problems.values()))
    return prepared, None
