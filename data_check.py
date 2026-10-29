# data_check.py
# Plain-language warnings about the data that commonly hurt models: identifier columns, constant columns,
# text with too many different values, mostly-empty columns, duplicate rows, features almost identical to
# the target, and imbalanced classes. Only cheap pandas statistics are used, on a sample of large data.

import numpy as np
import pandas as pd
import streamlit as st
from utils import short_list

SAMPLE_ROWS = 100_000      # large data is checked on a random sample
NEAR_LEAK = 0.95           # |correlation| with the target at or above this is suspicious
MINORITY_SHARE = 0.20      # a class with less than this share of the rows is a minority class


def _sample(df):
    return df.sample(SAMPLE_ROWS, random_state=42) if len(df) > SAMPLE_ROWS else df


def find_data_issues(df):
    # Returns a list of (title, columns, advice)
    s = _sample(df)
    n = len(s)
    issues = []
    if n == 0:
        return issues
    nunique = s.nunique(dropna=True)
    text = [c for c in s.columns if not pd.api.types.is_numeric_dtype(s[c]) and not pd.api.types.is_bool_dtype(s[c])]

    ids = []
    for c in s.columns:
        col = s[c].dropna()
        if len(col) < 20 or nunique[c] != len(col):
            continue
        name = str(c).lower().replace("_", " ").replace("-", " ")
        is_counter = pd.api.types.is_integer_dtype(col) and (col.sort_values().diff().dropna() == 1).all()
        if c in text or is_counter or name == "id" or name.endswith(" id") or name.startswith("id "):
            ids.append(c)
    if ids:
        issues.append(("Identifier columns", ids, "Every value is different, like a row number, name or ID. Models can't learn from "
                       "these and may memorise them instead: delete them in **Delete Columns**."))

    constant = [c for c in s.columns if nunique[c] <= 1]
    if constant:
        issues.append(("Constant columns", constant, "Only one value, so they carry no information: delete them in **Delete Columns**."))

    many = [c for c in text if c not in ids and nunique[c] > 50 and nunique[c] / n > 0.5]
    if many:
        issues.append(("Text with many different values", many, "Too many categories to encode usefully (one-hot encoding "
                       "would create a column for each): delete them, or extract something simpler from them first."))

    missing = s.isna().mean()
    mostly_empty = [c for c in s.columns if missing[c] > 0.5]
    if mostly_empty:
        issues.append(("Mostly empty columns", mostly_empty, "More than half of the values are missing: filling them would "
                       "mostly invent data, so consider dropping these columns in **Missing Values**."))

    dup = int(df.duplicated().sum()) if len(df) <= SAMPLE_ROWS else int(s.duplicated().sum())
    if dup:
        issues.append(("Duplicate rows", [], f"{dup:,} rows{' in the sample' if len(df) > SAMPLE_ROWS else ''} are exact copies of another row: "
                       "check them in **Show Duplicate Rows**."))
    return issues


def show_data_check(df):
    # Recomputed only when the data changes (every change records a step), not on every click
    key = (tuple(step["label"] for step in st.session_state.get('recipe') or []), df.shape)
    issues = _cached_issues_for(df, key)
    st.markdown("### 🩺 Data Check")
    with st.expander("Show Data Check", expanded=bool(issues)):
        if not issues:
            st.success("No common problems found: no identifier, constant, mostly empty or duplicate data.")
            return
        st.write("These are common causes of poor or misleading models. They are suggestions: you decide what to change.")
        for title, cols, advice in issues:
            st.warning(f"**{title}**{': ' + short_list([str(c) for c in cols]) if cols else ''}. {advice}")


def _cached_issues_for(df, key):
    # Keep the last result in the session (cheaper than hashing the whole DataFrame for st.cache_data)
    saved = st.session_state.get('data_check')
    if saved is None or saved[0] != key:
        saved = (key, find_data_issues(df))
        st.session_state['data_check'] = saved
    return saved[1]


def near_leaks(df, target):
    # Numeric features almost (but not exactly) identical to the target: often built from it or recorded after it.
    # Features exactly identical (100%) are removed at split time (data_splitting.py). Cached until the data or target changes.
    key = (target, tuple(step["label"] for step in st.session_state.get('recipe') or []), df.shape)
    saved = st.session_state.get('near_leaks')
    if saved and saved[0] == key:
        return saved[1]
    leaks = []
    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        s = _sample(df)
        features = s.drop(columns=[target]).select_dtypes(include=["number", "bool"]).astype(float)
        if not features.empty:
            corr = features.corrwith(s[target].astype(float)).abs()
            leaks = [c for c, v in corr.items() if v >= NEAR_LEAK and not np.isclose(v, 1.0)]
    st.session_state['near_leaks'] = (key, leaks)
    return leaks


def class_balance(y):
    # (minority classes, their share) when some class has less than MINORITY_SHARE of the rows, else ([], None)
    shares = pd.Series(y).value_counts(normalize=True)
    if len(shares) < 2:
        return [], None
    small = shares[shares < MINORITY_SHARE]
    return list(small.index), (float(small.min()) if len(small) else None)
