# guide.py
# Workflow guide in the sidebar: the usual order of the steps, which ones are done, and a suggested next step.
# It only guides; every section stays available on the page.

import pandas as pd
import streamlit as st


def _no_missing(df):
    # Cached until the data changes (every change records a step), so wide data isn't scanned on every click
    key = (tuple(step["label"] for step in st.session_state.get('recipe') or []), df.shape)
    saved = st.session_state.get('guide_no_missing')
    if saved is None or saved[0] != key:
        saved = (key, not df.isna().any().any())
        st.session_state['guide_no_missing'] = saved
    return saved[1]


def guide_steps():
    # [(step, where to find it, done: True / False / None for optional)]
    ss = st.session_state
    df = ss.get('df')
    loaded = df is not None
    trained = bool(ss.get('trained_model_bool')) and ss.get('trained_model') is not None
    metrics = ss.get('metrics_df')
    return [
        ("Load data", "Getting Started", loaded),
        ("Check and explore", "Data Check, Exploratory Data Analysis", None),
        ("Handle missing values", "Missing Values", loaded and _no_missing(df)),
        ("Prepare features", "Encoding, Scaling, Transformation, Feature Engineering",
         loaded and all(pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t) for t in df.dtypes)),
        ("Split the data", "Data Splitting", ss.get('X_train') is not None),
        ("Train a model", "Building the Model", trained),
        ("Evaluate it", "Evaluation", trained and metrics is not None and not metrics.empty),
        ("Predict new data", "Prediction", None),
    ]


def show_guide():
    # Drawn at the end of the run, so steps finished in this run are already ticked
    steps = guide_steps()
    icons = {True: "✓", False: "○", None: "○"}
    with st.sidebar:
        st.markdown("### 🧭 Workflow Guide")
        st.markdown("\n".join(f"{i}. {icons[done]} {name}{' (optional)' if done is None else ''}" for i, (name, _, done) in enumerate(steps, start=1)))
        next_step = next(((name, where) for name, where, done in steps if done is False), None)
        if next_step:
            st.info(f"**Suggested next:** {next_step[0].lower()}, in **{next_step[1]}**.")
        else:
            st.success("All the main steps are done. You can still go back to any section.")
        st.caption("✓ done · ○ to do. A guide only: you can use any section at any time.")
        st.divider()
