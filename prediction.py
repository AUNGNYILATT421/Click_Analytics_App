# prediction.py
# Predict the target for one new row: the user enters the original column values, the recorded steps
# (recipe.py) prepare the row exactly like the training data, and the trained model predicts.

import pandas as pd
import streamlit as st
from utils import new_line, short_list
from data_loading import load_data
from recipe import required_inputs, prepare_row, prepare_rows, feature_sources, target_steps, input_kind

# Largest file predicted at once: keeps memory use small on a laptop and on Streamlit Community Cloud
MAX_BATCH_ROWS = 100_000


def build_row(values, schema):
    # One-row DataFrame with the same column types as the data had when it was loaded
    data = {}
    for col, value in values.items():
        kind = schema[col]["kind"]
        if kind == "int":
            data[col] = pd.Series([int(value)], dtype="int64")
        elif kind == "float":
            data[col] = pd.Series([float(value)], dtype="float64")
        elif kind == "bool":
            data[col] = pd.Series([bool(value)], dtype="bool")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # A number typed for a column stored as text: keep it as text, the recorded conversion turns it into a number
            data[col] = pd.Series([str(int(value)) if float(value).is_integer() else repr(float(value))], dtype=schema[col]["dtype"])
        else:
            data[col] = pd.Series([value], dtype=schema[col]["dtype"])
    return pd.DataFrame(data)


def build_rows(df, needed, schema, recipe=()):
    # Many rows from an uploaded file, with the column types the data had when it was loaded.
    # Returns (rows that can be prepared, {row index: problem})
    problems, data = {}, {}
    def flag(mask, message):
        for i in df.index[mask]:
            problems.setdefault(i, message(i))
    for col in needed:
        s = df[col]
        empty = s.isna() | s.astype(str).str.strip().eq("")
        flag(empty, lambda i, c=col: f"**`{c}`** is empty.")
        kind = input_kind(col, schema, recipe)
        if kind in ("int", "float"):
            num = pd.to_numeric(s, errors='coerce')
            flag(num.isna() & ~empty, lambda i, c=col: f"**`{c}`** = `{s[i]}` isn't a number.")
            if kind == "int":
                flag(num.notna() & (num % 1 != 0), lambda i, c=col: f"**`{c}`** = `{s[i]}` isn't a whole number.")
            if schema[col].get("non_negative"):
                flag(num < 0, lambda i, c=col: f"**`{c}`** = `{s[i]}` is negative, but the data had no negative values.")
            if schema[col]["kind"] == "category":
                # Stored as text when loaded: keep the text, the recorded conversion turns it into a number
                data[col] = s.astype(object).where(s.isna(), s.astype(str).str.strip()).astype(schema[col]["dtype"])
            elif kind == "int":
                data[col] = num.where(num.notna() & (num % 1 == 0), 0).astype("int64")
            else:
                data[col] = num.astype("float64")
        elif kind == "bool":
            mapped = s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
            flag(mapped.isna() & ~empty, lambda i, c=col: f"**`{c}`** = `{s[i]}` isn't True or False.")
            data[col] = mapped.fillna(False).astype(bool)
        else:
            # A file may read text categories such as "1" as numbers; compare them as text when the training values were text
            if all(isinstance(v, str) for v in schema[col]["options"]):
                s = s.astype(object).where(s.isna(), s.astype(str).str.strip())
            data[col] = s.astype(schema[col]["dtype"]) if schema[col].get("dtype") is not None else s
    rows = pd.DataFrame(data, index=df.index)
    return rows.drop(index=list(problems)), problems


def predict_frame(model, prepared, problem_type):
    # Predictions (and class probabilities) for prepared rows, as columns to add to the user's file
    out = pd.DataFrame({"Prediction": model.predict(prepared)}, index=prepared.index)
    if problem_type == "Classification" and hasattr(model, "predict_proba"):
        probs = model.predict_proba(prepared)
        for j, c in enumerate(model.classes_):
            out[f"Probability {c}"] = probs[:, j].round(4)
    return out


def show_prediction():
    model = st.session_state.get('trained_model')
    X_train = st.session_state.get('X_train')
    schema = st.session_state.get('input_schema')
    if model is None or X_train is None or schema is None:
        return

    target = st.session_state.get('target_variable')
    recipe = st.session_state.get('recipe', [])
    model_columns = list(X_train.columns)
    needed = [c for c in required_inputs(recipe, list(schema), [str(c) for c in model_columns]) if c != target]
    # Features built from the target itself can't be computed for new data (the target is what we predict)
    sources = feature_sources(recipe, list(schema))
    from_target = [str(c) for c in model_columns if target in sources.get(str(c), set())]

    new_line()
    st.markdown("### 🔮 Prediction")
    new_line()
    with st.expander("Predict on New Data"):
        one_tab, file_tab = st.tabs(["One row", "Many rows (file)"])
        with one_tab:
            st.write(f"Enter the values of one new row. They are prepared with exactly the same steps as the training data "
                     f"(using the values learned from it), then the trained model predicts **`{target}`**.")

            # One field per original column the model needs; every field starts empty and must be filled
            values = {}
            columns = st.columns(2)
            for i, col in enumerate(needed):
                info, key, kind = schema[col], f"predict_input_{col}", input_kind(col, schema, recipe)
                # Columns that had no negative values in the data don't accept negative numbers
                low = info.get("non_negative")
                with columns[i % 2]:
                    if kind in ("category", "bool"):
                        values[col] = st.selectbox(str(col), info["options"], index=None, placeholder="Choose…", key=key)
                    elif kind == "int":
                        values[col] = st.number_input(str(col), value=None, step=1, format="%d", min_value=0 if low else None,
                                                      placeholder="Enter a whole number" + (" (0 or more)" if low else ""), key=key)
                    else:
                        values[col] = st.number_input(str(col), value=None, format="%.4f", min_value=0.0 if low else None,
                                                      placeholder="Enter a number" + (" (0 or more)" if low else ""), key=key)

            empty = [col for col, value in values.items() if value is None]
            prepared, problem = None, None
            if from_target:
                problem = (f"{'the feature' if len(from_target) == 1 else 'the features'} {short_list(from_target)} "
                           f"{'is' if len(from_target) == 1 else 'are'} built from the target **`{target}`** itself, which isn't known for new data. "
                           "This also means the model saw part of the answer during training: remove these features, then split and train again.")
                st.warning(f"This row can't be predicted: {problem}")
            elif empty:
                st.info(f"Please fill in every field. Still empty: {short_list(empty)}.")
            else:
                prepared, problem = prepare_row(build_row(values, schema), recipe, model_columns)
                if problem:
                    st.warning(f"This row can't be predicted: {problem}")

            col1, col2, col3 = st.columns([1, 0.7, 1])
            if col2.button("Predict", use_container_width=True, key="predict_apply", disabled=bool(empty or problem)):
                prediction = model.predict(prepared)[0]
                if st.session_state.get('problem_type') == "Classification":
                    st.success(f"Predicted **`{target}`**: **{prediction}**")
                    probabilities = pd.DataFrame({"Class": [str(c) for c in model.classes_],
                                                  "Probability": model.predict_proba(prepared)[0].round(4)})
                    st.dataframe(probabilities.sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)
                else:
                    st.success(f"Predicted **`{target}`**: **{float(prediction):,.4f}**")
                changed = target_steps(recipe, target)
                if changed:
                    st.info(f"Note: **`{target}`** itself was changed before training ({'; '.join(changed)}), "
                            "so the prediction is on that changed scale, not in the original units.")
                show_explanation(model, prepared, X_train, recipe, schema, needed, prediction)

        with file_tab:
            predict_file(model, needed, schema, recipe, model_columns, target, from_target)

        # Show how the row is prepared, so the process is transparent
        if recipe:
            st.caption("Steps applied to new rows, in order: "
                       + " → ".join(step["label"] for step in recipe if step["op"] != "rows")
                       + (" (row cleaning steps such as deleting rows apply to the training data only)." if any(s["op"] == "rows" for s in recipe) else "."))


def predict_table(df, model, needed, schema, recipe, model_columns, problem_type):
    # The user's rows plus Prediction (and probability) columns, and a Problem column for rows that can't be predicted.
    # Returns (result, problem for all rows or None). Also used outside the app by predict_with_bundle.py.
    rows, problems = build_rows(df, needed, schema, recipe)
    prepared, step_problems, problem = prepare_rows(rows, recipe, model_columns)
    if problem:
        return None, problem
    problems.update(step_problems)
    out = (predict_frame(model, prepared, problem_type) if not prepared.empty else pd.DataFrame({"Prediction": []})).reindex(df.index)
    # Plain text in the file: the messages use Markdown (**`col`**) for the page
    out["Problem"] = pd.Series({i: m.replace("**", "").replace("`", "") for i, m in problems.items()}, dtype=object).reindex(df.index)
    # The results first, then the user's own columns
    return pd.concat([out, df.drop(columns=[c for c in out.columns if c in df.columns])], axis=1), None


def predict_file(model, needed, schema, recipe, model_columns, target, from_target):
    # Predict every row of an uploaded file; rows that can't be predicted get a reason instead of stopping the whole file
    st.write(f"Upload a CSV or Excel file of new rows. It needs the {len(needed)} columns the model uses ({short_list(needed, limit=8)}); other columns are kept but not used. "
             f"Every row is prepared with the same steps as the training data, then the model predicts **`{target}`**. "
             f"Up to {MAX_BATCH_ROWS:,} rows at a time.")
    if from_target:
        st.warning(f"These rows can't be predicted: the model uses {short_list(from_target)}, built from the target **`{target}`** itself. "
                   "Remove these features, then split and train again.")
        return
    uploaded = st.file_uploader("Upload new rows", type=["csv", "xlsx", "xls"], key="predict_file")
    if uploaded is None:
        st.session_state.pop('batch_prediction', None)
        return
    try:
        df = load_data(uploaded).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.warning(f"The file is missing these columns: {short_list(missing, limit=len(missing))}.")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"The file has {len(df):,} rows. Please split it into files of at most {MAX_BATCH_ROWS:,} rows.")
        return

    # Predict only on click; keep the result for this file and this model across reruns (e.g. the download click)
    run_key = (uploaded.file_id, id(model))
    col1, col2, col3 = st.columns([1, 0.7, 1])
    if col2.button("Predict File", use_container_width=True, key="predict_file_apply"):
        result, problem = predict_table(df, model, needed, schema, recipe, model_columns, st.session_state.get('problem_type'))
        if problem:
            st.warning(f"The rows can't be predicted: {problem}")
            return
        st.session_state['batch_prediction'] = {"key": run_key, "result": result}

    saved = st.session_state.get('batch_prediction')
    if saved and saved["key"] == run_key:
        result = saved["result"]
        failed = int(result["Problem"].notna().sum())
        if failed == 0:
            st.success(f"Predicted all {len(result):,} rows." if len(result) != 1 else "Predicted the row.")
        else:
            st.warning(f"Predicted {len(result) - failed:,} of {len(result):,} rows. "
                       f"{failed:,} {'row' if failed == 1 else 'rows'} can't be predicted; the **Problem** column says why.")
        st.dataframe(result.head(100), use_container_width=True)
        if len(result) > 100:
            st.caption(f"Showing the first 100 of {len(result):,} rows. Download the file to see all of them.")
        changed = target_steps(recipe, target)
        if changed:
            st.info(f"Note: **`{target}`** itself was changed before training ({'; '.join(changed)}), "
                    "so the predictions are on that changed scale, not in the original units.")
        col1, col2, col3 = st.columns([1, 0.7, 1])
        col2.download_button("Download Predictions", lambda: result.to_csv(index=False).encode('utf-8'), "predictions.csv",
                             use_container_width=True, key="predict_file_download")


def typical_values(X_train):
    # A typical training value per model feature: the most common value for yes/no and category codes, else the median
    typical = {}
    for c in X_train.columns:
        s = X_train[c]
        few_whole_numbers = pd.api.types.is_numeric_dtype(s) and s.nunique() <= 20 and bool((s.dropna() % 1 == 0).all())
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s) or few_whole_numbers:
            typical[c] = s.mode().iloc[0]
        else:
            typical[c] = s.median()
    return typical


def explain_row(model, prepared, X_train, recipe, schema, needed, problem_type, prediction):
    # For each input the user entered: how much the prediction changes when that input (and every feature built from it)
    # is replaced by a typical training value. One small batch prediction: one row per input.
    sources = feature_sources(recipe, list(schema))
    typical = typical_values(X_train)
    names, rows = [], []
    for col in needed:
        features = [f for f in prepared.columns if col in sources.get(str(f), set())]
        if not features:
            continue
        row = prepared.copy()
        for f in features:
            row[f] = pd.Series([typical[f]], index=row.index).astype(prepared[f].dtype)
        names.append(str(col)); rows.append(row)
    if not rows:
        return None
    alternatives = pd.concat(rows, ignore_index=True)
    if problem_type == "Classification":
        k = list(model.classes_).index(prediction)
        base = model.predict_proba(prepared)[0, k]
        values = model.predict_proba(alternatives)[:, k]
    else:
        base = float(model.predict(prepared)[0])
        values = model.predict(alternatives).astype(float)
    return pd.DataFrame({"input": names, "effect": base - values}).sort_values("effect", key=abs, ascending=False)


def show_explanation(model, prepared, X_train, recipe, schema, needed, prediction):
    import matplotlib.pyplot as plt
    problem_type = st.session_state.get('problem_type')
    effects = explain_row(model, prepared, X_train, recipe, schema, needed, problem_type, prediction)
    if effects is None:
        return
    st.markdown("#### Why this prediction?")
    if problem_type == "Classification":
        st.write(f"How much each input raised or lowered the probability of **{prediction}**, compared with a typical value "
                 "from the training data (the median, or the most common value). Bars in percentage points.")
        effects["effect"] *= 100
        unit = "percentage points"
    else:
        st.write("How much each input raised or lowered the prediction, compared with a typical value from the training data "
                 "(the median, or the most common value).")
        unit = "change in prediction"
    top = effects.head(10).iloc[::-1]     # largest effect at the top
    # Blue raised it, red lowered it (a diverging pair checked for colour-blind readers); the sign is also on each label
    colors = ["#2a78d6" if v >= 0 else "#e34948" for v in top["effect"]]
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.38 * len(top) + 1)))
    ax.barh(top["input"], top["effect"], color=colors, height=0.55)
    ax.axvline(0, color="#52524e", linewidth=0.8)
    span = max(float(top["effect"].abs().max()), 1e-9)
    for y, v in enumerate(top["effect"]):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, f"{v:+.1f}" if problem_type == "Classification" else f"{v:+,.3g}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color="#2b2b29")
    ax.set_xlim(-span * 1.25 if (top["effect"] < 0).any() else -span * 0.05, span * 1.25 if (top["effect"] > 0).any() else span * 0.05)
    ax.set_xlabel(unit, color="#52524e")
    ax.tick_params(colors="#52524e"); ax.grid(axis="x", color="#e5e4e0", linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Blue: raised it · Red: lowered it" + (f" · showing the 10 largest of {len(effects)} inputs" if len(effects) > 10 else "")
               + ". Each input is changed on its own, so the effects don't add up exactly to the prediction.")
