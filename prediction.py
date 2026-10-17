# prediction.py
# Predict the target for one new row: the user enters the original column values, the recorded steps
# (recipe.py) prepare the row exactly like the training data, and the trained model predicts.

import pandas as pd
import streamlit as st
from utils import new_line, short_list
from recipe import required_inputs, prepare_row, feature_sources, target_steps


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
        else:
            data[col] = pd.Series([value], dtype=schema[col]["dtype"])
    return pd.DataFrame(data)


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
        st.write(f"Enter the values of one new row. They are prepared with exactly the same steps as the training data "
                 f"(using the values learned from it), then the trained model predicts **`{target}`**.")

        # One field per original column the model needs; every field starts empty and must be filled
        values = {}
        columns = st.columns(2)
        for i, col in enumerate(needed):
            info, key = schema[col], f"predict_input_{col}"
            with columns[i % 2]:
                if info["kind"] in ("category", "bool"):
                    values[col] = st.selectbox(str(col), info["options"], index=None, placeholder="Choose…", key=key)
                elif info["kind"] == "int":
                    values[col] = st.number_input(str(col), value=None, step=1, format="%d", placeholder="Enter a whole number", key=key)
                else:
                    values[col] = st.number_input(str(col), value=None, format="%.4f", placeholder="Enter a number", key=key)

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

        # Show how the row is prepared, so the process is transparent
        if recipe:
            st.caption("Steps applied to the new row, in order: "
                       + " → ".join(step["label"] for step in recipe if step["op"] != "rows")
                       + (" (row cleaning steps such as deleting rows apply to the training data only)." if any(s["op"] == "rows" for s in recipe) else "."))
