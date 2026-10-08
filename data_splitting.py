# data_splitting.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import new_line

def display_splitting_options(df):
    st.markdown("### 🪚 Data Splitting", unsafe_allow_html=True)
    
    target, sets = None, None
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select Target Variable", df.columns.tolist(), key='target', help="Target Variable is the variable that you want to predict.")
        st.session_state['target_variable'] = target
    with col2:
        sets = st.selectbox("Select The Split Sets", ["Select", "Train and Test", "Train, Validation, and Test"], key='sets', help="Train Set is the data used to train the model. Validation Set is the data used to validate the model. Test Set is the data used to test the model.")
        st.session_state['split_sets'] = sets
    
    return target, sets

def remove_target_leaks(df, target):
    # Features 100% correlated with the target give the answer away, so remove them before splitting
    if not pd.api.types.is_numeric_dtype(df[target]):
        return []
    features = df.drop(columns=target).select_dtypes(include=[np.number, 'bool']).astype(float)
    corr = features.corrwith(df[target].astype(float))
    leaks = corr[np.isclose(corr.abs(), 1.0)].index.tolist()
    if leaks:
        st.session_state.all_the_process += f"""
# Remove features 100% correlated with the target
df.drop(columns={leaks}, inplace=True)
\n """
        df.drop(columns=leaks, inplace=True)
        st.session_state['df'] = df
    return leaks

def show_removed_leaks(leaks, target):
    if leaks:
        st.warning(f"The features **`{leaks}`** are 100% correlated with the target **`{target}`**, so they would give the answer away "
                   "and the model wouldn't learn anything useful. They have been removed from the dataset.")

def data_fingerprint(df):
    # Changes whenever any value, column name or dtype changes
    return (tuple(map(str, df.columns)), tuple(map(str, df.dtypes)), int(pd.util.hash_pandas_object(df, index=True).sum()))

def clear_trained_model():
    # A model (and its predictions/metrics) belongs to the split it was trained on
    st.session_state['trained_model'] = None
    st.session_state['trained_model_bool'] = False
    st.session_state['lst_models_predictions'] = []
    st.session_state['models_with_eval'] = {}
    st.session_state['metrics_df'] = pd.DataFrame()
    for key in ('y_pred_train', 'y_pred_val', 'y_pred_test'):
        st.session_state[key] = None

def record_split():
    # Remember which data this split was made from, and drop any model trained on an earlier split
    st.session_state['split_fingerprint'] = data_fingerprint(st.session_state['df'])
    st.session_state['split_discarded'] = False
    clear_trained_model()

def discard_split_if_data_changed():
    # If the data changed after splitting, the split is out of date: discard it so Model Building stays hidden
    if st.session_state.get('X_train') is not None and st.session_state.get('split_fingerprint') != data_fingerprint(st.session_state['df']):
        for key in ('X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'):
            st.session_state[key] = None
        st.session_state['split_fingerprint'] = None
        st.session_state['split_discarded'] = True
        clear_trained_model()

    if st.session_state.get('split_discarded'):
        st.warning("The data has changed since it was split, so the old split and any model trained on it have been discarded. "
                   "Please **split the data again** to continue to model building.")

def train_test_split_ui(df, target, train_size, test_size):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Split Data"):
            leaks = remove_target_leaks(df, target)
            st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
\n """
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            # Clear any validation set left over from an earlier 3-way split
            st.session_state['X_val'] = None
            st.session_state['y_val'] = None
            record_split()
            st.success("Data Splitting Done!")
            show_removed_leaks(leaks, target)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Train")
                st.write(f"X Train Shape: {X_train.shape}")
                st.write(f"y Train Shape: {y_train.shape}")
                train = pd.concat([X_train, y_train], axis=1)
                train_csv = train.to_csv(index=False).encode('utf-8')
                st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key='train2')
            with col2:
                st.write("Test")
                st.write(f"X Test Shape: {X_test.shape}")
                st.write(f"y Test Shape: {y_test.shape}")
                test = pd.concat([X_test, y_test], axis=1)
                test_csv = test.to_csv(index=False).encode('utf-8')
                st.download_button("Download Test Set", test_csv, "test.csv", "text/csv", key='test2')

def train_val_test_split_ui(df, target, train_size, val_size, test_size):
    if float(train_size + val_size + test_size) != 1.0:
        st.error(f"The sum of Train, Validation, and Test sizes must be equal to 1.0, your sum is: **train** + **validation** + **test** = **{train_size}** + **{val_size}** + **{test_size}** = **{sum([train_size, val_size, test_size])}**")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Split Data", use_container_width=True):
                leaks = remove_target_leaks(df, target)
                st.session_state.all_the_process += f"""
# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(df.drop('{target}', axis=1), df['{target}'], train_size={train_size}, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size= {val_size} / (1.0 - {train_size}),random_state=42)
\n """
                X_train, X_rem, y_train, y_rem = train_test_split(df.drop(target, axis=1), df[target], train_size=train_size, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_size / (1.0 - train_size), random_state=42)
                st.session_state['X_train'] = X_train
                st.session_state['X_val'] = X_val
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_val'] = y_val
                st.session_state['y_test'] = y_test
                record_split()
                st.success("Data Splitting Done!")
                show_removed_leaks(leaks, target)
                
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Train")
                    st.write(f"X Train Shape: {X_train.shape}")
                    st.write(f"y Train Shape: {y_train.shape}")
                    train = pd.concat([X_train, y_train], axis=1)
                    train_csv = train.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Train Set", train_csv, "train.csv", "text/csv", key="train3")

                with col2:
                    st.subheader("Valid")
                    st.write(f"X Val Shape: {X_val.shape}")
                    st.write(f"y Val Shape: {y_val.shape}")
                    val = pd.concat([X_val, y_val], axis=1)
                    val_csv = val.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Validation Set", val_csv, "validation.csv", "text/csv", key="val3")

                with col3:
                    st.subheader("Test")
                    st.write(f"X Test Shape: {X_test.shape}")
                    st.write(f"y Test Shape: {y_test.shape}")
                    test = pd.concat([X_test, y_test], axis=1)
                    test_csv = test.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Test Set", test_csv, "test.csv", "text/csv", key="test3")


def split_data(df):
    target, sets = display_splitting_options(df)

    if not (sets and target):
        return

    if sets == "Train and Test":
        col1, col2 = st.columns(2)

        with col1:
            train_size = st.number_input(
                "Train Size",
                min_value=0.05,
                max_value=0.95,
                value=0.7,
                step=0.05,
                key="train_size"
            )

        test_size = 1.0 - train_size

        train_test_split_ui(df, target, train_size, test_size)

    elif sets == "Train, Validation, and Test":
        col1, col2, col3 = st.columns(3)

        with col1:
            train_size = st.number_input(
                "Train Size",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                key="train_size_3"
            )

        # compute allowed remaining after train
        remaining = 1.0 - train_size

        # keep validation within [0.05, remaining-0.05] so test stays >= 0.05
        val_min = 0.05
        val_max = max(val_min, remaining - 0.05)

        # choose a default that's always valid
        val_default = min(0.15, val_max)
        val_default = max(val_min, val_default)

        with col2:
            val_size = st.number_input(
                "Validation Size",
                min_value=val_min,
                max_value=val_max,
                value=val_default,
                step=0.05,
                key="val_size"
            )

        test_size = 1.0 - train_size - val_size

        with col3:
            st.markdown("### Sizes")
            st.write(f"Train: {train_size:.2f}")
            st.write(f"Validation: {val_size:.2f}")
            st.write(f"Test: {test_size:.2f}")

        train_val_test_split_ui(df, target, train_size, val_size, test_size)

