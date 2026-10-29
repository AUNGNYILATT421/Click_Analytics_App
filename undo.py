# undo.py
# Undo for data preparation: a copy of the data is kept after every recorded step (recipe.py), so the last
# step can be taken back. Copies are made once per step, not on every rerun, and their total size is capped
# so the history fits on a laptop and on Streamlit Community Cloud.

import streamlit as st

MAX_UNDO_STEPS = 5                 # how many steps can be undone in a row
MAX_HISTORY_BYTES = 200 * 1024**2  # total size of the kept copies (200 MB)


def _size(df):
    return int(df.memory_usage(deep=True).sum())


def checkpoint():
    # Keep a copy of the data whenever a new step has been recorded since the last copy.
    # Called at the start of every run (and before an undo), so a step applied in the previous run is captured.
    ss = st.session_state
    df, recipe = ss.get('df'), ss.get('recipe')
    if df is None or recipe is None:
        return
    history = ss.setdefault('undo_history', [])
    if history and history[-1]["steps"] == len(recipe):
        return
    size = _size(df)
    if size > MAX_HISTORY_BYTES:
        # Too large to copy: nothing before this point can be restored
        history.clear()
        history.append({"steps": len(recipe), "df": None, "recipe": list(recipe), "bytes": 0})
        return
    history.append({"steps": len(recipe), "df": df.copy(), "recipe": list(recipe), "bytes": size})
    # Forget the oldest copies beyond the step limit or the size budget (the current state is always kept)
    while len(history) > MAX_UNDO_STEPS + 1 or sum(h["bytes"] for h in history) > MAX_HISTORY_BYTES:
        history.pop(0)


def can_undo():
    history = st.session_state.get('undo_history', [])
    return len(history) >= 2 and history[-2]["df"] is not None


def undo_last_step():
    # Button callback: runs before the page, so every section sees the restored data
    checkpoint()
    if not can_undo():
        return
    ss = st.session_state
    history = ss['undo_history']
    undone = [step["label"] for step in history[-1]["recipe"][len(history[-2]["recipe"]):]]
    history.pop()
    # Work on a copy, so later in-place changes don't alter the kept copy
    ss['df'] = history[-1]["df"].copy()
    ss['recipe'] = list(history[-1]["recipe"])
    ss['undo_message'] = "Undone: " + "; ".join(undone)


def show_steps_panel():
    # Sidebar: the steps applied to the data so far, with Undo
    ss = st.session_state
    recipe = ss.get('recipe') or []
    with st.sidebar:
        st.markdown("### 🧾 Steps Applied")
        message = ss.pop('undo_message', None)
        if message:
            st.success(message)
        if not recipe:
            st.caption("No changes yet. Every change you make to the data is listed here and can be undone.")
            return
        shown = recipe[-10:]
        if len(recipe) > len(shown):
            st.caption(f"… {len(recipe) - len(shown)} earlier steps")
        st.markdown("\n".join(f"{i}. {step['label']}" for i, step in enumerate(shown, start=len(recipe) - len(shown) + 1)))
        history = ss.get('undo_history', [])
        if can_undo():
            st.button("↩️ Undo Last Step", on_click=undo_last_step, use_container_width=True, key="undo_last_step",
                      help=f"Restores the data as it was before the last step. Up to {MAX_UNDO_STEPS} steps can be undone in a row.")
        elif history and history[-1]["df"] is None:
            st.caption("Undo isn't available: the data is too large to keep copies of it.")
        else:
            st.caption("Nothing more to undo.")
