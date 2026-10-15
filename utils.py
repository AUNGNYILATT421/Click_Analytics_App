import streamlit as st

def new_line(n=1):
    for i in range(n):
        st.write("\n")

def short_list(cols, limit=5):
    # "`a`, `b`, `c` and 2 more" - keeps messages short when many columns are selected
    shown = ", ".join(f"`{c}`" for c in cols[:limit])
    return shown + (f" and {len(cols) - limit} more" if len(cols) > limit else "")
