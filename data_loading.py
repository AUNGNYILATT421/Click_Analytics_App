import streamlit as st
import pandas as pd
from urllib.parse import urlparse

def load_data(upd_file):
    # Uploaded files have a .name; URLs are plain strings
    if isinstance(upd_file, str):
        name = urlparse(upd_file).path.lower()
    else:
        name = upd_file.name.lower()

    if name.endswith('.csv'):
        return pd.read_csv(upd_file)
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(upd_file)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")
