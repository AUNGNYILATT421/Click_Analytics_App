# project_file.py
# Save and open a project: the current data, the column information as loaded, and the recorded preparation
# steps, so work can be continued later (Streamlit sessions are lost when the page is refreshed).
#
# The file is a zip of plain data - the data as Parquet and the steps as JSON - read back with a whitelist.
# Nothing in it can run code, unlike a pickle file, so opening a project from anyone is safe. The trained model
# isn't included (a model can only be stored as a pickle): split and train again after opening.

import io
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

FORMAT = "click-analytics-project"
VERSION = 1
MAX_PROJECT_BYTES = 200 * 1024**2     # same limit as uploaded files
SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}


class ProjectError(Exception):
    pass


def _encode(v):
    # JSON-safe, typed encoding of the values found in the recorded steps and the column information
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or isinstance(v, (str, int)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return {"$float": repr(float(v))} if not np.isfinite(v) else float(v)
    if isinstance(v, dict):
        return {"$dict": [[_encode(k), _encode(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return {"$tuple" if isinstance(v, tuple) else "$list": [_encode(x) for x in v]}
    if isinstance(v, (set, frozenset)):
        return {"$set": [_encode(x) for x in sorted(v, key=str)]}
    if isinstance(v, np.ndarray):
        return {"$array": [_encode(x) for x in v.tolist()], "dtype": str(v.dtype)}
    if type(v).__name__ in SCALERS:
        fitted = {k: _encode(x) for k, x in vars(v).items() if k.endswith("_")}
        return {"$scaler": type(v).__name__, "params": _encode(v.get_params()), "fitted": fitted}
    if isinstance(v, pd.Timestamp):
        return {"$timestamp": v.isoformat()}
    try:
        return {"$dtype": str(pd.api.types.pandas_dtype(v))}
    except TypeError:
        raise ProjectError(f"a value of type {type(v).__name__} can't be saved")


def _decode(v):
    if not isinstance(v, dict):
        if isinstance(v, list):
            raise ProjectError("unexpected list")
        return v
    (tag, payload), = [(k, x) for k, x in v.items() if k.startswith("$")]
    if tag == "$float":
        return float(payload)
    if tag == "$dict":
        return {_decode(k): _decode(x) for k, x in payload}
    if tag == "$list":
        return [_decode(x) for x in payload]
    if tag == "$tuple":
        return tuple(_decode(x) for x in payload)
    if tag == "$set":
        return {_decode(x) for x in payload}
    if tag == "$array":
        return np.array([_decode(x) for x in payload], dtype=np.dtype(v["dtype"]))
    if tag == "$timestamp":
        return pd.Timestamp(payload)
    if tag == "$dtype":
        return pd.api.types.pandas_dtype(payload)
    if tag == "$scaler":
        if payload not in SCALERS:
            raise ProjectError(f"unknown scaler {payload}")
        scaler = SCALERS[payload](**_decode(v["params"]))
        for k, x in v["fitted"].items():
            if not k.endswith("_") or k.startswith("_"):
                raise ProjectError("unexpected scaler attribute")
            setattr(scaler, k, _decode(x))
        return scaler
    raise ProjectError(f"unknown value {tag}")


def project_bytes(df, input_schema, recipe):
    meta = {"format": FORMAT, "version": VERSION, "rows": len(df), "columns": len(df.columns),
            "input_schema": _encode(input_schema), "recipe": _encode(recipe)}
    data = io.BytesIO()
    # Column names as text: Parquet requires it (they are restored from the column list below)
    frame = df.copy()
    names = list(frame.columns)
    frame.columns = [str(i) for i in range(len(names))]
    frame.to_parquet(data, index=False)
    meta["column_names"] = _encode(names)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(meta))
        z.writestr("data.parquet", data.getvalue())
    return out.getvalue()


def load_project(file):
    # Returns (df, input_schema, recipe); raises ProjectError with a plain-language reason
    try:
        z = zipfile.ZipFile(file)
    except zipfile.BadZipFile:
        raise ProjectError("this isn't a Click Analytics project file")
    with z:
        names = set(z.namelist())
        if not {"project.json", "data.parquet"} <= names:
            raise ProjectError("this isn't a Click Analytics project file")
        # Check the unpacked sizes before reading, so a small file can't expand to fill the memory
        if sum(z.getinfo(n).file_size for n in ("project.json", "data.parquet")) > MAX_PROJECT_BYTES:
            raise ProjectError("the project is too large to open")
        try:
            meta = json.loads(z.read("project.json"))
        except ValueError:
            raise ProjectError("the project information is damaged")
        if meta.get("format") != FORMAT:
            raise ProjectError("this isn't a Click Analytics project file")
        if meta.get("version") != VERSION:
            raise ProjectError("this project was saved by a different version of the app")
        df = pd.read_parquet(io.BytesIO(z.read("data.parquet")))
    try:
        df.columns = _decode(meta["column_names"])
        return df, _decode(meta["input_schema"]), _decode(meta["recipe"])
    except (ProjectError, KeyError, TypeError, ValueError) as e:
        raise ProjectError(f"the project information is damaged ({e})")
