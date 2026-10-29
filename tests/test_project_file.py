# Save and open a project: the data and the recorded steps come back exactly, and bad files are refused safely
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import new_app, prepare_titanic, titanic_like
from project_file import ProjectError, load_project, project_bytes
from recipe import prepare_rows
from prediction import build_rows


@pytest.fixture(scope="module")
def prepared():
    raw = titanic_like()
    at = new_app(raw)
    prepare_titanic(at)
    ss = at.session_state
    return raw, ss["df"], ss["input_schema"], ss["recipe"]


def test_project_round_trip(prepared):
    raw, df, schema, recipe = prepared
    df2, schema2, recipe2 = load_project(io.BytesIO(project_bytes(df, schema, recipe)))
    pd.testing.assert_frame_equal(df2, df)
    assert schema2 == schema
    assert [s["label"] for s in recipe2] == [s["label"] for s in recipe]
    # The reopened steps prepare new rows exactly like the original ones (including the fitted scaler)
    needed = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    rows = raw.loc[raw[needed].notna().all(axis=1), needed]
    cols = [c for c in df.columns if c != "Survived"]
    a = prepare_rows(build_rows(rows, needed, schema, recipe)[0], recipe, cols)[0]
    b = prepare_rows(build_rows(rows, needed, schema2, recipe2)[0], recipe2, cols)[0]
    assert np.array_equal(a.astype(float).values, b.astype(float).values)


def test_opened_project_continues_in_the_app(prepared):
    raw, df, schema, recipe = prepared
    df2, schema2, recipe2 = load_project(io.BytesIO(project_bytes(df, schema, recipe)))
    at = new_app()
    at.session_state["df"], at.session_state["input_schema"], at.session_state["recipe"] = df2, schema2, recipe2
    at.run()
    assert not at.exception and len(at.session_state["recipe"]) == len(recipe)


def make_zip(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    out.seek(0)
    return out


def test_bad_files_are_refused(prepared):
    raw, df, schema, recipe = prepared
    good = zipfile.ZipFile(io.BytesIO(project_bytes(df, schema, recipe)))
    meta, data = json.loads(good.read("project.json")), good.read("data.parquet")
    with pytest.raises(ProjectError, match="isn't a Click Analytics project"):
        load_project(io.BytesIO(b"not a zip"))
    with pytest.raises(ProjectError, match="isn't a Click Analytics project"):
        load_project(make_zip({"model.pkl": b"anything"}))
    with pytest.raises(ProjectError, match="different version"):
        load_project(make_zip({"project.json": json.dumps(dict(meta, version=99)), "data.parquet": data}))
    # Only whitelisted values can be rebuilt: an unknown object type is refused instead of being created
    evil = json.dumps(meta).replace('"$scaler": "MinMaxScaler"', '"$scaler": "os.system"')
    with pytest.raises(ProjectError, match="damaged"):
        load_project(make_zip({"project.json": evil, "data.parquet": data}))
