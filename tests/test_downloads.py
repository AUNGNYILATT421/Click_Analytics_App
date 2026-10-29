# Download buttons build their file only when clicked, and Streamlit runs that code after the page run, where
# st.session_state isn't available. This test runs every download the same way, so a download that reads the
# session state when clicked ("Failed to generate file for download") is caught.
import io
import zipfile

import joblib
import pytest
import streamlit
from streamlit.delta_generator import DeltaGenerator


class NoSessionState:
    def __getattr__(self, name):
        raise RuntimeError("st.session_state isn't available when a download is built")

    def __getitem__(self, key):
        raise RuntimeError("st.session_state isn't available when a download is built")


def test_every_download_builds_its_file_when_clicked(trained_titanic, monkeypatch):
    at, _ = trained_titanic
    builders = {}
    original = DeltaGenerator.download_button

    def recording(self, label, data, *args, **kwargs):
        if callable(data):
            builders[kwargs.get("key") or label] = data
        return original(self, label, data, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", recording)
    at.run()
    assert {"save_df", "save_project", "save_model"} <= set(builders), sorted(builders)

    monkeypatch.setattr(streamlit, "session_state", NoSessionState())
    files = {key: build() for key, build in builders.items()}
    assert all(isinstance(data, bytes) and data for data in files.values())
    assert zipfile.ZipFile(io.BytesIO(files["save_project"])).namelist() == ["project.json", "data.parquet"]
    assert joblib.load(io.BytesIO(files["save_model"]))["format"] == "click-analytics-model"
