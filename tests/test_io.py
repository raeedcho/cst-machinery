import logging
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from src import io as io_mod


def test_get_targets_reads_and_maps(tmp_path):
    dataset = "DS1"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)
    # Build a small dataframe with target level in index
    idx = pd.MultiIndex.from_product([[1, 2], ["center", "reachtarget"]], names=["trial_id", "target"])
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    path = data_dir / f"{dataset}_targets.parquet"
    df.to_parquet(path)

    out = io_mod.get_targets(tmp_path, dataset)
    assert set(out.index.get_level_values("target").unique()) == {"start", "outer"}


def test_get_targets_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_mod.get_targets(tmp_path / "does_not_exist", "ANY")


def test_setup_results_dir_creates(tmp_path):
    args = SimpleNamespace(results_dir=str(tmp_path), dataset="MySet")
    p = io_mod.setup_results_dir(args, subfolder_name="unit")
    assert p.exists() and p.is_dir()


def test_setup_results_dir_idempotent(tmp_path):
    args = SimpleNamespace(results_dir=str(tmp_path), dataset="MySet")
    p1 = io_mod.setup_results_dir(args, subfolder_name="unit")
    p2 = io_mod.setup_results_dir(args, subfolder_name="unit")
    assert p1 == p2 and p1.exists()


def test_setup_logging_configures_basicConfig(tmp_path, monkeypatch):
    args = SimpleNamespace(log_dir=str(tmp_path), dataset="LogSet", loglevel=logging.INFO)
    subfolder = "logs_case"
    captured = {}

    def fake_basicConfig(**kwargs):
        captured.update(kwargs)

    # Patch the function used inside module to avoid global logger state
    monkeypatch.setattr(io_mod.logging, "basicConfig", fake_basicConfig)

    io_mod.setup_logging(args, subfolder_name=subfolder)

    assert "filename" in captured and "level" in captured
    assert captured["level"] == args.loglevel
    assert (tmp_path / subfolder / f"{args.dataset}.log") == captured["filename"]


def test_load_trial_frame_with_mocks(tmp_path, monkeypatch):
    dataset = "DS2"
    trialframe_dir = tmp_path / "trialframe"
    trialframe_dir.mkdir(parents=True)

    # Prepare a fake composition config
    class FakeConf:
        info = "tf"
        composition = {"signals": "signals", "behavior": "behavior"}

    def fake_load(path):  # OmegaConf.load
        return FakeConf()

    # Stubs for read_parquet that are sensitive to filename
    def fake_read_parquet(path):
        name = str(path)
        if name.endswith(f"{dataset}_{FakeConf.info}.parquet"):
            return pd.DataFrame({"meta": [1, 2]})
        if name.endswith(f"{dataset}_signals.parquet"):
            return pd.DataFrame({"sig": [10, 20]})
        if name.endswith(f"{dataset}_behavior.parquet"):
            return pd.DataFrame({"beh": [100, 200]})
        raise AssertionError(f"Unexpected parquet path: {name}")

    # Stub for compose_from_frames
    def fake_compose_from_frames(meta, trialframe_dict):
        # Return a deterministic combined frame indicating it was called correctly
        return pd.DataFrame({"ok": [len(meta), len(trialframe_dict)]})

    # Monkeypatch inside src.io module
    monkeypatch.setattr(io_mod.OmegaConf, "load", fake_load)
    monkeypatch.setattr(io_mod.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(io_mod.smile_extract, "compose_from_frames", fake_compose_from_frames)

    args = SimpleNamespace(
        trialframe_dir=str(trialframe_dir),
        composition_config=str(tmp_path / "comp.yaml"),
        dataset=dataset,
    )

    # Create expected files so fake_read_parquet sees matching suffixes
    # (Content is ignored by our fake function).
    for suffix in [f"{dataset}_{FakeConf.info}.parquet", f"{dataset}_signals.parquet", f"{dataset}_behavior.parquet"]:
        p = trialframe_dir / dataset / suffix
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    out = io_mod.load_trial_frame(args)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["ok"]
    assert out.iloc[0, 0] == 2  # len(meta)=2