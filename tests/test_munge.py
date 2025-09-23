import numpy as np
import pandas as pd

from src.munge import get_index_level, multivalue_xs, hierarchical_assign


def test_get_index_level():
    idx = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=["trial_id", "state"])
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    out = get_index_level(df, level="state")
    vals = np.asarray(getattr(out, "to_numpy", lambda: out)()).ravel()
    assert set(np.unique(vals)) == {"a", "b"}


def test_multivalue_xs_rows():
    idx = pd.MultiIndex.from_product([["CST", "RTT"], [1, 2]], names=["task", "trial_id"])
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=idx)
    sub = multivalue_xs(df, keys=["RTT"], level="task")
    assert (sub.index.get_level_values("task") == "RTT").all()


def test_multivalue_xs_columns():
    cols = pd.MultiIndex.from_product([["signal1", "signal2"], ["a", "b"]], names=["signal", "channel"])
    df = pd.DataFrame(np.arange(8).reshape(2, 4), columns=cols)
    sub = multivalue_xs(df, keys=["signal2"], level="signal", axis=1)
    assert (sub.columns.get_level_values("signal") == "signal2").all()


def test_multivalue_xs_rows_multiple_keys():
    # Ask for both CST and RTT rows
    idx = pd.MultiIndex.from_product([["CST", "RTT", "DCO"], [1, 2]], names=["task", "trial_id"])
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    sub = multivalue_xs(df, keys=["CST", "RTT"], level="task")
    tasks = sub.index.get_level_values("task")
    assert set(tasks.unique()) == {"CST", "RTT"}
    # Order should follow the input frame (CST before RTT)
    assert tasks[:2].tolist() == ["CST", "CST"]


def test_multivalue_xs_columns_multiple_keys():
    # Ask for multiple top-level signals on columns
    cols = pd.MultiIndex.from_product([["pos", "vel", "acc"], ["x", "y"]], names=["signal", "channel"])
    df = pd.DataFrame(np.arange(12).reshape(2, 6), columns=cols)
    sub = multivalue_xs(df, keys=["vel", "acc"], level="signal", axis=1)
    sigs = sub.columns.get_level_values("signal")
    assert set(sigs.unique()) == {"vel", "acc"}
    # Column block order should preserve keys order requested: vel first, then acc
    assert sigs[:2].tolist() == ["vel", "vel"]


def test_multivalue_xs_rows_missing_keys_graceful():
    idx = pd.MultiIndex.from_product([["CST", "RTT"], [1, 2]], names=["task", "trial_id"])
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=idx)
    # Request one existing and one missing key
    sub = multivalue_xs(df, keys=["CST", "FAKE"], level="task")
    tasks = sub.index.get_level_values("task")
    assert set(tasks.unique()) == {"CST"}
    # No error thrown, missing key ignored


def test_multivalue_xs_columns_missing_keys_graceful():
    cols = pd.MultiIndex.from_product([["signal1", "signal2"], ["a", "b"]], names=["signal", "channel"])
    df = pd.DataFrame(np.arange(8).reshape(2, 4), columns=cols)
    # Request one existing and one missing signal
    sub = multivalue_xs(df, keys=["signal2", "signalX"], level="signal", axis=1)
    sigs = sub.columns.get_level_values("signal")
    assert set(sigs.unique()) == {"signal2"}


def test_hierarchical_assign_callable():
    idx = pd.MultiIndex.from_product([[1], [0, 1]], names=["trial_id", "time"])
    df = pd.DataFrame({"x": [0.0, 1.0]}, index=idx)
    out = hierarchical_assign(df, {"y": lambda _df: _df[["x"]] * 2})
    # Access the ('y','x') column created by hierarchical_assign
    assert (out[('y','x')] == out['x'] * 2).all()
