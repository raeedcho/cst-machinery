import numpy as np
import pandas as pd
import pytest

# Common small synthetic dataset fixtures for repeatable tests

@pytest.fixture
def simple_timeseries_df():
    # Create a simple MultiIndex DataFrame with time level at 10 ms sampling
    trial_ids = [1, 1]
    times = pd.to_timedelta([0.00, 0.01], unit="s")
    index = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}, index=index)
    return df

@pytest.fixture
def longer_timeseries_df():
    # 1 trial, 101 samples at 10 ms -> 1.0 s duration
    n = 101
    trial_ids = [1] * n
    times = pd.to_timedelta(np.arange(n) * 0.01, unit="s")
    index = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    # Linear ramp in each channel
    df = pd.DataFrame({"x": np.linspace(0, 1, n), "y": np.linspace(1, 0, n)}, index=index)
    return df

@pytest.fixture
def small_state_series(longer_timeseries_df):
    # Create a simple state series aligned to longer_timeseries_df
    # First half "Fixation", then switch to "Go Cue" once
    idx = longer_timeseries_df.index
    states = np.where(np.arange(len(idx)) < len(idx) // 2, "Fixation", "Go Cue")
    s = pd.Series(states, index=idx, name="state")
    return s

@pytest.fixture
def simple_points_and_reference():
    # 2D points around (1,1); reference a small cluster
    pts = pd.DataFrame({"a": [1.0, 1.2, 0.8], "b": [1.0, 0.9, 1.1]})
    ref = pd.DataFrame({"a": [1.0, 1.1, 0.9], "b": [1.0, 1.1, 0.9]})
    return pts, ref
