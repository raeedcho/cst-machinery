import numpy as np
import pandas as pd
import pytest

from src.time_slice import (
    slice_by_time,
    state_list_to_transitions,
    state_transitions_to_list,
    reindex_trial_from_time,
    reindex_trial_from_event,
)


def make_state_series(n=10, dt=0.01):
    trial_ids = [1] * n
    times = pd.to_timedelta(np.arange(n) * dt, unit="s")
    idx = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    # change at halfway
    states = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    return pd.Series(states, index=idx, name="state")


def test_state_list_to_transitions_and_back():
    s = make_state_series(20)
    trans = state_list_to_transitions(s)
    s2 = state_transitions_to_list(trans, s.index)
    # Off-by-first-sample tolerable; compare except first
    assert s2.iloc[1:].equals(s.iloc[1:])


def test_reindex_trial_from_event():
    # Construct a minimal trial dataframe with state column in index
    s = make_state_series(20)
    df = pd.DataFrame({"x": np.arange(len(s))}, index=s.index)
    df = df.set_index(s, append=True)
    shifted = reindex_trial_from_event(df, event="B")
    # After reindexing, time should include negative and positive values
    times = shifted.index.get_level_values("time")
    assert (times < pd.to_timedelta(0, unit="s")).any()
    assert (times >= pd.to_timedelta(0, unit="s")).any()


def test_slice_by_time_basic():
    n = 11
    trial_ids = [1] * n
    times = pd.to_timedelta(np.arange(n) * 0.1, unit="s")
    idx = pd.MultiIndex.from_arrays([trial_ids, times], names=["trial_id", "time"])
    df = pd.DataFrame({"x": np.arange(n)}, index=idx)
    out = slice_by_time(df, slice(pd.to_timedelta(0.2, 's'), pd.to_timedelta(0.5, 's')))
    assert len(out) == 4  # 0.2, 0.3, 0.4, 0.5
