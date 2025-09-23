import numpy as np
import pandas as pd
import pytest

from src.timeseries import (
    get_sample_spacing,
    estimate_kinematic_derivative,
    estimate_kinematic_derivative_savgol,
    remove_baseline,
    hold_mahal_distance,
)
from src.time_slice import (
    slice_by_time,
    state_list_to_transitions,
    state_transitions_to_list,
    reindex_trial_from_time,
    reindex_trial_from_event,
    get_epoch_data,
)


def test_get_sample_spacing(longer_timeseries_df):
    spacing = get_sample_spacing(longer_timeseries_df)
    assert pytest.approx(spacing, rel=1e-6) == 0.01


def test_estimate_kinematic_derivative_simple(longer_timeseries_df):
    # derivative of linear ramp should be approximately constant
    vel = estimate_kinematic_derivative(longer_timeseries_df[["x"]], deriv=1, cutoff=30)
    assert vel.shape == (len(longer_timeseries_df), 1)
    # mean close to slope 1 / total_time -> since values go 0..1 over 1s, slope ~1.0
    assert np.isfinite(vel.values).all()


def test_estimate_kinematic_derivative_savgol(longer_timeseries_df):
    vel = estimate_kinematic_derivative_savgol(longer_timeseries_df[["x"]], deriv=1)
    assert vel.shape[0] == len(longer_timeseries_df)
    assert np.isfinite(vel.values).all()


def test_slice_by_time(longer_timeseries_df):
    # take first 0.2s
    sl = slice(pd.to_timedelta(0.0, unit="s"), pd.to_timedelta(0.2, unit="s"))
    out = slice_by_time(longer_timeseries_df, sl, timecol="time")
    assert len(out) == 21  # inclusive of 0.0 to 0.2 at 10ms


def test_state_transitions_roundtrip(small_state_series):
    transitions = state_list_to_transitions(small_state_series)
    reconstructed = state_transitions_to_list(transitions, small_state_series.index)
    # The reconstructed should match original states after first label appears
    assert reconstructed.iloc[1:].equals(small_state_series.iloc[1:])


def test_reindex_trial_from_time(longer_timeseries_df):
    shifted = reindex_trial_from_time(longer_timeseries_df, pd.to_timedelta(0.5, unit="s"))
    # time zero should now be at 0.5s original -> there should be a time equal to -0.5
    assert pd.to_timedelta(-0.5, unit="s") in shifted.index.get_level_values("time")


def test_get_epoch_data(longer_timeseries_df):
    # fabricate a state column for epochs; repeat state per row
    df = longer_timeseries_df.copy()
    df["state"] = np.where(
        np.arange(len(df)) < len(df) // 2, "Fixation", "Go Cue"
    )
    df = df.set_index("state", append=True)
    epochs = {"go": ("Go Cue", slice(pd.to_timedelta(0, 's'), pd.to_timedelta(0.1, 's')))}
    out = get_epoch_data(df, epochs)
    assert "phase" in out.index.names
    assert out.index.get_level_values("phase").str.contains("go").any()


def test_remove_baseline(longer_timeseries_df):
    df = longer_timeseries_df.copy()
    df["state"] = ["Fixation"] * (len(df) // 2) + ["Go Cue"] * (len(df) - len(df) // 2)
    df = df.set_index("state", append=True)
    out = remove_baseline(df, ref_event="Fixation", ref_slice=slice(pd.to_timedelta(0,'s'), pd.to_timedelta(0.1,'s')))
    # Baseline subtraction should center early time near zero for first channel
    assert np.isfinite(out.values).all()


def test_hold_mahal_distance(simple_points_and_reference):
    pts, ref = simple_points_and_reference
    dist = hold_mahal_distance(pts, ref)
    assert dist.shape[0] == len(pts)
    assert np.isfinite(dist.values).all()
