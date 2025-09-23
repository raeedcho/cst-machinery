import numpy as np
import pandas as pd

from src.chop_merge import chop_data, merge_chops, frame_to_chops, chops_to_frame


def test_chop_and_merge_roundtrip():
    # Create a simple 2D signal
    T, N = 100, 3
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.normal(size=(T, N)), axis=0).astype("f")

    chops = chop_data(data, overlap=10, window=20)
    assert chops.ndim == 3 and chops.shape[1] == 20 and chops.shape[2] == N

    merged = merge_chops(chops, overlap=10, orig_len=T, smooth_pwr=2)
    assert merged.shape == (T, N)
    # Not identical due to ramping but finite
    assert np.isfinite(merged).all()


def test_frame_to_chops_and_back():
    # Build a tiny DataFrame with one trial
    idx = pd.MultiIndex.from_product([[1], list(np.arange(50))], names=["trial_id", "time"])
    df = pd.DataFrame(np.arange(100).reshape(50, 2), index=idx, columns=["a", "b"])

    chops = frame_to_chops(df, window_len=10, overlap=5)
    assert {n for n in chops.index.names} == {"trial_id", "chop_id"}

    rec = chops_to_frame(chops, orig_frame=df, overlap=5, smooth_pwr=2)
    assert rec.shape == df.shape
