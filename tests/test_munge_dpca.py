import numpy as np
import pandas as pd

from src.munge import make_dpca_tensor, make_dpca_tensor_simple


def test_make_dpca_tensor_shapes():
    # Build minimal trialframe with hierarchical columns (signal, channel)
    idx = pd.MultiIndex.from_product([["A", "B"], [1, 2]], names=["cond1", "cond2"])  # conditions
    cols = pd.MultiIndex.from_product([["signal"], ["c1", "c2"]], names=["signal", "channel"])
    df = pd.DataFrame(np.arange(len(idx) * len(cols)).reshape(len(idx), len(cols)), index=idx, columns=cols)

    tensor = make_dpca_tensor(df, conditions=["cond1", "cond2"])
    assert tensor.ndim == 3  # channel x cond1 x cond2


def test_make_dpca_tensor_simple_shapes():
    idx = pd.MultiIndex.from_product([["A", "B"], [1, 2]], names=["cond1", "cond2"])  # conditions
    cols = pd.MultiIndex.from_product([["signal"], ["c1", "c2"]], names=["signal", "channel"])
    df = pd.DataFrame(np.arange(len(idx) * len(cols)).reshape(len(idx), len(cols)), index=idx, columns=cols)

    tensor = make_dpca_tensor_simple(df, conditions=["cond1", "cond2"])
    assert tensor.ndim == 3
