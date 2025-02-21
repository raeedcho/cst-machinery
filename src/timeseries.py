import pandas as pd
import numpy as np
import scipy.signal as scs
from . import munge

def get_sample_spacing(df: pd.DataFrame)->float:
    sample_spacing = (
        munge.get_index_level(df, 'time')
        .diff()
        .value_counts()
        .idxmax()
        .total_seconds()
    )
    return sample_spacing

def estimate_kinematic_derivative(df: pd.DataFrame, deriv: int = 1, cutoff: float = 30):
    # assert that columns are not a multiindex, so we know we're working with one "signal"
    assert not isinstance(df.columns, pd.MultiIndex), 'must work with only one "signal"'
    assert deriv == 1, 'only first derivative is supported currently'

    sample_spacing = get_sample_spacing(df)
    samprate = 1/sample_spacing
    filt_b, filt_a = scs.butter(4, cutoff / (samprate / 2), 'low')
    return (
        df
        .groupby('trial_id', group_keys=False)
        .apply(lambda x: pd.DataFrame(
            scs.filtfilt(filt_b, filt_a, x.values, axis=0),
            columns=x.columns,
            index=x.index
        ))
        .groupby('trial_id', group_keys=False)
        .apply(lambda x: pd.DataFrame(
            np.gradient(x.values, sample_spacing, axis=0),
            columns=x.columns,
            index=x.index
        ))
    )