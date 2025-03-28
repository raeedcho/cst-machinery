import pandas as pd
import numpy as np
import scipy.signal as scs
from . import munge

def get_sample_spacing(df: pd.DataFrame) -> float:
    time_diffs = munge.get_index_level(df, 'time').diff()
    most_common_diff = time_diffs.value_counts().idxmax()
    
    # Ensure most_common_diff is a timedelta
    if isinstance(most_common_diff, pd.Timedelta):
        sample_spacing = most_common_diff.total_seconds()
    else:
        raise ValueError("The most common time difference is not a timedelta object")
    
    return sample_spacing

def estimate_kinematic_derivative(df: pd.DataFrame, deriv: int = 1, cutoff: float = 30)->pd.DataFrame:
    # assert that columns are not a multiindex, so we know we're working with one "signal"
    assert not isinstance(df.columns, pd.MultiIndex), 'must work with only one "signal"'
    assert deriv == 1, 'only first derivative is supported currently'

    sample_spacing = get_sample_spacing(df)
    samprate = 1/sample_spacing
    nyquist = samprate / 2
    normalized_cutoff = cutoff / nyquist
    assert 0 < normalized_cutoff < 1, 'cutoff frequency must be between 0 and Nyquist frequency'
    filt_b, filt_a = scs.butter(4, normalized_cutoff, 'low',output='ba') # type: ignore
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