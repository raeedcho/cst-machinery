import pandas as pd
import numpy as np
from .time_slice import slice_by_time, reindex_trial_from_event

def get_index_level(df,level=None):
    if level is None:
        level = df.index.names
    return df.reset_index(level=level)[level]

def multivalue_xs(df: pd.DataFrame,keys: list,level,**kwargs) -> pd.DataFrame:
    return pd.concat([df.xs(key=key,level=level,drop_level=False,**kwargs) for key in keys])

def hierarchical_assign(df,assign_dict):
    '''
    Extends pandas.DataFrame.assign to work with hierarchical columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign to
    assign_dict : dict of pandas.DataFrame or callable
        dictionary of dataframes to assign to df
    '''
    return (
        df
        .join(
            pd.concat(
                [val(df) if callable(val) else val for val in assign_dict.values()], # type: ignore
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            ) # type: ignore
        )
    )

def remove_baseline(
    tf: pd.DataFrame,
    ref_event: str,
    ref_slice: slice,
    timecol: str = 'time',
) -> pd.DataFrame:
    """
    Remove the baseline (found from time slice w.r.t. an event) from the trial frame.

    Parameters
    ----------
    tf : pd.DataFrame
        The trial frame containing the data.
    ref_event : str
        The event to use as a reference for baseline removal.
    ref_slice : slice
        The time slice to use for baseline removal.
    timecol : str, optional
        The name of the time column, by default 'time'.

    Returns
    -------
    pd.DataFrame
        The trial frame with the baseline removed.
    """
    baseline = (
        tf
        .groupby('trial_id',group_keys=False)
        .apply(reindex_trial_from_event,event=ref_event,timecol=timecol) # type: ignore
        .pipe(slice_by_time,time_slice=ref_slice,timecol=timecol)
        .groupby('trial_id')
        .agg(lambda s: np.nanmean(s,axis=0))
    )
    return tf - baseline