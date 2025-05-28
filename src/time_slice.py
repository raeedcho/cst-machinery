import pandas as pd
import numpy as np
from typing import Union
from pandas._libs.tslibs.nattype import NaTType

from .munge import get_index_level

def slice_by_time(data: pd.DataFrame, time_slice: slice, timecol: str='time') -> pd.DataFrame:
    '''
    Slice a DataFrame by a time slice.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame to be sliced.
    slicer : slice
        The slice object defining the time range.
    timecol : str, optional
        The name of the time column in the DataFrame, default is 'time'.

    Returns:
    --------
    pandas.DataFrame
        The sliced DataFrame.
    '''
    
    assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
    assert isinstance(data.index, pd.MultiIndex), "data must have a MultiIndex"
    assert timecol in data.index.names, f"'{timecol}' is not a valid index level in the DataFrame"
    assert isinstance(time_slice, slice), "time_slice must be a slice object"

    num_indices_before_time: int = data.index.names.index(timecol)
    multiindex_slicer: tuple[slice] = num_indices_before_time*(slice(None),) + (time_slice,)
    return data.loc[multiindex_slicer,:]

def get_state_transition_times(state_list: pd.Series, timecol: str='time') -> pd.Series:
    timestep = (
        state_list
        .reset_index(level=timecol)
        [timecol]
        .diff()
        .mode()
        .values[0]
    )
    prev_state = (
        state_list
        .rename(lambda t: t+timestep,level=timecol)
        .reindex(state_list.index)
    )
    state_transition_times = (
        pd.concat(
            [prev_state,state_list],
            keys=['previous_state','new_state'],
            axis=1,
        )
        .loc[
            lambda df: df['previous_state']!=df['new_state']
        ]
        .reset_index(level=timecol)
        .dropna(axis=0,how='any')
        .set_index('new_state',append=True)
        [timecol]
    )
    
    return state_transition_times

from typing import Optional

def reindex_trial_from_time(trial: pd.DataFrame,reference_time: Union[pd.Timedelta,NaTType], timecol: str='time') -> pd.DataFrame:
    return (
        trial
        .rename(
            index= lambda time: time-reference_time,
            level=timecol,
        )
        # .rename_axis(index={'time': 'relative time'})
    )

def reindex_trial_from_event(trial: pd.DataFrame,event: str, timecol: str='time') -> pd.DataFrame:
    try:
        reference_time = (
            trial
            .pipe(get_index_level,level='state')
            .pipe(get_state_transition_times,timecol=timecol)
            .xs(level='new_state',key=event)
            .values[-1]
        )
    except KeyError:
        reference_time = pd.NaT

    return reindex_trial_from_time(trial,reference_time,timecol=timecol)

def get_epoch_data(data: pd.DataFrame,epochs: dict, timecol: str='time'):
    '''
    Get data arranged by epoch and relative time

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to be arranged by epoch and relative time.
    epochs : dict
        The list of epoch names to be used for grouping the data.
        Keys correspond to state name and values are datetime slices
        around the event onset.

    Returns:
    --------
    pandas.DataFrame
        The data arranged by epoch and relative time.
    '''
    epoch_data_list = [
        (
            data
            .groupby('trial_id',group_keys=False)
            .apply(lambda df: reindex_trial_from_event(df,event=event,timecol=timecol))
            .assign(phase=event)
            .set_index('phase',append=True)
            .pipe(slice_by_time, time_slice=event_slice, timecol=timecol)
        )
        for event,event_slice in epochs.items()
    ]
    return pd.concat(epoch_data_list)