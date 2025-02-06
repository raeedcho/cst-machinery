import pandas as pd
import numpy as np

def reindex_by_trial_time(td: pd.DataFrame):
    return (
        td
        .reset_index(level='session_time')
        .groupby('trial',group_keys=False)
        .apply(lambda df: df.assign(trial_time = df['session_time']-df['session_time'].iloc[0]))
        .drop(columns='session_time', level='signal')
        .set_index('trial_time',append=True)
    )

def get_state_transition_times(state_list: pd.Series, timecol: str='session_time'):
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

def reindex_from_event_times(data: pd.DataFrame,event_times: pd.DataFrame, timecol: str='trial_time'):
    new_data = (
        data
        .assign(**{
            'relative time': lambda df: (df.reset_index(level=timecol)[timecol]-event_times).values,
        })
        .reset_index(level=timecol,drop=True)
        .set_index('relative time',append=True)
        .swaplevel('relative time',1)
    )

    return new_data

def reindex_from_event(data: pd.DataFrame,event: str, timecol: str='trial_time'):
    event_times = (
        data
        .reset_index(level='state')
        ['state']
        .pipe(get_state_transition_times,timecol=timecol)
        .xs(event,level='new_state')
    )

    return reindex_from_event_times(data,event_times,timecol=timecol)

def get_epoch_data(data: pd.DataFrame,epochs: dict, timecol: str='trial_time'):
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
            .pipe(reindex_from_event,event,timecol=timecol)
            .assign(phase=event)
            .set_index('phase',append=True)
            .loc[(slice(None),event_slice),:]
        )
        for event,event_slice in epochs.items()
    ]
    return (
        pd.concat(epoch_data_list)
        .reset_index(level='relative time')
        .assign(**{
            'relative time': lambda df: df['relative time'] / np.timedelta64(1,'s'),
        })
        .set_index('relative time',append=True)
        .swaplevel('relative time',1)
    )

def get_index_level(df,level=None):
    if level is None:
        level = df.index.names
    return df.reset_index(level=level)[level]

def group_average(td,keys=[]):
    return (
        td
        .stack()
        .groupby(keys+['channel'],observed=True)
        .agg('mean')
        .unstack()
        .dropna(axis=1,how='all')
    )

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
                [val(df) if callable(val) else val for val in assign_dict.values()],
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            )
        )
    )