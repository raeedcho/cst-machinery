from .munge import get_index_level
from .time_slice import state_list_to_transitions, state_transitions_to_list
import numpy as np
import pandas as pd
from typing import Callable,Union

def get_movement_state_renamer(
        hand_pos: pd.DataFrame,
        start_target_info: pd.DataFrame,
        go_state: str = 'Go Cue'
) -> pd.Series:

    old_states = get_index_level(hand_pos, 'state')
    state_transition_times = state_list_to_transitions(old_states, timecol='time')

    trial_info = (
        state_transition_times
        .reset_index()
        .groupby('trial_id')
        .first()
        .drop(columns=['new_state','time'])
    )

    # get only first element of start target info per trial
    start_target_info = (
        start_target_info
        .groupby('trial_id')
        .last()
    )

    start_targ_radius = start_target_info['radius'].max()
    cursor_radius = 2

    start_exit_time = (
        (hand_pos[['x','y']] - start_target_info[['x','y']])
        .apply(
            lambda row: np.linalg.norm(row), # type: ignore
            axis=1
        ) # type: ignore
        .rename('distance from start target')
        .gt(start_targ_radius + cursor_radius)
        .loc[hand_pos.index]
        .to_frame('out of start target')
        .assign(**{
            'go state': lambda df: get_index_level(df, 'state') == go_state,
            'moving': lambda df: df['out of start target'] & df['go state'],
        })
        .loc[lambda df: df['moving']]
        .pipe(get_index_level, level='time')
        .groupby('trial_id')
        .min()
        .to_frame()
        .assign(new_state='Move')
        .assign(**trial_info)
        .reset_index()
        .set_index(state_transition_times.index.names)
    )

    new_state_transitions: pd.Series = (
        pd.concat([state_transition_times,start_exit_time])
        .groupby('trial_id',group_keys=False)
        .apply(lambda df: df.sort_values(by='time'))
        .squeeze() # type: ignore
    )

    new_states = state_transitions_to_list(new_state_transitions, new_index=old_states.index)

    return new_states

def reassign_state(trialframe: pd.DataFrame, new_state: Union[pd.Series,Callable]) -> pd.DataFrame:
    """
    Rename movement states in the trial frame based on the start target information.

    Parameters
    ----------
    trialframe : pd.DataFrame
        The trial frame containing the states.
    new_state : pd.Series

    Returns
    -------
    pd.DataFrame
        The trial frame with renamed movement states.
    """
    
    return (
        trialframe
        .assign(state=new_state)
        .reset_index(level='state', drop=True)
        .set_index('state', append=True)
    )

def detect_movements(
        hand_pos: pd.DataFrame,
) -> pd.Series:
    """
    Detect movements based on hand velocity and acceleration.
    Steps:
    1) differentiate position to get velocity and acceleration.
    2) find the distribution of vel and acc during the hold period.
    3) use mahalanobis distance from hold period distribution to detect movements.
    4) return a boolean Series indicating whether a movement is detected.

    Parameters
    ----------
    hand_pos : pd.DataFrame
        DataFrame containing hand position data with 'x', 'y', and 'z' columns.

    Returns
    -------
    pd.Series
        Series indicating whether a movement is detected.
    """
    
    pass