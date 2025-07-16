from .munge import get_index_level
import numpy as np
import pandas as pd
from typing import Callable,Union

def get_movement_state_renamer(
        hand_pos: pd.DataFrame,
        start_target_info: pd.DataFrame,
        go_state: str = 'Go Cue'
) -> pd.Series:
    start_exit_time = (
        (hand_pos[['x','y']] - start_target_info[['x','y']])
        .apply(
            lambda row: np.linalg.norm(row), # type: ignore
            axis=1
        ) # type: ignore
        .rename('distance from start target')
        .gt(start_target_info['radius'])
        .loc[hand_pos.index]
        .to_frame('out of start target')
        .assign(**{
            'go state': lambda df: get_index_level(df, 'state') == go_state,
            'moving': lambda df: df['out of start target'] & df['go state'],
            'state': lambda df: np.where(
                df['moving'],
                'Move',
                get_index_level(df, 'state'),
            ),
        })
        ['state']
    )
    return start_exit_time

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