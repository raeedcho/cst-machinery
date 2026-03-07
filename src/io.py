import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import smile_extract
from typing import Sequence, Union
from trialframe import (
    get_index_level,
    multivalue_xs,
    hierarchical_assign,
    estimate_kinematic_derivative,
    estimate_kinematic_derivative_savgol,
)
from .states import get_movement_state_renamer, reassign_state

def setup_logging(args, subfolder_name: str='default') -> None:
    log_dir = Path(args.log_dir) / subfolder_name
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(
        filename=log_dir/ f'{args.dataset}.log',
        level=args.loglevel,
    )
    
def load_trial_frame(args) -> pd.DataFrame:
    trialframe_dir = Path(args.trialframe_dir)
    composition_config = OmegaConf.load(args.composition_config)
    tf = smile_extract.compose_from_frames(
        meta=pd.read_parquet(trialframe_dir / args.dataset / f'{args.dataset}_{composition_config.info}.parquet'),
        trialframe_dict={
            key: pd.read_parquet(trialframe_dir / args.dataset / f'{args.dataset}_{filepart}.parquet')
            for key, filepart in composition_config.composition.items()
        }
    )
    return tf

def setup_results_dir(args, subfolder_name: str='default') -> Path:
    results_dir: Path = Path(args.results_dir) / subfolder_name / args.dataset
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir

def get_targets(
        trialframe_dir: Union[str, Path],
        dataset: str,
) -> pd.DataFrame:
    """
    Load and preprocess target data for the specified dataset.
    
    Parameters
    ----------
    trialframe_dir : str
        Directory containing the trialframe data.
    dataset : str, optional
        Name of the dataset to load.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed target data.
    """

    if not isinstance(trialframe_dir, Path):
        trialframe_dir = Path(trialframe_dir)
    if not trialframe_dir.exists():
        raise FileNotFoundError(f"Directory {trialframe_dir} does not exist.")

    target_name_mapper = {
        'center': 'start',
        'starttarget': 'start',
        'touchbarcircle': 'start',
        'reachtargettouchbarmg': 'outer',
        'reachtarget': 'outer',
        'target left right': 'outer',
    }
    targets = (
        pd.read_parquet(trialframe_dir / dataset / f'{dataset}_targets.parquet')
        .rename(level='target',index=target_name_mapper)
    )
    return targets

DEFAULT_STATE_MAPPER = {
    'Center Hold': 'Hold Center (Ambiguous Cue)',
    'Hold Center (CST Cue)': 'Target On',
    'Hold Center (RTT Cue)': 'Target On',
    'Reach Target On': 'Target On',
    'Target Cue': 'Target On',
    'Center Hold 2': 'Memory Delay',
    'Hold Center (Memory)': 'Memory Delay',
    'Memory Period': 'Memory Delay',
    'Control System': 'Go Cue',
    'Reach to Target 1': 'Go Cue',
    'Hold at Target 1': 'Go Cue', # Sometimes first reach state is skipped in this table (if the first target is in the center)
    'Cheat Period': 'Go Cue', # Period after go cue but when animal has to keep hand still (to avoid predicting go cue in training)
    'Reach Target No Cheat': 'Go Cue',
    'No Cheat Window': 'Go Cue',
    'Target Acquire': 'Go Cue',
    'Reach to Target': 'Go Cue',
}


def generic_preproc(
    args,
    composition_config: str | None = None,
    task_keys: Sequence[str] | None = ('CST', 'RTT', 'DCO'),
    result_key: str | None = 'success',
    state_mapper: dict[str, str] | None = None,
    require_go_cue: bool = True,
    add_kinematics: bool = True,
    drop_reach_to_center: bool = True,
    reassign_movement_state_flag: bool = True,
) -> pd.DataFrame:
    """
    Generic preprocessing for composed trialframe data.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments containing at least trialframe and dataset fields.
    composition_config : str, optional
        Path to composition YAML to use for this call. If None, uses
        args.composition_config.
    task_keys : sequence of str, optional
        Task values to keep from task index level. If None, no task filtering.
    result_key : str, optional
        Result value to keep from result index level. If None, no result filter.
    state_mapper : dict, optional
        Mapping applied to state index values. If None, uses
        DEFAULT_STATE_MAPPER.
    require_go_cue : bool, default True
        Keep only trials containing at least one Go Cue state.
    add_kinematics : bool, default True
        Add hand/cursor velocity and acceleration signals.
    drop_reach_to_center : bool, default True
        Drop state entries named Reach to Center.
    reassign_movement_state_flag : bool, default True
        Reassign movement states using hand kinematics and start targets.

    Returns
    -------
    pd.DataFrame
        Preprocessed trialframe with preserved MultiIndex structure.
    """
    selected_state_mapper = DEFAULT_STATE_MAPPER if state_mapper is None else state_mapper

    original_config = args.composition_config
    if composition_config is not None:
        args.composition_config = composition_config

    try:
        preproc = load_trial_frame(args)
    finally:
        args.composition_config = original_config

    if 'state' not in preproc.index.names and 'state' in preproc.columns:
        preproc = preproc.set_index('state', append=True)

    if task_keys is not None and 'task' in preproc.index.names:
        preproc = preproc.pipe(multivalue_xs, level='task', keys=list(task_keys))

    if result_key is not None and 'result' in preproc.index.names:
        preproc = preproc.xs(level='result', key=result_key)

    if selected_state_mapper is not None and 'state' in preproc.index.names:
        preproc = preproc.rename(index=selected_state_mapper, level='state')

    if require_go_cue and 'state' in preproc.index.names:
        preproc = preproc.groupby('trial_id').filter(
            lambda df: np.any(get_index_level(df, 'state') == 'Go Cue')
        )

    if add_kinematics:
        missing_signals = [
            signal for signal in ('hand position', 'cursor position')
            if signal not in preproc.columns.get_level_values(0)
        ]
        if missing_signals:
            raise ValueError(
                f"Cannot add kinematics; missing signals in composition: {missing_signals}"
            )
        preproc = (
            preproc
            .pipe(hierarchical_assign, {
                'hand velocity': lambda df: (
                    df['hand position']
                    .groupby('trial_id', group_keys=False)
                    .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
                ),
            })
            .pipe(hierarchical_assign, {
                'hand acceleration': lambda df: (
                    df['hand velocity']
                    .groupby('trial_id', group_keys=False)
                    .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
                ),
            })
            .pipe(hierarchical_assign, {
                'cursor velocity': lambda df: (
                    df['cursor position']
                    .groupby('trial_id', group_keys=False)
                    .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
                ),
            })
            .pipe(hierarchical_assign, {
                'cursor acceleration': lambda df: (
                    df['cursor velocity']
                    .groupby('trial_id', group_keys=False)
                    .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
                ),
            })
        )

    if drop_reach_to_center and 'state' in preproc.index.names:
        preproc = preproc.groupby('state').filter(lambda df: df.name != 'Reach to Center')

    if reassign_movement_state_flag:
        if 'hand position' not in preproc.columns.get_level_values(0):
            raise ValueError('Cannot reassign movement state; missing hand position signal.')
        start_targets = (
            get_targets(args.trialframe_dir, args.dataset)
            .xs(level='target', key='start')
        )
        preproc = preproc.pipe(
            reassign_state,
            new_state=lambda df: get_movement_state_renamer(
                df['hand position'],
                start_targets,
            ),
        )

    return preproc