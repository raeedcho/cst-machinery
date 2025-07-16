import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import smile_extract
from typing import Union
from .munge import (
    get_index_level,
    multivalue_xs,
    hierarchical_assign,
)
from .timeseries import estimate_kinematic_derivative, estimate_kinematic_derivative_savgol
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
        'touchbarcircle': 'start',
        'reachtargettouchbarmg': 'outer',
        'reach': 'outer',
    }
    targets = (
        pd.read_parquet(trialframe_dir / dataset / f'{dataset}_targets.parquet')
        .rename(level='target',index=target_name_mapper)
    )
    return targets

def generic_preproc(args) -> pd.DataFrame:
    """
    Generic preprocessing function for trialframe data
    into a format that works for most of the analyses in this package.
    """
    state_mapper = {
        'Hold Center (CST Cue)': 'Target On',
        'Hold Center (RTT Cue)': 'Target On',
        'Reach Target On': 'Target On',
        'Center Hold 2': 'Memory Delay',
        'Control System': 'Go Cue',
        'Reach to Target 1': 'Go Cue',
        'Hold at Target 1': 'Go Cue', # Sometimes first reach state is skipped in this table (if the first target is in the center)
        'Cheat Period': 'Go Cue', # Period after go cue but when animal has to keep hand still (to avoid predicting go cue in training)
        'Reach Target No Cheat': 'Go Cue',
        'Reach to Target': 'Go Cue',
    }

    start_targets = (
        get_targets(args.trialframe_dir, args.dataset)
        .xs(level='target',key='start')
    )

    preproc = (
        load_trial_frame(args)
        .set_index(['task','result','state'],append=True)
        [['hand position','motor cortex']]
        .pipe(multivalue_xs, level='task', keys=['CST','RTT','DCO'])
        .xs(level='result',key='success')
        .rename(index=state_mapper, level='state')
        .groupby('trial_id').filter(lambda df: np.any(get_index_level(df,'state') == 'Go Cue'))
        # .pipe(hierarchical_assign,{
        #     'hand velocity': lambda df: (
        #         df['hand position']
        #         .groupby('trial_id',group_keys=False)
        #         .apply(estimate_kinematic_derivative_savgol, deriv=1)
        #     ),
        #     'hand acceleration': lambda df: (
        #         df['hand position']
        #         .groupby('trial_id',group_keys=False)
        #         .apply(estimate_kinematic_derivative_savgol, deriv=2)
        #     ),
        # })
        .pipe(hierarchical_assign,{
            'hand velocity': lambda df: (
                df['hand position']
                .groupby('trial_id',group_keys=False)
                .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
            ),
        })
        .pipe(hierarchical_assign,{
            'hand acceleration': lambda df: (
                df['hand velocity']
                .groupby('trial_id',group_keys=False)
                .apply(estimate_kinematic_derivative, deriv=1, cutoff=30)
            ),
        })
        .groupby('state').filter(lambda df: df.name != 'Reach to Center')
        .pipe(reassign_state,new_state=lambda df: get_movement_state_renamer(
            df['hand position'],
            start_targets,
        ))
    )

    return preproc