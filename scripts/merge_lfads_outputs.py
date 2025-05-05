import argparse
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import pandas as pd
import numpy as np
from src.chop_merge import chops_to_frame
from src.munge import get_index_level

from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import h5py

def main(args):
    dataset: str = args.dataset
    monkey, session_date = dataset.split('_')

    trialframe_folder: Path = Path(args.trialframe_dir) / dataset
    trialframe_path: Path = trialframe_folder / f'{dataset}_neural-spikes-binned.parquet'
    lfads_tensors_path: Path = Path(args.lfads_dir) / dataset / f'lfads_output_{dataset}_tensors.h5'
    log_dir: Path = Path(args.logdir) / 'merge-lfads-outputs'

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / f'{dataset}.log',
        level=args.loglevel,
    )
    
    binned_spikes = pd.read_parquet(trialframe_path)
    logger.info(f'Loaded data from {trialframe_path}')

    with h5py.File(lfads_tensors_path, 'r') as f:
        lfads_dict = {k: v[()] for k, v in f.items()}
    chops = extract_chops(lfads_dict)
    logger.info(f'Extracted chops from {lfads_tensors_path}')
    
    lfads_counts: pd.DataFrame = chops_to_frame(
        chops,
        overlap=args.overlap,
        smooth_pwr=2,
        orig_frame=binned_spikes, # type: ignore
    )
    logger.info(f'Converted chops to frame with shape {lfads_counts.shape}')

    bin_size = (
        get_index_level(lfads_counts, 'time')
        .dt.total_seconds()
        .diff()
        .mode()
        .values[0]
    )
    logger.debug(f'Bin size: {bin_size} seconds')
    
    (
        lfads_counts
        .map(lambda x: x/bin_size)
        .to_parquet(trialframe_folder / f'{dataset}_neural-lfads-rates.parquet')
    )
    logger.info(f'Saved LFADS rates to {trialframe_folder / f"{dataset}_neural-lfads-rates.parquet"}')

def get_split_chops(data_dict: dict, split: str, key: str) -> pd.Series:
    """
    Takes a dictionary of data and a split ('train' or 'test') and returns a DataFrame of the split chops.
    """
    if split not in ['train', 'valid']:
        raise ValueError("split must be 'train' or 'valid'")
    
    return pd.Series(
        [arr for arr in data_dict[f'{split}_{key}']],
        index=pd.MultiIndex.from_arrays(
            [data_dict[f'{split}_trial_id'], data_dict[f'{split}_chop_id']],
            names=['trial_id', 'chop_id']
        )
    )

def extract_chops(data_dict: dict, key: str='output_params') -> pd.Series:
    """
    Takes a dictionary of lfads data and returns a DataFrame of the chops.
    """
    return (
        pd.concat(
            [
                get_split_chops(data_dict, split, key='output_params')
                for split in ['train', 'valid']
            ],
            axis=0,
        )
        .sort_index()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge LFADS outputs into a single dataframe and save to disk')
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name, e.g. $(monkey)_$(session_date)',
        required=True,
    )
    parser.add_argument(
        '--overlap',
        type=int,
        help='Overlap between windows',
        required=True,
    )
    parser.add_argument(
        '--trialframe_dir',
        type=str,
        help='Path to parent folder containing trial frame outputs',
        default='data/trialframe/',
    )
    parser.add_argument(
        '--lfads_dir',
        type=str,
        help='Path to parent folder containing LFADS outputs',
        default='results/lfads/',
    )
    parser.add_argument(
        '--logdir',
        type=str,
        help='Logging directory',
        default='logs/',
    )
    parser.add_argument(
        '--loglevel',
        type=str,
        help='Logging level',
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )

    args,_ = parser.parse_known_args()
    main(args)