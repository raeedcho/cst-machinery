import yaml
import argparse
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import pandas as pd
import numpy as np
from src.chop_merge import frame_to_chops
from trialframe import get_index_level

from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import h5py

def main(args):
    input_path = Path(args.path)
    output_path = Path(args.out)
    log_dir = Path(args.log_dir) / 'prep-lfads-tensors'

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / f'{output_path.stem}.log',
        level=args.loglevel,
    )
    
    binned_spikes = pd.read_parquet(input_path)
    logger.info(f'Loaded data from {input_path}')

    chops = frame_to_chops(
        binned_spikes, # type: ignore
        window_len=args.window_len,
        overlap=args.overlap,
    )
    logger.info(f'Chopped data into {len(chops)} segments')
    logger.info(f'Chop shape: {np.stack(chops).shape}') # type: ignore

    save_chops(
        chops,
        output_path,
        group_split=args.group_split,
    )
    logger.info(f'Saved tensors to {output_path}')

    if args.info_path is not None:
        dataset_info = {
            'dataset_name': output_path.name,
            'tensor_shape': {
                'num_neurons': np.stack(chops).shape[2], # type: ignore
                'num_time_bins': np.stack(chops).shape[1], # type: ignore
            }
        }

        info_path = Path(args.info_path)
        if not info_path.exists():
            info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w') as f:
            yaml.dump(dataset_info, f)
        logger.info(f'Saved dataset info to {info_path}')

def save_chops(chops: pd.Series, output_path: Path, group_split: bool = False) -> None:
    """Save the chops to the specified output path.

    Parameters
    ----------
    chops : pd.DataFrame
        The tensors to be saved.
    output_path : Path
        The path to save the tensors to.
    """
    
    if group_split:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
        )
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=0.2,
        )

    train_idx, valid_idx = next(splitter.split(chops, groups=chops.index.get_level_values(0))) # type: ignore
    train_chops = chops.iloc[train_idx]
    valid_chops = chops.iloc[valid_idx]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('train_encod_data', data=np.stack(train_chops.values), compression='gzip')
        f.create_dataset('train_recon_data', data=np.stack(train_chops.values), compression='gzip')
        f.create_dataset('train_trial_id', data=get_index_level(train_chops,'trial_id'), compression='gzip')
        f.create_dataset('train_chop_id', data=get_index_level(train_chops,'chop_id'), compression='gzip')
        f.create_dataset('valid_encod_data', data=np.stack(valid_chops.values), compression='gzip')
        f.create_dataset('valid_recon_data', data=np.stack(valid_chops.values), compression='gzip')
        f.create_dataset('valid_trial_id', data=get_index_level(valid_chops,'trial_id'), compression='gzip')
        f.create_dataset('valid_chop_id', data=get_index_level(valid_chops,'chop_id'), compression='gzip')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save trial frame from SMILE data')
    parser.add_argument(
        'path',
        type=str,
        help='path to the input file',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Path to the output file',
        required=True,
    )
    parser.add_argument(
        '--info_path',
        type=str,
        help='Path to the output dataset info file',
        default=None,
    )
    parser.add_argument(
        '--log_dir',
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
    parser.add_argument(
        '--window_len',
        type=int,
        help='Length of the window to chop the data into',
        default=60,
    )
    parser.add_argument(
        '--overlap',
        type=int,
        help='Overlap between windows',
        default=20,
    )
    parser.add_argument(
        '--group_split',
        action='store_true',
        help='Use group split for train/valid split',
    )

    args, _= parser.parse_known_args()
    main(args)