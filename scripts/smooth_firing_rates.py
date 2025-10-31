import argparse
import logging
from pathlib import Path
import pandas as pd
from trialframe import get_index_level
from smile_extract.smoothing import smooth_data

logger = logging.getLogger(__name__)

def main(args):
    dataset: str = args.dataset

    log_dir: Path = Path(args.log_dir) / 'smooth-firing-rates'
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f'{dataset}.log',
        level=args.loglevel,
    )

    trialframe_folder: Path = Path(args.trialframe_dir) / dataset
    binned_spikes_path: Path = trialframe_folder / f'{dataset}_neural-spikes-binned.parquet'
    binned_spikes: pd.DataFrame = pd.read_parquet(binned_spikes_path)
    logger.info(f'Loaded data from {binned_spikes_path}')

    bin_size = (
        get_index_level(binned_spikes, 'time')
        .dt.total_seconds()
        .diff()
        .mode()
        .values[0]
    )
    logger.debug(f'Bin size: {bin_size} seconds')

    smooth_rates = (
        binned_spikes
        .map(lambda x: x/bin_size)
        .groupby('trial_id')
        .transform(
            smooth_data,
            dt=bin_size,
            std=pd.to_timedelta(args.smoothing_std).total_seconds(),
            backend='convolve',
        )
    )

    smooth_rates.to_parquet(trialframe_folder / f'{dataset}_neural-smooth-rates.parquet')
    logger.info(f'Saved smoothed rates to {trialframe_folder / f"{dataset}_neural-smooth-rates.parquet"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and plot the context axis of CST/RTT data. Output is an SVG plot.')
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name, e.g. $(monkey)_$(session_date)',
        required=True,
    )
    parser.add_argument(
        '--smoothing_std',
        type=str,
        help='Smoothing standard deviation in Timedelta format, e.g. 0.1s, 0.5s, 1s',
        default='0.1s',
    )
    parser.add_argument(
        '--trialframe_dir',
        type=str,
        help='Path to parent folder containing trial frame outputs',
        default='data/trialframe/',
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
    
    args, _ = parser.parse_known_args()
    main(args)