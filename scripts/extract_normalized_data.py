import pandas as pd
import smile_extract
from pathlib import Path
import argparse
from typing import Any
import logging
logger = logging.getLogger(__name__)

def main(args):
    dataset: str = args.dataset
    monkey, session_date = dataset.split('_')

    dataset_folder: Path = Path(args.raw_data_dir) / monkey / session_date
    data_blocks = dataset_folder.glob(f'{monkey}_{session_date}*_sorted.mat')

    output_folder = Path(args.trialframe_dir) / dataset
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) / 'extract'
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / f'{dataset}.log',
        level=args.loglevel,
    )

    smile_data_blocks: dict[str,list[Any]] = {
        file.stem.split('_')[2]: smile_extract.direct_load_smile_data(str(file))
        for file in data_blocks
    }
    logger.info(f'Loaded sorted data from {dataset_folder}')

    save_meta(smile_data_blocks, output_folder / f'{dataset}_meta.parquet')
    save_states(
        smile_data_blocks,
        output_folder / f'{dataset}_states.parquet',
        bin_size=args.bin_size
    )
    save_phasespace(
        smile_data_blocks,
        output_folder / f'{dataset}_hand-pos.parquet',
        final_sampling_rate=1/pd.to_timedelta(args.bin_size).total_seconds(),
    )
    save_binned_spikes(
        smile_data_blocks,
        output_folder / f'{dataset}_neural-spikes-binned.parquet',
        bin_size=args.bin_size,
        min_firing_rate=args.min_firing_rate,
        max_spike_coincidence=args.max_spike_coincidence,
        rate_artifact_threshold=args.rate_artifact_threshold,
    )

def save_meta(smile_data_blocks: dict[str, list[Any]], output_path: Path) -> None:
    meta = smile_extract.concat_block_func_results(
        smile_extract.get_smile_meta,
        smile_data_blocks,
    )
    meta.to_parquet(output_path)
    logger.info(f'Saved meta information to {output_path}')

def save_states(smile_data_blocks: dict[str, list[Any]], output_path: Path, **kwargs) -> None:
    state_list = smile_extract.concat_block_trial_func_results(
        smile_extract.get_trial_states,
        smile_data_blocks,
    )
    state_list.to_frame().to_parquet(output_path) # type: ignore
    logger.info(f'Saved trial states to {output_path}')

def save_phasespace(smile_data_blocks: dict[str, list[Any]], output_path: Path, **kwargs) -> None:
    phasespace = smile_extract.concat_block_trial_func_results(
        smile_extract.get_trial_hand_data,
        smile_data_blocks,
    )
    phasespace.to_parquet(output_path)
    logger.info(f'Saved phasespace to {output_path}')

def save_binned_spikes(smile_data_blocks: dict[str, list[Any]], output_path: Path, **kwargs) -> None:
    def spike_processor(block_data: list) -> pd.DataFrame:
        return (
            smile_extract.get_smile_spike_times(block_data, keep_sorted_only=True)
            .pipe(smile_extract.remove_abnormal_firing_units, min_firing_rate=kwargs['min_firing_rate'], rate_artifact_threshold=kwargs['rate_artifact_threshold'])
            .pipe(smile_extract.remove_artifact_trials, rate_artifact_threshold=kwargs['rate_artifact_threshold'])
            .pipe(smile_extract.remove_correlated_units, max_spike_coincidence=kwargs['max_spike_coincidence'])
            .pipe(smile_extract.bin_spikes, bin_size=kwargs['bin_size'])
            .pipe(smile_extract.collapse_channel_unit_index)
        )
    binned_spikes = (
        smile_extract.concat_block_func_results(
            spike_processor,
            smile_data_blocks,
        )
        .dropna(axis='columns', how='any')
    )
    binned_spikes.to_parquet(output_path)
    logger.info(f'Saved binned spikes to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save trial frame from SMILE data')
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name, e.g. $(monkey)_$(session_date)',
        required=True,
    )
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        help='Path to the raw data directory',
        required=True,
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
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
    )
    parser.add_argument(
        '--bin_size',
        type=str,
        help='Bin size',
        default='10ms',
    )
    parser.add_argument(
        '--min_firing_rate',
        type=float,
        help='Minimum firing rate for included neurons',
        default=0.1,
    )
    parser.add_argument(
        '--max_spike_coincidence',
        type=float,
        help='Maximum spike coincidence for included neurons',
        default=0.2,
    )
    parser.add_argument(
        '--rate_artifact_threshold',
        type=float,
        help='Maximum firing rate threshold for neurons to get rid of artifacts',
        default=350,
    )
    # TODO: add arguments somehow for the following:
    # - resampling window (default ('kaiser', 20))
    # - reference location for phasespace
    # - threshold for coincident spike detection
    # - threshold for minimum firing rate

    args,_ = parser.parse_known_args()

    main(args)