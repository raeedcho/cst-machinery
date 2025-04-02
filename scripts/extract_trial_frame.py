import pandas as pd
import smile_extract
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

def main(args):
    input_path = Path(args.path)
    output_path = Path(args.out)
    log_dir = Path(args.logdir)

    logging.basicConfig(
        filename=log_dir / 'extract' / f'{output_path.stem}.log',
        level=args.loglevel,
    )

    smile_data = smile_extract.direct_load_smile_data(input_path)
    logger.info(f'Loaded data from {input_path}')

    trial_frame = smile_extract.compose_session_frame(
        smile_data,
        block=args.block,
        bin_size=args.bin_size,
        min_firing_rate=args.min_firing_rate,
        max_spike_coincidence=args.max_spike_coincidence,
        rate_artifact_threshold=args.rate_artifact_threshold,
    )
    logger.info(f'Composed trial frame from {input_path}')

    trial_frame.to_parquet(output_path)
    logger.info(f'Saved trial frame to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save trial frame from SMILE data')
    parser.add_argument(
        'path',
        type=str,
        help='Path to the data file',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Path to the output directory',
        required=True,
    )
    parser.add_argument(
        '--block',
        type=str,
        help='Block name',
        default='',
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
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
    )
    # TODO: add arguments somehow for the following:
    # - resampling window (default ('kaiser', 20))
    # - reference location for phasespace
    # - threshold for coincident spike detection
    # - threshold for minimum firing rate

    args = parser.parse_args()

    main(args)