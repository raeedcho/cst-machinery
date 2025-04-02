import pandas as pd
import smile_extract
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

def main(args):
    input_path = Path(args.folder)
    output_path = Path(args.out)
    log_dir = Path(args.logdir)

    logging.basicConfig(
        filename=log_dir / 'combine' / f'{output_path.stem}.log',
        level=args.loglevel,
    )
    
    block_names = [file.stem.split('_')[2] for file in input_path.glob('*.parquet')]
    block_frames = [pd.read_parquet(file) for file in input_path.glob('*.parquet')]
    logger.info(f'Loaded blocks from {input_path}')
    logger.debug(f'Block names: {block_names}')

    concat_frame = (
        pd.concat(block_frames)
        .dropna(axis='columns',how='any')
    )
    concat_frame.to_parquet(output_path)
    logger.info(f'Wrote combined session frame to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save trial frame from SMILE data')
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing the data files to combine into one session DataFrame',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Path to the output directory',
        required=True,
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

    args = parser.parse_args()

    main(args)