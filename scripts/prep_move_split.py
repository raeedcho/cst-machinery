"""
Prep-move subspace partition pipeline.

Fits a CISDirPartition model that decomposes neural population activity into:
  - CIS (condition-invariant signal) subspace
  - Prep-unique subspace
  - Move-unique subspace
  - Shared prep/move subspace

Saves the fitted model, transformed neural subspace parquets,
and logs the subspace overlap index (SOI) between preparatory and movement activity.

Intended as a DVC pipeline stage (see dvc.yaml: prep_move_split).
"""

import numpy as np
import pandas as pd
import cloudpickle
import logging
from pathlib import Path

from trialframe import SoftnormScaler, get_epoch_data
from src import crystal_models, subspace_tools
from src.io import generic_preproc, get_targets, setup_logging, setup_results_dir
from src.targets import get_target_direction
from src.cli import create_default_parser

logger = logging.getLogger(__name__)


def main(args):
    dataset = args.dataset

    setup_logging(args, subfolder_name='prep-move-split')
    results_dir = setup_results_dir(args, subfolder_name='prep-move-split')

    # Load preprocessed data and target directions
    preproc = generic_preproc(args)
    targets = get_targets(args.trialframe_dir, dataset)
    target_dir = get_target_direction(targets)
    logger.info(f'Loaded preprocessed data and target directions for {dataset}')

    # Softnorm and add target direction
    softnormer = SoftnormScaler()
    neural_data = (
        preproc
        .assign(**{'target direction': target_dir})
        .set_index('target direction', append=True)
        ['motor cortex']
        .pipe(softnormer.fit_transform)
    )
    logger.info(f'Softnormed neural data: {neural_data.shape}')

    # Define training epochs
    training_epochs = {
        'prep': ('Go Cue', slice(pd.to_timedelta(args.prep_start), pd.to_timedelta(args.prep_end))),
        'move': ('Move', slice(pd.to_timedelta(args.move_start), pd.to_timedelta(args.move_end))),
    }

    # Fit CISDirPartition model
    partition_model = crystal_models.CISDirPartition(
        n_comps_per_cond=args.n_comps_per_cond,
        reference_task=args.reference_task,
        training_epochs=training_epochs,
        var_cutoff=args.var_cutoff,
        split_transform=True,
    )
    partition_model.fit(neural_data)
    logger.info(f'Fitted CISDirPartition model to {dataset}')

    # Save model
    model_path = results_dir / f'{dataset}_prep-move-partition.pkl'
    with open(model_path, 'wb') as f:
        cloudpickle.dump({
            'softnormer': softnormer,
            'partition_model': partition_model,
        }, f)
    logger.info(f'Saved model to {model_path}')

    # Compute and log subspace overlap index
    train_data = (
        neural_data
        .xs(level='task', key=args.reference_task)
        .pipe(get_epoch_data, epochs=training_epochs)
    )
    unsplit_activity = train_data @ partition_model.unsplit_projmat_
    soi = subspace_tools.subspace_overlap_index(
        unsplit_activity.xs(level='phase', key='prep').values,
        unsplit_activity.xs(level='phase', key='move').values,
    )
    logger.info(f'{dataset} subspace overlap index (prep→move): {soi:.4f}')

    # Transform ALL neural data and save each subspace to its own parquet
    transformed = partition_model.transform_split(neural_data)

    # Reset index to ['block', 'trial_id', 'time'] for composition compatibility
    extra_levels = [n for n in transformed.index.names if n not in ('block', 'trial_id', 'time')]
    transformed = transformed.reset_index(level=extra_levels, drop=True)

    parquet_dir = Path(args.trialframe_dir) / dataset
    
    # Save each subspace to its own parquet file
    for space_name in transformed.columns.get_level_values('space').unique():
        space_data = transformed.xs(space_name, level='space', axis=1)
        parquet_path = parquet_dir / f'{dataset}_neural-partition-{space_name}.parquet'
        space_data.to_parquet(parquet_path)
        logger.info(f'Saved {space_name} subspace parquet ({space_data.shape[1]} components) to {parquet_path}')


if __name__ == '__main__':
    parser = create_default_parser(
        description='Fit prep/move subspace partition (CISDirPartition) model and save results.',
    )
    parser.add_argument(
        '--n_comps_per_cond', type=int, default=15,
        help='Number of components per condition for joint PCA reduction',
    )
    parser.add_argument(
        '--var_cutoff', type=float, default=0.99,
        help='Variance cutoff for DekODec prep/move decomposition',
    )
    parser.add_argument(
        '--reference_task', type=str, default='DCO',
        help='Reference task used for fitting the partition model',
    )
    parser.add_argument(
        '--prep_start', type=str, default='-300ms',
        help='Start of prep epoch relative to Go Cue',
    )
    parser.add_argument(
        '--prep_end', type=str, default='0ms',
        help='End of prep epoch relative to Go Cue',
    )
    parser.add_argument(
        '--move_start', type=str, default='-100ms',
        help='Start of move epoch relative to Move onset',
    )
    parser.add_argument(
        '--move_end', type=str, default='200ms',
        help='End of move epoch relative to Move onset',
    )
    args, _ = parser.parse_known_args()
    main(args)
