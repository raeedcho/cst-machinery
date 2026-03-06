import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import trialframe as tfr
from trialframe import SoftnormScaler
from src.crystal_models import JointSubspace
from src.io import setup_logging, setup_results_dir, load_trial_frame
from src.cli import with_parsed_args, create_default_parser
from src.plot import plot_split_subspace_variance
from sklearn.pipeline import make_pipeline
from dekodec import DekODec
import cloudpickle
from typing import Union,Optional
import logging
logger = logging.getLogger(__name__)

def main(args):
    dataset = args.dataset
    
    setup_logging(args, subfolder_name='subspace-split')
    results_dir = setup_results_dir(args, subfolder_name='subspace-split')
    
    tf = load_trial_frame(args)
    logger.info(f'Loaded trial frame from {dataset}')
    neural_data = precondition_data(tf)

    subspace_split_pipeline = make_pipeline(
        SoftnormScaler(),
        JointSubspace(
            n_comps_per_cond=args.num_components_per_cond,
            condition='task',
            remove_latent_offsets=False,
        ),
        DekODec(
            var_cutoff=args.dekodec_var_cutoff,
            condition='task',
        )
    )

    _ = subspace_split_pipeline.fit_transform(neural_data)
    logger.info(f'Fitted subspace split pipeline to {dataset} data')

    with open(results_dir / f'{dataset}_subspace-split-pipeline.pkl', 'wb') as f:
        cloudpickle.dump(subspace_split_pipeline, f)
    logger.info(f'Saved subspace split pipeline to {results_dir / f"{dataset}_subspace-split-pipeline.pkl"}')

    split_activity = subspace_split_pipeline.transform(neural_data)
    unsplit_activity = subspace_split_pipeline[:-1].transform(neural_data)
    fig = plot_split_subspace_variance(unsplit_activity, split_activity)
    fig.savefig(results_dir / f'{dataset}_subspace-split-variance.svg')
    logger.info(f'Saved subspace split variance plot to {results_dir / f"{dataset}_subspace-split-variance.svg"}')

def precondition_data(tf: pd.DataFrame)->pd.DataFrame:
    state_mapper = {
        'Control System': 'Go Cue',
        'Reach to Target 1': 'Go Cue',
        'Hold at Target 1': 'Go Cue', # Sometimes first reach state is skipped in this table (if the first target is in the center)
        'Reach to Target': 'Go Cue',
    }
    # task and result are already in the index from compose_from_frames
    neural_data = (
        tf
        .set_index(['state'],append=True)
        .pipe(tfr.multivalue_xs,keys=['CST','RTT','DCO'],level='task')
        .xs(level='result',key='success')
        .rename(index=state_mapper, level='state')
        .groupby('trial_id')
        .filter(lambda df: np.any(tfr.get_index_level(df,'state') == 'Go Cue'))
        .groupby('state')
        .filter(lambda df: df.name != 'Reach to Center')
        ['motor cortex']
    )

    return neural_data # type: ignore

if __name__=='__main__':
    parser = create_default_parser(
        description='Split neural population data into subspaces based on task conditions and save the results.',
    )
    parser.add_argument(
        '--num_components_per_cond',
        type=int,
        default=15,
        help='Number of components per condition for joint PCA',
    )
    parser.add_argument(
        '--dekodec_var_cutoff',
        type=float,
        default=0.99,
        help='Variance cutoff for DekODec subspace decomposition',
    )
    args,_ = parser.parse_known_args()
    main(args)