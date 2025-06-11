import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from src import munge, time_slice, crystal_models, timeseries, subspace_tools, io
from src.cli import with_parsed_args, create_default_parser
from sklearn.pipeline import make_pipeline
from dekodec import DekODec
import cloudpickle
from typing import Union,Optional
import logging
logger = logging.getLogger(__name__)

def main(args):
    dataset = args.dataset
    
    io.setup_logging(args, subfolder_name='subspace-split')
    results_dir = io.setup_results_dir(args, subfolder_name='subspace-split')
    
    tf = io.load_trial_frame(args)
    logger.info(f'Loaded trial frame from {dataset}')
    neural_data = precondition_data(tf)

    subspace_split_pipeline = make_pipeline(
        crystal_models.SoftnormScaler(),
        crystal_models.JointSubspace(
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
    neural_data = (
        tf
        .set_index(['task','result','state'],append=True)
        .pipe(munge.multivalue_xs,keys=['CST','RTT','DCO'],level='task')
        .xs(level='result',key='success')
        .rename(index=state_mapper, level='state')
        .groupby('trial_id')
        .filter(lambda df: np.any(munge.get_index_level(df,'state') == 'Go Cue'))
        .groupby('state')
        .filter(lambda df: df.name != 'Reach to Center')
        ['motor cortex']
    )

    return neural_data # type: ignore

def plot_split_subspace_variance(unsplit_activity, split_activity) -> Figure:
    neural_data = pd.concat(
        {'unsplit': unsplit_activity, 'split': split_activity},
        axis=1,
        names=['neural space', 'component']
    )

    compared_var = (
        neural_data
        .stack(level='neural space')
        .groupby(['task', 'neural space'])
        .apply(subspace_tools.calculate_fraction_variance)
        .stack(level='component')
        .to_frame(name='fraction variance')
    )

    g = sns.catplot(
        data=compared_var,
        x='component',
        y='fraction variance',
        hue='task',
        hue_order=['CST', 'RTT', 'DCO'],
        kind='bar',
        row='neural space',
        sharex=True,
        sharey=True,
        aspect=3,
        height=2,
    )
    return g.figure

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