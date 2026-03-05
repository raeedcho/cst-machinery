import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from trialframe import SoftnormScaler

from src.io import generic_preproc, setup_logging, setup_results_dir
from src.cli import create_default_parser

logger = logging.getLogger(__name__)


def main(args):
    dataset = args.dataset

    setup_logging(args, subfolder_name='neural-explained-variance')
    results_dir = setup_results_dir(args, subfolder_name='neural-explained-variance')

    preproc = generic_preproc(args)
    logger.info(f'Loaded preprocessed data for {dataset}')

    scaled = (
        preproc
        ['motor cortex']
        .pipe(SoftnormScaler().fit_transform)
    )
    state_data = scaled.xs(level='state', key=args.reference_state)

    task_order = ['CST', 'RTT', 'DCO']
    palette = {
        'CST': 'tab:blue',
        'RTT': 'tab:orange',
        'DCO': 'tab:green',
    }

    records = []
    dims95 = {}
    for task in task_order:
        if task not in state_data.index.get_level_values('task'):
            continue

        task_data = state_data.xs(level='task', key=task)
        cumsum = PCA().fit(task_data.values).explained_variance_ratio_.cumsum()

        dim_95 = int(np.searchsorted(cumsum, args.variance_threshold) + 1)
        dims95[task] = dim_95

        records.append(
            pd.DataFrame({
                'task': task,
                'component': np.arange(1, cumsum.shape[0] + 1),
                'cumulative explained variance': cumsum,
            })
        )

    expl_var = pd.concat(records, ignore_index=True)
    max_plot_component = max(args.min_plot_components, max(dims95.values()))
    plot_data = expl_var.query('component <= @max_plot_component')

    g = sns.relplot(
        data=plot_data,
        x='component',
        y='cumulative explained variance',
        kind='line',
        hue='task',
        hue_order=[task for task in task_order if task in dims95],
        palette=palette,
    )

    ax = g.axes[0, 0]
    ax.axhline(args.variance_threshold, linestyle='--', color='k')
    for task in task_order:
        if task in dims95:
            ax.axvline(dims95[task], linestyle='--', color=palette[task], alpha=0.8)
    sns.despine(trim=True)

    plot_path = results_dir / f'{dataset}_neural-explained-variance-curves.svg'
    g.figure.savefig(plot_path)
    plt.close(g.figure)

    dims_path = results_dir / f'{dataset}_neural-explained-variance-95pct-dims.csv'
    (
        pd.Series(dims95, name='n_components_95pct')
        .rename_axis('task')
        .to_csv(dims_path)
    )

    logger.info(f'Saved explained variance curves to {plot_path}')
    logger.info(f'Saved 95% dimension table to {dims_path}')


if __name__ == '__main__':
    parser = create_default_parser(
        description='Plot cumulative explained variance by task with dynamic 95% dimension markers.',
    )
    parser.add_argument('--reference_state', type=str, default='Move')
    parser.add_argument('--variance_threshold', type=float, default=0.95)
    parser.add_argument('--min_plot_components', type=int, default=15)

    args, _ = parser.parse_known_args()
    main(args)
