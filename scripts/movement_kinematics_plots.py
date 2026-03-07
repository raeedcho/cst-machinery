import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from trialframe import get_epoch_data

from src.io import generic_preproc, setup_logging, setup_results_dir
from src.cli import create_default_parser

logger = logging.getLogger(__name__)


def save_figure(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)


def main(args):
    dataset = args.dataset

    setup_logging(args, subfolder_name='movement-kinematics-plots')
    results_dir = setup_results_dir(args, subfolder_name='movement-kinematics')

    preproc = generic_preproc(args)
    logger.info(f'Loaded preprocessed data for {dataset}')

    # Plot 1: hand movements by task (notebook cell 6)
    hand_movements = (
        preproc
        .pipe(get_epoch_data, epochs={
            'trial': (
                'Go Cue',
                slice(pd.to_timedelta(args.hand_trial_start), pd.to_timedelta(args.hand_trial_end)),
            ),
        })
        .xs(axis=1, level='channel', key='x')
        .stack(future_stack=True)
        .to_frame('hand kinematics')
    )
    g = sns.relplot(
        data=hand_movements,
        x='time',
        y='hand kinematics',
        row='signal',
        col='task',
        col_order=['DCO', 'RTT', 'CST'],
        hue='task',
        hue_order=['CST', 'RTT', 'DCO'],
        kind='line',
        aspect=2,
        height=2,
        units='trial_id',
        estimator=None,
        alpha=0.1,
        facet_kws=dict(
            margin_titles=True,
            sharey='row',
        ),
    )
    sns.despine(fig=g.figure, trim=True)
    save_figure(g.figure, results_dir / f'{dataset}_hand-movements.svg')

    # Plot 2: RTT kinematics aligned to each target onset (notebook cell 9)
    target_onset_slice = slice(pd.to_timedelta(args.target_onset_start), pd.to_timedelta(args.target_onset_end))
    rtt_epochs = {
        'targ 1': ('Go Cue', target_onset_slice),
        'targ 2': ('Reach to Target 2', target_onset_slice),
        'targ 3': ('Reach to Target 3', target_onset_slice),
        'targ 4': ('Reach to Target 4', target_onset_slice),
    }
    rtt_horz_kin = (
        preproc
        .xs(level='task', key='RTT')
        .pipe(get_epoch_data, epochs=rtt_epochs)
        .xs(axis=1, level='channel', key='x')
        .stack(future_stack=True)
        .to_frame('kinematics')
        .sort_index(level=['phase', 'time'])
    )
    g = sns.relplot(
        data=rtt_horz_kin.reset_index(),
        x='time',
        y='kinematics',
        col='phase',
        col_order=['targ 1', 'targ 2', 'targ 3', 'targ 4'],
        row='signal',
        row_order=['hand position', 'hand velocity', 'hand acceleration'],
        kind='line',
        aspect=1,
        height=2,
        color='tab:orange',
        units='trial_id',
        estimator=None,
        alpha=0.3,
        facet_kws=dict(
            margin_titles=True,
            sharey='row',
        ),
    )
    sns.despine(fig=g.figure, trim=True)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    save_figure(g.figure, results_dir / f'{dataset}_rtt-target-onset-kinematics.svg')

    logger.info(f'Generated movement kinematics plot suite for {dataset}')


if __name__ == '__main__':
    parser = create_default_parser(
        description='Generate task and RTT target-onset aligned movement kinematics plots.',
    )
    parser.add_argument('--hand_trial_start', type=str, default='-500ms')
    parser.add_argument('--hand_trial_end', type=str, default='3000ms')
    parser.add_argument('--target_onset_start', type=str, default='-100ms')
    parser.add_argument('--target_onset_end', type=str, default='500ms')

    args, _ = parser.parse_known_args()
    main(args)
