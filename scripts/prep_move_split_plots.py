import cloudpickle
import json
import logging
import subprocess
import sys

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from trialframe import SoftnormScaler, get_epoch_data, multivalue_xs

from src.io import generic_preproc, get_targets, setup_logging, setup_results_dir
from src.targets import get_target_direction
from src.plot import plot_split_subspace_variance
from src.cli import create_default_parser

logger = logging.getLogger(__name__)


def save_figure(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)


def save_split_space_svg(chart, out_path):
    """Export an Altair chart to SVG via vl-convert in an isolated subprocess.

    Spawning a fresh interpreter prevents conflicts between vl-convert's embedded
    V8 engine and PyTorch (loaded later when deserializing the partition model).
    Falls back to saving interactive HTML if SVG export fails.

    Args:
        chart: Altair Chart object.
        out_path: Path for the .svg output (or .html on fallback).
    """
    convert_script = (
        "import sys, vl_convert as vlc; "
        "svg = vlc.vegalite_to_svg(sys.stdin.read()); sys.stdout.write(svg)"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', convert_script],
            input=json.dumps(chart.to_dict()),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout:
            out_path.write_text(result.stdout)
            logger.info(f'Saved SVG: {out_path}')
        else:
            raise RuntimeError(result.stderr[:500])
    except Exception as e:
        logger.warning(f'SVG export failed ({e}); saving as HTML instead')
        html_path = out_path.with_suffix('.html')
        chart.save(str(html_path))
        logger.info(f'Saved HTML fallback: {html_path}')


def main(args):
    dataset = args.dataset

    setup_logging(args, subfolder_name='prep-move-split-plots')
    results_dir = setup_results_dir(args, subfolder_name='prep-move-split')

    preproc = generic_preproc(args)
    targets = get_targets(args.trialframe_dir, dataset)
    target_dir = get_target_direction(targets)
    logger.info(f'Loaded preprocessed data and target directions for {dataset}')

    model_path = results_dir / f'{dataset}_prep-move-partition.pkl'
    with open(model_path, 'rb') as f:
        model_bundle = cloudpickle.load(f)

    softnormer: SoftnormScaler = model_bundle['softnormer']
    partition_model = model_bundle['partition_model']
    logger.info(f'Loaded fitted partition model from {model_path}')

    neural_data = (
        preproc
        .assign(**{'target direction': target_dir})
        .set_index('target direction', append=True)
        ['motor cortex']
        .pipe(softnormer.transform)
    )

    training_epochs = {
        'prep': ('Go Cue', slice(pd.to_timedelta(args.prep_start), pd.to_timedelta(args.prep_end))),
        'move': ('Move', slice(pd.to_timedelta(args.move_start), pd.to_timedelta(args.move_end))),
    }

    # Plot 1: split/unsplit variance comparison (from notebook cell 19 context)
    train_data = (
        neural_data
        .xs(level='task', key=args.reference_task)
        .pipe(get_epoch_data, epochs=training_epochs)
    )
    unsplit_activity = train_data @ partition_model.unsplit_projmat_
    split_activity = partition_model.transform_full(train_data)
    fig = plot_split_subspace_variance(
        unsplit_activity,
        split_activity,
        grouper='phase',
        group_order=['prep', 'move'],
    )
    save_figure(fig, results_dir / f'{dataset}_prep-move-variance.svg')

    # Plot 1b: split space trajectory (components 0 vs 1, notebook cell 13)
    split_space = (
        neural_data
        .pipe(get_epoch_data, epochs={
            'hold': ('Target On', slice(pd.to_timedelta('-200ms'), pd.to_timedelta('0ms'))),
            'trial': ('Go Cue', slice(pd.to_timedelta('-1000ms'),pd.to_timedelta('3000ms'))),
        })
        .pipe(partition_model.transform)
        .pipe(multivalue_xs, axis=1, level='component', keys=[0, 1])
        .stack(level='space')
        .rename(columns=lambda col: f'component {col}')
    )

    # Keep only the columns needed for plotting — drops extra index levels
    # (monkey, session date, state, etc.) to keep the Altair spec size manageable.
    _plot_cols = ['trial_id', 'time', 'task', 'space', 'phase', 'component 0', 'component 1']
    split_space_data = (
        split_space
        .reset_index()
        .assign(time=lambda df: df['time'].dt.total_seconds())
        [_plot_cols]
    )
    alt.data_transformers.disable_max_rows()
    split_space_chart = alt.Chart(split_space_data).mark_line().encode(
        x='component 0:Q',
        y='component 1:Q',
        color=alt.Color('phase:N').scale(scheme='set1').sort(['hold', 'trial']),
        row='space:N',
        column=alt.Column('task:N').sort(['DCO', 'RTT', 'CST']),
        detail='trial_id',
        order='time',
        opacity=alt.value(0.1),
    ).configure_axis(
        grid=False,
    ).configure_view(
        stroke=None,
    ).resolve_scale(
        x='independent',
        y='independent',
    )
    save_split_space_svg(split_space_chart, results_dir / f'{dataset}_split-space-trajectory.svg')

    # Shared RTT target-onset epochs (used by projected RTT panel from notebook cell 24)
    target_onset_slice = slice(pd.to_timedelta(args.target_onset_start), pd.to_timedelta(args.target_onset_end))
    rtt_epochs = {
        'targ 1': ('Go Cue', target_onset_slice),
        'targ 2': ('Reach to Target 2', target_onset_slice),
        'targ 3': ('Reach to Target 3', target_onset_slice),
        'targ 4': ('Reach to Target 4', target_onset_slice),
    }

    # Plot 2: projected DCO activity (notebook cell 19)
    projected_activity = (
        neural_data
        .xs(level='task', key=args.reference_task)
        .pipe(get_epoch_data, epochs={
            'prep': (
                'Target On',
                slice(pd.to_timedelta(args.proj_prep_start), pd.to_timedelta(args.proj_prep_end)),
            ),
            'move': (
                'Move',
                slice(pd.to_timedelta(args.proj_move_start), pd.to_timedelta(args.proj_move_end)),
            ),
        })
        .pipe(partition_model.transform)
        .xs(axis=1, level='component', key=0)
        .stack()
        .to_frame('activity')
        .sort_index(level=['phase', 'time'])
    )
    g = sns.relplot(
        data=projected_activity,
        x='time',
        y='activity',
        hue='target direction',
        col='phase',
        col_order=['prep', 'move'],
        row='space',
        row_order=['cis', 'move_unique', 'prep_unique'],
        kind='line',
        aspect=2,
        height=2,
        units='trial_id',
        estimator=None,
        alpha=0.3,
        facet_kws={
            'sharey': 'row',
        },
    )
    sns.despine(fig=g.figure, trim=True)
    g.refline(x=0, linestyle='--', color='k', linewidth=0.5)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    save_figure(g.figure, results_dir / f'{dataset}_projected-activity-dco.svg')

    # Plot 3: projected activity over full trial window for all tasks (notebook cell 20)
    all_projected = (
        neural_data
        .pipe(get_epoch_data, epochs={
            'trial': (
                'Go Cue',
                slice(pd.to_timedelta(args.all_trial_start), pd.to_timedelta(args.all_trial_end)),
            ),
        })
        .pipe(partition_model.transform)
        .xs(axis=1, level='component', key=0)
        .stack()
        .to_frame('activity')
        .sort_index(level=['phase', 'time'])
    )
    g = sns.relplot(
        data=all_projected,
        x='time',
        y='activity',
        hue='task',
        hue_order=['CST', 'RTT', 'DCO'],
        col='task',
        col_order=['DCO', 'RTT', 'CST'],
        row='space',
        row_order=['cis', 'move_unique', 'prep_unique', 'shared'],
        kind='line',
        aspect=2,
        height=2,
        units='trial_id',
        estimator=None,
        alpha=0.1,
        linewidth=1.5,
        facet_kws={
            'sharey': 'row',
        },
    )
    sns.despine(fig=g.figure, trim=True)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    save_figure(g.figure, results_dir / f'{dataset}_projected-activity-all-tasks.svg')

    # Plot 4: RTT projected activity aligned to target onsets (notebook cell 24)
    rtt_projections = (
        neural_data
        .xs(level='task', key='RTT')
        .pipe(get_epoch_data, epochs=rtt_epochs)
        .pipe(partition_model.transform)
        .xs(axis=1, level='component', key=0)
        .stack()
        .to_frame('activity')
        .sort_index(level=['phase', 'time'])
    )
    g = sns.relplot(
        data=rtt_projections.reset_index(),
        x='time',
        y='activity',
        col='phase',
        col_order=['targ 1', 'targ 2', 'targ 3', 'targ 4'],
        row='space',
        row_order=['cis', 'move_unique', 'prep_unique'],
        kind='line',
        aspect=1,
        height=2,
        color='tab:orange',
        units='trial_id',
        estimator=None,
        alpha=0.3,
        facet_kws={
            'margin_titles': True,
            'sharey': 'row',
        },
    )
    sns.despine(fig=g.figure, trim=True)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    save_figure(g.figure, results_dir / f'{dataset}_projected-activity-rtt-target-onset.svg')

    logger.info(f'Generated prep-move split plot suite for {dataset}')


if __name__ == '__main__':
    parser = create_default_parser(
        description='Generate prep/move split diagnostic and projected-activity plots.',
    )
    parser.add_argument('--reference_task', type=str, default='DCO')

    # Training epochs for split variance panel
    parser.add_argument('--prep_start', type=str, default='-300ms')
    parser.add_argument('--prep_end', type=str, default='0ms')
    parser.add_argument('--move_start', type=str, default='-100ms')
    parser.add_argument('--move_end', type=str, default='200ms')

    # RTT target-onset aligned projected panel (cell 24)
    parser.add_argument('--target_onset_start', type=str, default='-100ms')
    parser.add_argument('--target_onset_end', type=str, default='500ms')

    # DCO projected activity panel (cell 19)
    parser.add_argument('--proj_prep_start', type=str, default='-500ms')
    parser.add_argument('--proj_prep_end', type=str, default='500ms')
    parser.add_argument('--proj_move_start', type=str, default='-500ms')
    parser.add_argument('--proj_move_end', type=str, default='1000ms')

    # All-task projected activity panel (cell 20)
    parser.add_argument('--all_trial_start', type=str, default='-1000ms')
    parser.add_argument('--all_trial_end', type=str, default='3000ms')

    args, _ = parser.parse_known_args()
    main(args)
