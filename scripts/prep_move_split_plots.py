import cloudpickle
import json
import logging

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from trialframe import get_epoch_data

from src.io import generic_preproc, get_targets, setup_logging, setup_results_dir
from src.targets import get_target_direction
from src.plot import plot_split_subspace_variance
from src.cli import create_default_parser

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def save_figure(fig, out_path):
    """Save a matplotlib figure and close it to free memory."""
    fig.savefig(out_path)
    plt.close(fig)


def save_vegalite_spec(chart, out_path):
    """Save an Altair chart as a Vega-Lite JSON spec.

    The spec is converted to SVG in a separate DVC stage (prep_move_split_svg)
    by vegalite_to_svg.py, which runs in a fresh process without torch loaded.

    Args:
        chart: Altair Chart object.
        out_path: Path for the .json output.
    """
    out_path.write_text(json.dumps(chart.to_dict()))


# -----------------------------------------------------------------------------
# Data/model prep helpers
# -----------------------------------------------------------------------------
def load_partition_pipeline(results_dir, dataset):
    """Load fitted sklearn pipeline (softnorm + partition) from disk."""
    model_path = results_dir / f'{dataset}_prep-move-partition.pkl'
    with open(model_path, 'rb') as f:
        pipeline = cloudpickle.load(f)
    logger.info(f'Loaded fitted pipeline from {model_path}')
    return pipeline


def prepare_neural_data(preproc):
    """Isolate motor cortex data."""
    return preproc['motor cortex']


def assign_target_direction(data, target_dir):
    # Attach target direction to index
    return (
        data
        .assign(**{'target direction': target_dir})
        .set_index('target direction', append=True)
    )

# -----------------------------------------------------------------------------
# Plot panel helpers
# -----------------------------------------------------------------------------
def plot_split_variance_panel(neural_data, transformed_data, pipeline, args, results_dir, dataset):
    """Generate prep/move variance explained plot comparing unsplit vs split activity."""
    training_epochs = {
        'prep': ('Go Cue', slice(pd.to_timedelta(args.prep_start), pd.to_timedelta(args.prep_end))),
        'move': ('Move', slice(pd.to_timedelta(args.move_start), pd.to_timedelta(args.move_end))),
    }

    partition_model = pipeline.named_steps['cisdirpartition']
    train_data_raw = (
        neural_data
        .xs(level='task', key=args.reference_task)
        .pipe(get_epoch_data, epochs=training_epochs)
    )
    train_data = pipeline.named_steps['softnormscaler'].transform(train_data_raw)
    unsplit_activity = train_data @ partition_model.unsplit_projmat_
    unsplit_activity.columns = pd.RangeIndex(unsplit_activity.shape[1], name='component')

    split_activity = (
        transformed_data
        .xs(level='task', key=args.reference_task)
        .pipe(get_epoch_data, epochs=training_epochs)
    )
    if isinstance(split_activity.columns, pd.MultiIndex):
        split_activity = split_activity.sort_index(axis=1)
        split_activity.columns = pd.RangeIndex(split_activity.shape[1], name='component')

    fig = plot_split_subspace_variance(
        unsplit_activity,
        split_activity,
        grouper='phase',
        group_order=['prep', 'move'],
    )
    save_figure(fig, results_dir / f'{dataset}_prep-move-variance.svg')


def build_split_space_data(transformed_data):
    """Build long-form split-space component dataframe for trajectory plotting."""
    long_df = (
        transformed_data
        .pipe(get_epoch_data, epochs={
            'hold': ('Target On', slice(pd.to_timedelta('-200ms'), pd.to_timedelta('0ms'))),
            'trial': ('Go Cue', slice(pd.to_timedelta('-1000ms'), pd.to_timedelta('3000ms'))),
        })
        .stack(level='space', future_stack=True)  # Move space from columns to index
        [[0, 1]]  # Select only first two components
        .rename(columns=lambda x: f'component {x}')  # Rename component columns
        .reset_index()
        .assign(time=lambda df: df['time'].dt.total_seconds())
    )

    plot_cols = ['trial_id', 'time', 'task', 'space', 'phase', 'target direction', 'component 0', 'component 1']
    return long_df[plot_cols]


def apply_split_space_colors(split_space_data):
    """Add custom color groups for hold/trial and DCO direction shades."""
    split_space_data = split_space_data.copy()
    split_space_data['color_group'] = 'hold'

    trial_mask = split_space_data['phase'] == 'trial'
    split_space_data.loc[trial_mask & (split_space_data['task'] == 'RTT'), 'color_group'] = 'RTT trial'
    split_space_data.loc[trial_mask & (split_space_data['task'] == 'CST'), 'color_group'] = 'CST trial'

    dco_trial_mask = trial_mask & (split_space_data['task'] == 'DCO')
    split_space_data.loc[dco_trial_mask, 'color_group'] = (
        'DCO trial ' + split_space_data.loc[dco_trial_mask, 'target direction'].astype(str)
    )

    dco_groups = sorted(
        split_space_data.loc[dco_trial_mask, 'color_group']
        .dropna()
        .unique()
        .tolist()
    )

    if dco_groups:
        if len(dco_groups) == 1:
            dco_colors = ['#2ca02c']
        else:
            dco_colors = sns.light_palette('#2ca02c', n_colors=len(dco_groups) + 2).as_hex()[2:]
    else:
        dco_colors = []

    color_domain = ['hold', 'RTT trial', 'CST trial'] + dco_groups
    color_range = ['#b3b3b3', '#ff7f0e', '#1f77b4'] + dco_colors
    return split_space_data, color_domain, color_range


def plot_split_space_trajectory_panel(transformed_data, results_dir, dataset):
    """Create and save split-space trajectory Vega-Lite spec."""
    split_space_data = build_split_space_data(transformed_data)
    split_space_data, color_domain, color_range = apply_split_space_colors(split_space_data)

    alt.data_transformers.disable_max_rows()
    split_space_chart = alt.Chart(split_space_data).mark_line().encode(
        x='component 0:Q',
        y='component 1:Q',
        color=alt.Color('color_group:N').scale(domain=color_domain, range=color_range),
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
    save_vegalite_spec(split_space_chart, results_dir / f'{dataset}_split-space-trajectory.json')


def get_rtt_epochs(args):
    """Build RTT target-onset aligned epoch definitions from CLI args."""
    target_onset_slice = slice(pd.to_timedelta(args.target_onset_start), pd.to_timedelta(args.target_onset_end))
    return {
        'targ 1': ('Go Cue', target_onset_slice),
        'targ 2': ('Reach to Target 2', target_onset_slice),
        'targ 3': ('Reach to Target 3', target_onset_slice),
        'targ 4': ('Reach to Target 4', target_onset_slice),
    }


def plot_projected_dco_panel(transformed_data, args, results_dir, dataset):
    """Create and save projected DCO activity panel."""
    projected_activity = (
        transformed_data
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
        .xs(axis=1, level='component', key=0)
        .stack(future_stack=True)
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


def plot_projected_all_tasks_panel(transformed_data, args, results_dir, dataset):
    """Create and save projected activity panel across all tasks."""
    all_projected = (
        transformed_data
        .pipe(get_epoch_data, epochs={
            'trial': (
                'Go Cue',
                slice(pd.to_timedelta(args.all_trial_start), pd.to_timedelta(args.all_trial_end)),
            ),
        })
        .xs(axis=1, level='component', key=0)
        .stack(future_stack=True)
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


def plot_rtt_target_onset_panel(transformed_data, rtt_epochs, results_dir, dataset):
    """Create and save RTT target-onset aligned projected activity panel."""
    rtt_projections = (
        transformed_data
        .xs(level='task', key='RTT')
        .pipe(get_epoch_data, epochs=rtt_epochs)
        .xs(axis=1, level='component', key=0)
        .stack(future_stack=True)
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


def main(args):
    """Generate the full prep-move split plotting suite for a dataset."""
    dataset = args.dataset

    setup_logging(args, subfolder_name='prep-move-split-plots')
    results_dir = setup_results_dir(args, subfolder_name='prep-move-split')

    targets = get_targets(args.trialframe_dir, dataset)
    target_dir = get_target_direction(targets)
    full_data = (
        generic_preproc(
            args,
            add_kinematics=False,
        )
        .pipe(assign_target_direction, target_dir=target_dir)
    )
    orig_data = full_data['motor cortex']
    split_data = full_data[['cis', 'move_unique', 'prep_unique', 'shared']].copy()
    split_data.columns = split_data.columns.set_names(['space', 'component'])
    logger.info(f'Loaded preprocessed data and target directions for {dataset}')

    pipeline = load_partition_pipeline(results_dir, dataset)

    plot_split_variance_panel(orig_data, split_data, pipeline, args, results_dir, dataset)
    plot_split_space_trajectory_panel(split_data, results_dir, dataset)
    plot_projected_dco_panel(split_data, args, results_dir, dataset)
    plot_projected_all_tasks_panel(split_data, args, results_dir, dataset)

    rtt_epochs = get_rtt_epochs(args)
    plot_rtt_target_onset_panel(split_data, rtt_epochs, results_dir, dataset)

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
