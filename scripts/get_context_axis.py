import pandas as pd
import smile_extract
import src
import seaborn as sns
import matplotlib as mpl

from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from src.crystal_models import SoftnormScaler

def main(args):
    input_path = Path(args.path)
    output_path = Path(args.out)

    logging.basicConfig(
        filename=output_path.with_suffix('.log'),
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )

    data = process_trial_frame(input_path)
    logger.info(f'Loaded data from {input_path}')
    logger.info(f'Fitting LDA model to data')

    lda_pipe = fit_context_lda(data)
    fig = plot_context_axis(data,lda_pipe)
    fig.savefig(output_path)
    
    logger.info(f'Wrote context axis plot to {output_path}')

def process_trial_frame(input_path: Path) -> pd.DataFrame:
    state_mapper = {
        'Hold Center (CST Cue)': 'Hold Center (Task Cue)',
        'Hold Center (RTT Cue)': 'Hold Center (Task Cue)',
    }
    return (
        pd.read_pickle(input_path)
        .set_index(['block','task','result','state'],append=True)
        ['motor cortex']
        .pipe(src.munge.multivalue_xs,keys=['CST','RTT'],level='task')
        .xs(level='result',key='success')
        .rename(index=state_mapper,level='state')
        .groupby('trial_id',group_keys=False)
        .apply(src.time_slice.reindex_trial_from_event,event='Hold Center (Task Cue)')
        .groupby('trial_id')
        .transform(smile_extract.smooth_data,dt=0.01,std=0.1,backend='convolve')
        .pipe(SoftnormScaler().fit_transform)
    )

def fit_context_lda(data: pd.DataFrame)->Pipeline:
    train_data = (
        data
        .loc[(slice(None),slice('1 sec','3 sec')),:]
    )
    lda_pipe = Pipeline([
        ('svd', TruncatedSVD(n_components=15)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    lda_pipe.fit(train_data,src.munge.get_index_level(train_data,'task'))
    return lda_pipe

def plot_context_axis(data: pd.DataFrame, lda_pipe: Pipeline) -> mpl.figure.Figure:
    test_data = (
        data
        .loc[(slice(None),slice('-1 sec','3 sec')),:]
    )

    output = pd.DataFrame(
        lda_pipe.transform(test_data),
        index=test_data.index,
        columns=['LDA projection'],
    )

    g = sns.relplot(
        output,
        x='time',
        y='LDA projection',
        hue='task',
        hue_order=['CST','RTT'],
        kind='line',
        height=3,
        aspect=1.5,
        errorbar=None,
        estimator=None,
        units='trial_id',
        lw=0.5,
        alpha=0.1,
    )
    sns.lineplot(
        (
            output
            .groupby(['task','time'])
            .mean()
        ),
        x='time',
        y='LDA projection',
        hue='task',
        hue_order=['CST','RTT'],
        lw=3,
        alpha=1,
        ax=g.ax,
    )
    g.set_axis_labels('Time from task cue(s)','LDA Projection')
    g.refline(x=0,linestyle='--')
    sns.despine(fig=g.figure,trim=True)
    return g.figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and plot the context axis of CST/RTT data. Output is an SVG plot.')
    parser.add_argument(
        'path',
        type=str,
        help='Path to the trial frame data file',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Path to the output directory',
        required=True,
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )
    
    args = parser.parse_args()
    main(args)