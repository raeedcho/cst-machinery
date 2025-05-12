import pandas as pd
import smile_extract
import seaborn as sns
from matplotlib.figure import Figure
from src.cli import with_parsed_args

from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from src.crystal_models import SoftnormScaler
from src.munge import get_index_level, multivalue_xs
from src.time_slice import reindex_trial_from_event
from omegaconf import OmegaConf

@with_parsed_args(description='Find and plot the context axis of CST/RTT data. Output is an SVG plot.')
def main(args):
    dataset = args.dataset

    log_dir = Path(args.log_dir) / 'context_axis'
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(
        filename=log_dir / f'{dataset}.log',
        level=args.loglevel,
    )

    trialframe_dir = Path(args.trialframe_dir)
    composition_config = OmegaConf.load(args.composition_config)
    tf = smile_extract.compose_from_frames(
        meta=pd.read_parquet(trialframe_dir / dataset / f'{dataset}_{composition_config.info}.parquet'),
        trialframe_dict={
            key: pd.read_parquet(trialframe_dir / dataset / f'{dataset}_{filepart}.parquet')
            for key, filepart in composition_config.composition.items()
        }
    )
    data = process_trial_frame(tf)
    logger.info(f'Loaded trial frame from {dataset}')

    logger.info(f'Fitting LDA model to data')
    lda_pipe = fit_context_lda(data)

    output_path = Path(args.results_dir) / 'context_axis' / f'{dataset}_context_axis.svg'
    fig = plot_context_axis(data,lda_pipe)
    fig.savefig(output_path)
    logger.info(f'Wrote context axis plot to {output_path}')

def process_trial_frame(trialframe: pd.DataFrame) -> pd.DataFrame:
    state_mapper = {
        'Hold Center (CST Cue)': 'Hold Center (Task Cue)',
        'Hold Center (RTT Cue)': 'Hold Center (Task Cue)',
    }
    return (
        trialframe
        .set_index(['task','result','state'],append=True)
        ['motor cortex']
        .pipe(multivalue_xs,keys=['CST','RTT'],level='task') # type: ignore
        .xs(level='result',key='success')
        .rename(index=state_mapper,level='state')
        .groupby('trial_id',group_keys=False)
        .apply(reindex_trial_from_event,event='Hold Center (Task Cue)')
        .pipe(SoftnormScaler().fit_transform)
    )

def fit_context_lda(data: pd.DataFrame)->Pipeline:
    train_data = (
        data
        .loc[(slice(None),slice(None),slice('1 sec','3 sec')),:]
    )
    lda_pipe = Pipeline([
        ('svd', TruncatedSVD(n_components=15)),
        ('lda', LinearDiscriminantAnalysis()),
    ])
    lda_pipe.fit(train_data,get_index_level(train_data,'task'))
    return lda_pipe

def plot_context_axis(data: pd.DataFrame, lda_pipe: Pipeline) -> Figure:
    test_data = (
        data
        .loc[(slice(None),slice(None),slice('-1 sec','3 sec')),:]
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
    main() # type: ignore