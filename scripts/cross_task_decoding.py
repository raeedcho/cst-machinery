import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import munge, time_slice, crystal_models, timeseries
import smile_extract
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
    cross_validate,
)
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.base import RegressorMixin,BaseEstimator

def main(args):
    input_path = Path(args.path)
    output_folder = Path(args.out)

    monkey = input_path.stem.split('_')[0]
    date = input_path.stem.split('_')[1]

    logging.basicConfig(
        filename=output_folder / f'{monkey}_{date}_cross-task-decoding.log',
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )

    tf = pd.read_parquet(input_path)
    neural_data, hand_data = precondition_data(tf)

    trial_predictions = decode_single_trials(neural_data,hand_data['x'])
    trial_scores = score_single_trials(hand_data['x'],trial_predictions)
    task_scores = score_tasks(hand_data['x'],trial_predictions)
    
    task_score_heatmap = sns.heatmap(
        data = (
            task_scores
            .unstack(level='model')
        ),
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={'fontsize': 21},
        cmap='gray',
    )

    scores_plot = sns.jointplot(
        data=(
            trial_scores
            .unstack(level='model')
        ),
        x='CST-trained',
        y='RTT-trained',
        hue='task',
        hue_order=['CST','RTT'],
        palette=['C0','C1'],
        xlim=(-1,1),
        ylim=(-1,1),
        marginal_ticks=False
    )
    scores_plot.plot_marginals(sns.rugplot,height=0.1,palette=['C0','C1'])
    scores_plot.refline(x=0,y=0)

    task_score_heatmap.figure.savefig(output_folder / f'{monkey}_{date}_decoder-task-score-heatmap.svg')
    scores_plot.figure.savefig(output_folder / f'{monkey}_{date}_decoder-trial-scores-scatter.svg')

    if not (output_folder / 'trials').exists():
        (output_folder / 'trials').mkdir()
    for trial_id in hand_data.index.get_level_values('trial_id').unique():
        trial_fig = plot_trial_predictions(
            true_data=hand_data.loc[trial_id,'x'],
            pred_data=trial_predictions.loc[trial_id,:],
        )
        trial_fig.savefig(output_folder / 'trials' / f'{monkey}_{date}_trial-{trial_id}_predictions.svg')
        plt.close(trial_fig)

def precondition_data(tf: pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    state_mapper = {
        'Control System': 'Go Cue',
        'Reach to Target 1': 'Go Cue',
        'Hold at Target 1': 'Go Cue', # Sometimes first reach state is skipped in this table (if the first target is in the center)
    }
    preproc = (
        tf
        .set_index(['task','result','state'],append=True)
        .pipe(munge.multivalue_xs,keys=['CST','RTT'],level='task')
        .xs(level='result',key='success')
        .rename(index=state_mapper, level='state')
        .groupby('trial_id', group_keys=False)
        .apply(time_slice.reindex_trial_from_event, event='Go Cue')
    )
    trim_pipeline = lambda df: (
        df
        .loc[(slice(None),slice('-0.5 sec','3sec')),:]
    )

    neural_data = (
        preproc
        ['motor cortex']
        .groupby('trial_id')
        .transform(smile_extract.smooth_data, dt=0.01,std=0.1,backend='convolve')
        .pipe(crystal_models.SoftnormScaler().fit_transform)
        .pipe(lambda df: pd.DataFrame(
            PCA(n_components=15).fit_transform(df),
            index=df.index,
        ))
        .pipe(trim_pipeline)
    )

    hand_data = (
        preproc
        ['hand position']
        .pipe(timeseries.estimate_kinematic_derivative, deriv=1, cutoff=30)
        .pipe(trim_pipeline)
    )

    return neural_data, hand_data

def decode_single_trials(predictor_data: pd.DataFrame, target_data: pd.Series)->pd.DataFrame:
    models = {
        'CST-trained': TaskTrainedDecoder(task='CST'),
        'RTT-trained': TaskTrainedDecoder(task='RTT'),
        'Dual': TaskTrainedDecoder(task=None),
    }

    trial_predictions = pd.DataFrame(
        {
            name: cross_val_predict(
                model,
                predictor_data,
                target_data,
                cv=LeaveOneGroupOut(),
                groups=target_data.index.get_level_values('trial_id'),
            )
            for name, model in models.items()
        },
        columns=pd.Index(list(models.keys()),name='model'),
        index=target_data.index,
    )

    return trial_predictions

def score_single_trials(true_data: pd.Series,pred_data: pd.DataFrame)->pd.Series:
    return score_groups(true_data,pred_data,['trial_id','task','model'])

def score_tasks(true_data: pd.Series,pred_data: pd.DataFrame)->pd.Series:
    return score_groups(true_data,pred_data,['task','model'])

def score_groups(true_data: pd.Series,pred_data: pd.DataFrame, grouper: str)->pd.Series:
    return (
        pred_data
        .stack(level='model')
        .to_frame('predicted')
        .assign(true=true_data)
        .groupby(grouper)
        .apply(lambda set: r2_score(set['true'],set['predicted']))
    )

def plot_trial_predictions(true_data: pd.Series,pred_data: pd.DataFrame)->plt.Figure:
    data = (
        pred_data
        .assign(true=true_data)
        .stack(level='model')
        .to_frame('hand velocity')
    )
    task = munge.get_index_level(data,'task').unique()
    assert len(task)==1, 'Data contains multiple trials'

    fig,ax = plt.subplots(figsize=(6,3))
    sns.lineplot(
        data=data,
        x='time',
        y='hand velocity',
        hue='model',
        hue_order=['true','Dual','CST-trained','RTT-trained'],
        palette=['k','0.5','C0','C1'],
        ax=ax,
    )
    ax.set(
        xlabel='time (s)',
        ylabel='hand velocity (cm/s)',
        title=munge.get_index_level(data,'task').unique(),    
    )
    ax.axvline(0, color='k', linestyle='--')
    sns.despine(ax=ax,trim=True)
    
    return fig

class TaskTrainedDecoder(BaseEstimator,RegressorMixin):
    """
    A linear regression model trained on data from only a single task.
    """
    def __init__(self,task=None,**kwargs):
        self.task = task
        self.model = LinearRegression(**kwargs)
    
    def fit(self,X,y,**kwargs):
        if isinstance(y,pd.Series):
            y = y.to_frame()

        self.output_columns = y.columns

        if self.task is not None:
            X_task = X.xs(level='task',key=self.task)
            y_task = y.xs(level='task',key=self.task)
        else:
            X_task = X
            y_task = y

        self.model.fit(X_task,y_task)
        return self

    def predict(self,X,**kwargs):
        return (
            pd.DataFrame(self.model.predict(X),index=X.index,columns=self.output_columns)
            .squeeze()
        )
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Cross-task decoding analysis--train on one task and test on the other.')
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